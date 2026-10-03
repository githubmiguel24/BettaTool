# main pipeline connecting hrnet keypoint detection, uncertainty propagation, and ibc rule evaluation

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from app.perception.geometry import AffineTransform
from app.analytical.calibration import load_uncertainty_scales
from app.analytical.gum_propagation import (
    assemble_block_covariance,
    combined_uncertainty,
    expanded_uncertainty,
)
from app.analytical.jacobian import numerical_jacobian
from app.analytical.morphometrics import MORPHOMETRIC_FUNCTIONS
from app.analytical.tsi import compute_tsi
from app.decisional.abstention_gate import CriterionResult, evaluate_criterion
from app.decisional.ibc_standards import CRITERION_THRESHOLDS
from app.perception.heatmap import batch_heatmaps_to_gaussians, soft_argmax
from app.perception.hrnet import HRNetKeypointDetector


@dataclass
class PipelineOutput:
    # holds full output for a single image in the target coordinate space

    criteria: list[CriterionResult]
    keypoints: np.ndarray  # (K, 2)
    covariances: np.ndarray  # (K, 2, 2)
    visibility: np.ndarray  # (K,) probabilities in [0, 1]
    low_visibility_keypoints: list[int]


class AssessmentPipeline:
    # sets up model on device and sets cutoff for keypoint visibility
    def __init__(
        self,
        model: HRNetKeypointDetector,
        device: str = "cpu",
        visibility_threshold: float = 0.5,
        uncertainty_scales: dict[str, float] | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.visibility_threshold = visibility_threshold
        # per-criterion post-hoc U scale; None -> read calibration.json if present, {} -> all 1.0
        self.uncertainty_scales = load_uncertainty_scales() if uncertainty_scales is None else uncertainty_scales

    @torch.no_grad()
    def analyze(self, image_tensor: torch.Tensor) -> list[CriterionResult]:
        # wrapper around analyze_detailed that just returns the criteria list
        return self.analyze_detailed(image_tensor).criteria

    @torch.no_grad()
    def analyze_detailed(
        self,
        image_tensor: torch.Tensor,
        to_original: AffineTransform | None = None,
    ) -> PipelineOutput:
        # runs the image tensor through detection, uncertainty math, and threshold checks
        model_out = self.model(image_tensor.to(self.device))

        if len(model_out) == 3:
            # unpack 3-tuple output and use soft argmax for means in crop space
            heatmaps, covariances, visibility_logits = model_out
            visibility_probs = torch.sigmoid(visibility_logits)[0].cpu().numpy()
            mu_heatmap_space = soft_argmax(heatmaps)
            mu_crop_space = HRNetKeypointDetector.mu_to_crop_space(mu_heatmap_space)
            means = mu_crop_space[0].cpu().numpy()
            per_keypoint_covariances = covariances[0].cpu().numpy()
        else:
            # fallback for older 2-tuple models without a visibility head
            heatmaps, legacy_covariances = model_out
            heatmaps_np = heatmaps[0].cpu().numpy()
            legacy_covariances_np = legacy_covariances[0].cpu().numpy()
            means, heatmap_covariances = batch_heatmaps_to_gaussians(heatmaps_np)
            per_keypoint_covariances = legacy_covariances_np if legacy_covariances_np.size else heatmap_covariances
            visibility_probs = np.ones(means.shape[0])  # assume all visible when head is missing

        # flag keypoints that fall below the visibility threshold
        low_visibility_keypoints = [i for i, p in enumerate(visibility_probs) if p < self.visibility_threshold]

        # map coordinates and covariances back to Orignal image space before running gum math
        if to_original is not None:
            means = to_original.apply_points(means)
            per_keypoint_covariances = to_original.apply_covariances(per_keypoint_covariances)

        flat_keypoints = means.reshape(-1)
        block_covariance = assemble_block_covariance(per_keypoint_covariances)

        results: list[CriterionResult] = []
        # evaluate each morphometric criterion and propagate uncertainty
        for criterion_key, measurement_fn in MORPHOMETRIC_FUNCTIONS.items():
            jacobian = numerical_jacobian(measurement_fn, flat_keypoints)
            measurement = measurement_fn(flat_keypoints)

            u_c = combined_uncertainty(jacobian, block_covariance)
            uncertainty = expanded_uncertainty(u_c) * self.uncertainty_scales.get(criterion_key, 1.0)

            threshold = CRITERION_THRESHOLDS[criterion_key]
            tsi = compute_tsi(measurement, threshold, jacobian)

            actual_rmse = float(np.sqrt(np.mean(np.diag(block_covariance))))

            result = evaluate_criterion(criterion_key, measurement, threshold, uncertainty, tsi, actual_rmse)
            result.low_visibility_keypoints = low_visibility_keypoints
            results.append(result)

        return PipelineOutput(
            criteria=results,
            keypoints=means,
            covariances=per_keypoint_covariances,
            visibility=visibility_probs,
            low_visibility_keypoints=low_visibility_keypoints,
        )