# main pipeline connecting hrnet keypoint detection, uncertainty propagation, and ibc rule evaluation

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from app.perception.geometry import AffineTransform
from app.analytical.measurement_factor import load_tsi_scales, load_uncertainty_scales
from app.analytical.gum_propagation import (
    assemble_block_covariance,
    combined_uncertainty,
    expanded_uncertainty,
)
from app.analytical.jacobian import numerical_jacobian
from app.analytical.morphometrics import MORPHOMETRIC_FUNCTIONS
from app.analytical.tsi import compute_tsi, predicted_keypoint_sigma
from app.decisional.abstention_gate import CriterionResult, evaluate_criterion
from app.decisional.ibc_standards import CRITERION_THRESHOLDS
from app.perception.heatmap import soft_argmax
from app.perception.hrnet import HRNetKeypointDetector


@dataclass
class PipelineOutput:
    # holds full output for a single image in the target coordinate space

    criteria: list[CriterionResult]
    keypoints: np.ndarray  # (K, 2)
    covariances: np.ndarray  # (K, 2, 2)


class AssessmentPipeline:
    # sets up model on device and loads the per-criterion uncertainty scales
    def __init__(
        self,
        model: HRNetKeypointDetector,
        device: str = "cpu",
        uncertainty_scales: dict[str, float] | None = None,
        tsi_scales: dict[str, float] | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        # per-criterion post-hoc U scale; None -> read the measurement_factor block of the config, {} -> all 1.0
        self.uncertainty_scales = load_uncertainty_scales() if uncertainty_scales is None else uncertainty_scales
        # per-criterion scale on sigma_hat for the TSI gate, fitted on val, same measurement_factor block
        self.tsi_scales = load_tsi_scales() if tsi_scales is None else tsi_scales

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
        image_on_device = image_tensor.to(self.device)
        heatmaps, covariances = self.model(image_on_device)
        mu_crop_space = HRNetKeypointDetector.mu_to_crop_space(soft_argmax(heatmaps))
        means = mu_crop_space[0].cpu().numpy()
        per_keypoint_covariances = covariances[0].cpu().numpy()

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

            sigma_hat = predicted_keypoint_sigma(jacobian, per_keypoint_covariances) * self.tsi_scales.get(criterion_key, 1.0)

            results.append(evaluate_criterion(criterion_key, measurement, threshold, uncertainty, tsi, sigma_hat))

        return PipelineOutput(criteria=results, keypoints=means, covariances=per_keypoint_covariances)