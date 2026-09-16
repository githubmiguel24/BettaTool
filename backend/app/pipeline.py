"""End-to-end pipeline: image -> Reliability-Annotated Assessment Report.

Wires the three tiers from Figure 6 together:
Perceptual (HRNet) -> Analytical (GUM propagation + TSI) -> Decisional
(IBC rule engine + selective abstention gate).

MIGRATION NOTE (Build Prompt v2 §4.3, §13.2): `HRNetKeypointDetector.forward`
now returns a 3-tuple `(heatmaps, covariances, visibility)` instead of the
old 2-tuple `(heatmaps, covariances)`. This module has been updated to
consume all three — the visibility head's output is NOT silently dropped
(as the spec explicitly warns against): a keypoint whose predicted
visibility probability falls below `visibility_threshold` is now flagged in
each `CriterionResult` via `low_visibility_keypoints`, which
`app/decisional/abstention_gate.py` does not yet act on. Wiring that into an
actual abstention/defer decision (e.g. "defer if a criterion depends on a
low-visibility landmark") is flagged as follow-up work — see
training/README.md, "Known limitations" §4.

This module also now uses `soft_argmax` (not the old moment-based
`batch_heatmaps_to_gaussians`) as the mean-extraction path when the model
provides a covariance head, per the units contract in
`app/perception/geometry.py` — Sigma from the covariance head is in
CROP-SPACE pixel units and must be transformed to original-image units
before the GUM tier consumes it. Since this module receives an already-
cropped/letterboxed tensor with no affine transform of its own (the caller
owns cropping), it treats its input tensor's pixel space as the reporting
space; a caller that crops before calling `analyze` is responsible for
composing that crop's affine with `app/perception/geometry.py` if it needs
original-image-pixel output. This is an explicit scope boundary, not an
oversight — seeing it wired end-to-end needs the FastAPI upload route
(`app/api/routes/analyze.py`) to own the crop, which is still unwired to a
trained checkpoint (see README "Status").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from app.perception.geometry import AffineTransform
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
    """Everything one image produces, in a single reporting coordinate space.

    `keypoints` is (K, 2) and `covariances` is (K, 2, 2), both in whatever
    space the caller asked for via `analyze_detailed(to_original=...)`:
    crop-space pixels when that argument is None, original-image pixels when
    an inverse-crop affine is supplied. `criteria` holds one
    `CriterionResult` per IBC criterion, computed in that same space.
    """

    criteria: list[CriterionResult]
    keypoints: np.ndarray  # (K, 2)
    covariances: np.ndarray  # (K, 2, 2)
    visibility: np.ndarray  # (K,) probabilities in [0, 1]
    low_visibility_keypoints: list[int]


class AssessmentPipeline:
    def __init__(self, model: HRNetKeypointDetector, device: str = "cpu", visibility_threshold: float = 0.5) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.visibility_threshold = visibility_threshold

    @torch.no_grad()
    def analyze(self, image_tensor: torch.Tensor) -> list[CriterionResult]:
        """Runs one image (1, 3, H, W) through all three tiers.

        Returns one `CriterionResult` per IBC criterion. Thin wrapper over
        `analyze_detailed` kept for backwards compatibility with callers
        that only want the decisional-tier output.
        """
        return self.analyze_detailed(image_tensor).criteria

    @torch.no_grad()
    def analyze_detailed(
        self,
        image_tensor: torch.Tensor,
        to_original: AffineTransform | None = None,
    ) -> PipelineOutput:
        """Runs one image (1, 3, H, W) through all three tiers, keeping the
        perceptual-tier output alongside the decisional-tier verdicts.

        Args:
            image_tensor: (1, 3, H, W) preprocessed crop.
            to_original: optional affine mapping CROP space -> ORIGINAL image
                pixels (i.e. the inverse of the preprocessing letterbox). When
                supplied, means AND covariances are transformed into original-
                image pixels *before* the analytical tier runs, so the GUM
                propagation and the TSI/abstention comparison against keypoint
                RMSE all happen in the space the units contract
                (`app/perception/geometry.py`) designates for them. Passing
                None keeps everything in crop space, which is what the older
                `analyze()` callers assumed.
        """
        model_out = self.model(image_tensor.to(self.device))

        if len(model_out) == 3:
            heatmaps, covariances, visibility_logits = model_out
            visibility_probs = torch.sigmoid(visibility_logits)[0].cpu().numpy()
            mu_heatmap_space = soft_argmax(heatmaps)
            mu_crop_space = HRNetKeypointDetector.mu_to_crop_space(mu_heatmap_space)
            means = mu_crop_space[0].cpu().numpy()
            per_keypoint_covariances = covariances[0].cpu().numpy()
        else:
            # Backward-compat path for an old 2-tuple checkpoint/model (Build
            # Prompt v1 contract). Falls back to the moment-based heatmap
            # covariance, since there is no explicit covariance head output
            # to trust in that case.
            heatmaps, legacy_covariances = model_out
            heatmaps_np = heatmaps[0].cpu().numpy()
            legacy_covariances_np = legacy_covariances[0].cpu().numpy()
            means, heatmap_covariances = batch_heatmaps_to_gaussians(heatmaps_np)
            per_keypoint_covariances = legacy_covariances_np if legacy_covariances_np.size else heatmap_covariances
            visibility_probs = np.ones(means.shape[0])  # no visibility head available; assume all visible

        low_visibility_keypoints = [i for i, p in enumerate(visibility_probs) if p < self.visibility_threshold]

        # Move into the reporting space BEFORE the analytical tier, so that
        # propagated uncertainty and the pixel-denominated TSI comparison are
        # expressed in the same units the caller reports to the user.
        if to_original is not None:
            means = to_original.apply_points(means)
            per_keypoint_covariances = to_original.apply_covariances(per_keypoint_covariances)

        flat_keypoints = means.reshape(-1)
        block_covariance = assemble_block_covariance(per_keypoint_covariances)

        results: list[CriterionResult] = []
        for criterion_key, measurement_fn in MORPHOMETRIC_FUNCTIONS.items():
            jacobian = numerical_jacobian(measurement_fn, flat_keypoints)
            measurement = measurement_fn(flat_keypoints)

            u_c = combined_uncertainty(jacobian, block_covariance)
            uncertainty = expanded_uncertainty(u_c)

            threshold = CRITERION_THRESHOLDS[criterion_key]
            tsi = compute_tsi(measurement, threshold, jacobian)

            actual_rmse = float(np.sqrt(np.mean(np.diag(block_covariance))))

            result = evaluate_criterion(criterion_key, measurement, uncertainty, tsi, actual_rmse)
            result.low_visibility_keypoints = low_visibility_keypoints
            results.append(result)

        return PipelineOutput(
            criteria=results,
            keypoints=means,
            covariances=per_keypoint_covariances,
            visibility=visibility_probs,
            low_visibility_keypoints=low_visibility_keypoints,
        )
