# horizontal-flip test-time augmentation shared by the deployed pipeline (same maths as training/evaluate.py)

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def flip_averaged_heatmaps(model: torch.nn.Module, image: torch.Tensor, model_out: tuple | None = None) -> tuple:
    """Average the heatmaps of the image and its mirror image, then renormalise (Build Prompt v2 §9.1).

    The flip keypoint permutation is the identity, so only the spatial flip-back is needed. The flipped heatmap is
    shifted one column with ZERO padding (not torch.roll, which wraps the last column round to column 0 and drags
    right-edge keypoints, e.g. the snout of right-facing fish, toward the left edge).

    Covariance and visibility stay those of the ORIGINAL pass, as in training/evaluate.py, because the scale factors
    in calibration.json were fitted that way.

    Returns (averaged_heatmaps, covariances, visibility_logits). `model_out` may carry an already computed
    original-image forward pass to avoid running it twice.
    """
    heatmaps, covariances, visibility_logits = model_out if model_out is not None else model(image)
    flipped_heatmaps, _, _ = model(torch.flip(image, dims=[-1]))
    flipped_back = F.pad(torch.flip(flipped_heatmaps, dims=[-1]), (1, 0))[..., :-1]
    averaged = 0.5 * (heatmaps + flipped_back)
    averaged = averaged / averaged.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    return averaged, covariances, visibility_logits
