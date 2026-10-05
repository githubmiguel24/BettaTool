"""Heatmap loss as a KL divergence to a sum-normalised Gaussian target.

The predicted heatmap is a spatial softmax (sums to 1), but the MSE targets in `targets.py` have peak 1.0
(sum ~ 25), so with MSE the shape of the target (its sigma, or an elongated shape) hardly matters. Normalising
the target to a distribution and using KL makes the predicted heatmap match the target SHAPE.
"""

from __future__ import annotations

import torch


def heatmap_kl_loss(pred_heatmaps: torch.Tensor, target_heatmaps: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """KL(target || pred), summed over pixels, weight-averaged over keypoints, then averaged over the batch.

    Args:
        pred_heatmaps: (B, K, H, W) spatial-softmax heatmaps.
        target_heatmaps: (B, K, H, W) rendered targets (peak 1.0); all-zero for a masked keypoint.
        weights: (B, K) per-keypoint weights (0 for masked keypoints).
    """
    with torch.autocast(device_type=pred_heatmaps.device.type, enabled=False):
        pred = pred_heatmaps.float().clamp_min(1e-12)
        target = target_heatmaps.float()
        target = target / target.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
        kl = (target * (torch.log(target.clamp_min(1e-12)) - torch.log(pred))).sum(dim=(-1, -2))  # (B, K)
        weights = weights.float()
        per_sample = (kl * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-8)
        return per_sample.mean()
