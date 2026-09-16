"""Stage-1 heatmap regression loss against rendered Gaussian targets."""

from __future__ import annotations

import torch


def heatmap_mse_loss(pred_heatmaps: torch.Tensor, target_heatmaps: torch.Tensor, visibility_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predicted and target heatmaps, masked per-keypoint.

    Args:
        pred_heatmaps: (B, K, H, W) predicted (spatial-softmax normalized) heatmaps.
        target_heatmaps: (B, K, H, W) rendered Gaussian targets
            (see `training/targets.py`); zero everywhere for a masked keypoint.
        visibility_mask: (B, K) float tensor, 1.0 for keypoints with
            visibility flag 2 or 1, 0.0 for flag 0 or -1 (Build Prompt v2 §3.3).

    Returns:
        Scalar loss: per-pixel MSE averaged over H*W, then averaged over
        VISIBLE keypoints only, then averaged over the batch. A batch where
        every keypoint is masked returns 0.0 rather than NaN.
    """
    per_keypoint_mse = ((pred_heatmaps - target_heatmaps) ** 2).mean(dim=(-1, -2))  # (B, K)
    masked = per_keypoint_mse * visibility_mask
    denom = visibility_mask.sum(dim=-1).clamp_min(1e-8)  # (B,)
    per_sample = masked.sum(dim=-1) / denom
    return per_sample.mean()
