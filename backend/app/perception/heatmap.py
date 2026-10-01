# Heatmap coordinate extraction and legacy gaussian fitting utilities
# soft_argmax is shared by both training and inference so behavior stays consistent
# The gaussian moment fit functions are only kept as a fallback if CovarianceHead is missing

from __future__ import annotations

import numpy as np
import torch


# Extracts sub-pixel coords from (B, K, H, W) spatial probability heatmaps
# Returns (B, K, 2) in heatmap pixel space, multiply by stride for crop space
def soft_argmax(heatmaps: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        # temperatre scaling isn't used right now
        raise NotImplementedError("temperature != 1.0 is reserved; not used by the current config.")

    b, k, h, w = heatmaps.shape
    device = heatmaps.device
    dtype = heatmaps.dtype

    # build coordinate grids on the same device
    xs = torch.arange(w, device=device, dtype=dtype)
    ys = torch.arange(h, device=device, dtype=dtype)

    # compute weighted centroids normalized by total mass
    mass = heatmaps.sum(dim=(-1, -2)).clamp_min(1e-12)
    mean_x = (heatmaps.sum(dim=-2) * xs).sum(dim=-1) / mass
    mean_y = (heatmaps.sum(dim=-1) * ys).sum(dim=-1) / mass

    return torch.stack([mean_x, mean_y], dim=-1)


# Fits a 2D gaussian (mean and 2x2 covarince) to a single (H, W) heatmap
def heatmap_to_gaussian(heatmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = heatmap.shape
    weights = heatmap / (heatmap.sum() + 1e-12)

    # get expected x and y pixel coordinates
    ys, xs = np.mgrid[0:h, 0:w]
    mean_x = float((weights * xs).sum())
    mean_y = float((weights * ys).sum())
    mean = np.array([mean_x, mean_y])

    # calculate spatial variance and covariance around the mean
    dx = xs - mean_x
    dy = ys - mean_y
    var_xx = float((weights * dx * dx).sum())
    var_yy = float((weights * dy * dy).sum())
    cov_xy = float((weights * dx * dy).sum())

    covariance = np.array([[var_xx, cov_xy], [cov_xy, var_yy]])
    return mean, covariance


# Runs gaussian fitting across a batch of (N, H, W) keypoint Heatmaps
def batch_heatmaps_to_gaussians(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = heatmaps.shape[0]
    means = np.zeros((n, 2))
    covariances = np.zeros((n, 2, 2))
    # loop over each keypoint channel to get its mean and covariance
    for i in range(n):
        means[i], covariances[i] = heatmap_to_gaussian(heatmaps[i])
    return means, covariances