# renders gaussian heatmap targets for stage 1 mse loss in pure numpy before dataset.py converts to tensors

from __future__ import annotations

import numpy as np


def render_gaussian_heatmap(
    heatmap_size: int,
    center_heatmap_space: tuple[float, float],
    sigma_px: float,
) -> np.ndarray:
    # builds one unnormalized gaussian bump with peak 1.0 using heatmap coords already divded by stride
    if sigma_px <= 0:
        raise ValueError(f"sigma_px must be positive, got {sigma_px}")

    cx, cy = center_heatmap_space
    ys, xs = np.mgrid[0:heatmap_size, 0:heatmap_size]
    # keep peak at 1.0 instead of normalizing to sum to 1 to match standard HRNet mse targets
    heatmap = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma_px**2))
    return heatmap.astype(np.float32)


def render_all_targets(
    heatmap_size: int,
    centers_heatmap_space: np.ndarray,
    sigma_px: float,
    visibility_mask: np.ndarray,
) -> np.ndarray:
    # generates (K, H, W) heatmaps for all keypoints and leaves masked ones as Zeros
    k = centers_heatmap_space.shape[0]
    out = np.zeros((k, heatmap_size, heatmap_size), dtype=np.float32)
    for i in range(k):
        # only render target if keypoint visibility flag is 1 or 2
        if visibility_mask[i]:
            out[i] = render_gaussian_heatmap(heatmap_size, tuple(centers_heatmap_space[i]), sigma_px)
    return out