"""Renders Gaussian heatmap targets for Stage-1 heatmap MSE supervision.

Pure NumPy (no PyTorch dependency) so this module — and its correctness —
does not depend on whether `torch` happens to be importable in a given
environment; `training/dataset.py` converts the result to a tensor when
collating a batch.
"""

from __future__ import annotations

import numpy as np


def render_gaussian_heatmap(
    heatmap_size: int,
    center_heatmap_space: tuple[float, float],
    sigma_px: float,
) -> np.ndarray:
    """Renders one (heatmap_size, heatmap_size) unnormalized Gaussian bump.

    Args:
        heatmap_size: output side length (e.g. 96).
        center_heatmap_space: (x, y) center IN HEATMAP-SPACE pixels (i.e.
            already divided by stride — see `app/perception/geometry.py`'s
            units contract). May lie outside [0, heatmap_size) — the
            rendered bump is simply mostly/entirely clipped in that case,
            which is fine because masked keypoints never reach this function
            with a meaningful target anyway.
        sigma_px: Gaussian standard deviation, in heatmap-space pixels
            (config key `loss.heatmap_target_sigma_px`, default 2.0).

    Returns:
        (heatmap_size, heatmap_size) float32 array, peak value 1.0 at the
        (rounded) center, NOT normalized to sum to 1 — `heatmap_mse.py`
        compares this directly against the predicted (softmax-normalized)
        heatmap, matching the common HRNet convention of an MSE target with
        peak 1.0 rather than a matched-scale probability target.
    """
    if sigma_px <= 0:
        raise ValueError(f"sigma_px must be positive, got {sigma_px}")

    cx, cy = center_heatmap_space
    ys, xs = np.mgrid[0:heatmap_size, 0:heatmap_size]
    heatmap = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma_px**2))
    return heatmap.astype(np.float32)


def render_all_targets(
    heatmap_size: int,
    centers_heatmap_space: np.ndarray,
    sigma_px: float,
    visibility_mask: np.ndarray,
) -> np.ndarray:
    """Renders one heatmap per keypoint; masked keypoints get an all-zero target.

    Args:
        heatmap_size: output side length.
        centers_heatmap_space: (K, 2) array of (x, y) heatmap-space centers.
        sigma_px: Gaussian sigma, heatmap-space pixels.
        visibility_mask: (K,) array, truthy for keypoints with visibility
            flag 2 or 1 (i.e. the ones this function should actually render
            a target for).

    Returns:
        (K, heatmap_size, heatmap_size) float32 array.
    """
    k = centers_heatmap_space.shape[0]
    out = np.zeros((k, heatmap_size, heatmap_size), dtype=np.float32)
    for i in range(k):
        if visibility_mask[i]:
            out[i] = render_gaussian_heatmap(heatmap_size, tuple(centers_heatmap_space[i]), sigma_px)
    return out
