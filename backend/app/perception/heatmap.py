"""Heatmap -> coordinate extraction, and (legacy) heatmap -> Gaussian fitting.

`soft_argmax` is the single source of truth for turning a spatial-softmax
heatmap into a sub-pixel (x, y) mean. Build Prompt v2 §4.2 requires that
training and inference use the *same* function for this — `training/`
imports `soft_argmax` from this module rather than reimplementing it, so a
divergence between the training objective and deployed behavior is
structurally impossible.

`heatmap_to_gaussian` / `batch_heatmaps_to_gaussians` implement the older
Payer et al. (2020)-style moment fit (mean + covariance both read off the
heatmap distribution). That approach is superseded by the explicit
CovarianceHead in `app/perception/hrnet.py` (Build Prompt v2 §6), which is
now the primary source of Sigma; the moment-based covariance is kept only as
a documented fallback in `app/pipeline.py` for a model checkpoint that has no
covariance head.
"""

from __future__ import annotations

import numpy as np
import torch


def soft_argmax(heatmaps: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Differentiable integral regression: heatmap -> sub-pixel (x, y).

    This is the shared mean-extraction function used at BOTH training time
    (to compute the L1-on-mu term and to sample CovarianceHead features at
    mu) and inference time (`app/pipeline.py`). Do not reimplement this
    logic anywhere else (Build Prompt v2 §4.2).

    Args:
        heatmaps: (B, K, H, W) tensor. Each (H, W) channel is expected to
            already be a spatial probability distribution (non-negative,
            summing to ~1 over H*W), e.g. the output of a spatial softmax —
            this function does not re-normalize.
        temperature: divides the heatmap's log-values before the (already
            applied) softmax would be re-taken; exposed here only so a
            caller that wants a *sharper* expectation can pre-scale logits
            before the softmax step upstream. Left at 1.0, this function is
            a pure weighted-centroid computation. Kept as an explicit
            parameter (not a magic constant) so both training and inference
            set it from the same config key (`model.softargmax_temperature`,
            currently unused / reserved) rather than diverging.

    Returns:
        (B, K, 2) tensor of (x, y) coordinates in heatmap-space pixels
        (i.e. in the range [0, W-1] x [0, H-1]); multiply by `stride` to
        reach crop-space, per the units contract in
        `app/perception/geometry.py`.
    """
    if temperature != 1.0:
        # Reserved hook: re-sharpen before centroid extraction if ever needed.
        # heatmaps = torch.softmax(torch.log(heatmaps.clamp_min(1e-12)) / temperature, dim=(-1, -2))
        raise NotImplementedError("temperature != 1.0 is reserved; not used by the current config.")

    b, k, h, w = heatmaps.shape
    device = heatmaps.device
    dtype = heatmaps.dtype

    xs = torch.arange(w, device=device, dtype=dtype)
    ys = torch.arange(h, device=device, dtype=dtype)

    mass = heatmaps.sum(dim=(-1, -2)).clamp_min(1e-12)  # (B, K)
    mean_x = (heatmaps.sum(dim=-2) * xs).sum(dim=-1) / mass  # (B, K)
    mean_y = (heatmaps.sum(dim=-1) * ys).sum(dim=-1) / mass  # (B, K)

    return torch.stack([mean_x, mean_y], dim=-1)


def heatmap_to_gaussian(heatmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts one (H, W) heatmap channel into a mean vector and 2x2 covariance matrix.

    Args:
        heatmap: non-negative (H, W) array, typically a softmax'd network output.

    Returns:
        mean: (2,) array of (x, y) pixel coordinates.
        covariance: (2, 2) array, the spatial covariance matrix (Sigma).
    """
    h, w = heatmap.shape
    weights = heatmap / (heatmap.sum() + 1e-12)

    ys, xs = np.mgrid[0:h, 0:w]
    mean_x = float((weights * xs).sum())
    mean_y = float((weights * ys).sum())
    mean = np.array([mean_x, mean_y])

    dx = xs - mean_x
    dy = ys - mean_y
    var_xx = float((weights * dx * dx).sum())
    var_yy = float((weights * dy * dy).sum())
    cov_xy = float((weights * dx * dy).sum())

    covariance = np.array([[var_xx, cov_xy], [cov_xy, var_yy]])
    return mean, covariance


def batch_heatmaps_to_gaussians(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized `heatmap_to_gaussian` over the keypoint axis.

    Args:
        heatmaps: (N, H, W) array, one channel per keypoint.

    Returns:
        means: (N, 2) array.
        covariances: (N, 2, 2) array.
    """
    n = heatmaps.shape[0]
    means = np.zeros((n, 2))
    covariances = np.zeros((n, 2, 2))
    for i in range(n):
        means[i], covariances[i] = heatmap_to_gaussian(heatmaps[i])
    return means, covariances
