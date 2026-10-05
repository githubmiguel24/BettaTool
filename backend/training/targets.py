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


# fin keypoints whose target can be stretched ACROSS the fin ray, and the keypoints that define the ray's base
# (indices follow app/perception/keypoints.py: 2/3 dorsal base, 5/6 peduncle, 10/11 anal base)
FIN_RAY_BASES: dict[int, tuple[int, ...]] = {4: (2, 3), 7: (5, 6), 8: (5, 6), 9: (5, 6), 12: (10, 11)}


def render_anisotropic_heatmap(
    heatmap_size: int,
    center_heatmap_space: tuple[float, float],
    sigma_along: float,
    sigma_across: float,
    direction: tuple[float, float],
) -> np.ndarray:
    # gaussian with sigma_along along `direction` (unit vector) and sigma_across perpendicular to it, peak 1.0
    cx, cy = center_heatmap_space
    ux, uy = direction
    ys, xs = np.mgrid[0:heatmap_size, 0:heatmap_size]
    dx, dy = xs - cx, ys - cy
    along = dx * ux + dy * uy
    across = -dx * uy + dy * ux
    return np.exp(-(along**2 / (2.0 * sigma_along**2) + across**2 / (2.0 * sigma_across**2))).astype(np.float32)


def render_all_targets_anisotropic(
    heatmap_size: int,
    centers_heatmap_space: np.ndarray,
    sigma_px: float,
    visibility_mask: np.ndarray,
    across_scale: float,
    keypoint_indices: tuple[int, ...],
) -> np.ndarray:
    # like render_all_targets, but the listed fin keypoints get a target stretched across the base->tip ray
    # (sigma_across = across_scale * sigma_px); a keypoint falls back to the round target if its base is not visible
    out = render_all_targets(heatmap_size, centers_heatmap_space, sigma_px, visibility_mask)
    for i in keypoint_indices:
        bases = FIN_RAY_BASES.get(i)
        if bases is None or not visibility_mask[i] or not all(visibility_mask[b] for b in bases):
            continue
        base = np.mean([centers_heatmap_space[b] for b in bases], axis=0)
        ray = np.asarray(centers_heatmap_space[i], dtype=float) - base
        length = float(np.linalg.norm(ray))
        if length < 1e-6:
            continue
        out[i] = render_anisotropic_heatmap(
            heatmap_size, tuple(centers_heatmap_space[i]), sigma_px, sigma_px * across_scale, (ray[0] / length, ray[1] / length)
        )
    return out
