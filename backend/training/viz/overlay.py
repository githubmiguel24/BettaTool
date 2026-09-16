"""Overlay renderer: predicted keypoints, 95% covariance ellipses, visibility
state, and ground truth (when available), drawn on the original image
(Build Prompt v2 §10.5). Also auto-exports the 20 best/worst test images by
radial error into the run directory.

Pure PIL + NumPy + Matplotlib (no torch dependency) — takes plain arrays,
so it can be called from `evaluate.py` (torch tensors, `.cpu().numpy()`'d
first) or directly from a notebook while writing up results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import patches
from matplotlib.figure import Figure

from app.perception.keypoints import KEYPOINT_SHORT_CODES, SKELETON_EDGES

_VIS_COLORS = {2: "#22c55e", 1: "#eab308", 0: "#f97316", -1: "#94a3b8"}  # clear/ambiguous/occluded/oof


def _covariance_ellipse_params(cov: np.ndarray, n_std: float) -> tuple[float, float, float]:
    """Returns (width, height, angle_degrees) of the `n_std`-sigma ellipse for a 2x2 covariance.

    `n_std` for a 95% confidence ellipse under a 2D Gaussian is
    sqrt(chi2.ppf(0.95, df=2)) ≈ 2.448 — pass that in explicitly from the
    caller (which reads it from `evaluation.coverage_levels` in config)
    rather than hardcoding it here.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0, None)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    width, height = 2 * n_std * np.sqrt(eigvals)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return float(width), float(height), float(angle)


def render_overlay(
    image: np.ndarray,
    pred_mu: np.ndarray,
    pred_cov: np.ndarray,
    visibility_probs: np.ndarray,
    gt_mu: np.ndarray | None = None,
    ellipse_n_std: float = 2.448,
    title: str | None = None,
) -> Figure:
    """Renders one overlay figure.

    Args:
        image: (H, W, 3) uint8 array, ORIGINAL-image pixels.
        pred_mu: (K, 2) predicted coordinates, original-image pixels.
        pred_cov: (K, 2, 2) predicted covariances, original-image px^2 units
            (already transformed via `app/perception/geometry.py` — this
            function does no unit conversion of its own).
        visibility_probs: (K,) predicted visibility probability in [0, 1].
        gt_mu: optional (K, 2) ground truth, drawn as small crosses.
        ellipse_n_std: number of standard deviations for the drawn ellipse.
        title: optional figure title (e.g. the image_id and its radial error).

    Returns:
        A matplotlib Figure — caller saves it (`fig.savefig(...)`) and closes it.
    """
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(image)

    for i, j in SKELETON_EDGES:
        ax.plot([pred_mu[i, 0], pred_mu[j, 0]], [pred_mu[i, 1], pred_mu[j, 1]], color="white", linewidth=1, alpha=0.6)

    for k in range(pred_mu.shape[0]):
        color = _VIS_COLORS.get(2 if visibility_probs[k] >= 0.5 else 0, "#94a3b8")
        width, height, angle = _covariance_ellipse_params(pred_cov[k], ellipse_n_std)
        ellipse = patches.Ellipse(pred_mu[k], width, height, angle=angle, facecolor="none", edgecolor=color, linewidth=1.5, alpha=0.85)
        ax.add_patch(ellipse)
        ax.scatter(*pred_mu[k], s=14, color=color, zorder=3)
        ax.annotate(KEYPOINT_SHORT_CODES[k], pred_mu[k], fontsize=6, color=color, xytext=(3, 3), textcoords="offset points")

        if gt_mu is not None:
            ax.scatter(*gt_mu[k], s=30, marker="x", color="red", zorder=4)

    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


def export_best_worst(
    image_ids: list[str],
    images: list[np.ndarray],
    pred_mu: np.ndarray,
    pred_cov: np.ndarray,
    visibility_probs: np.ndarray,
    gt_mu: np.ndarray,
    per_image_error: np.ndarray,
    out_dir: str | Path,
    k: int = 20,
) -> None:
    """Saves the `k` best and `k` worst images by mean radial error into `out_dir/best/` and `out_dir/worst/`."""
    out_dir = Path(out_dir)
    order = np.argsort(per_image_error)
    best_idx = order[:k]
    worst_idx = order[::-1][:k]

    for subdir, indices in [("best", best_idx), ("worst", worst_idx)]:
        target_dir = out_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        for rank, i in enumerate(indices):
            fig = render_overlay(
                images[i], pred_mu[i], pred_cov[i], visibility_probs[i], gt_mu[i],
                title=f"{image_ids[i]} — mean radial error {per_image_error[i]:.2f}px",
            )
            fig.savefig(target_dir / f"{rank:02d}_{image_ids[i]}.png", dpi=100)
