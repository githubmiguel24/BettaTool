# Shared affine transforms between heatmap (96x96), crop (384x384), and original image space.
# Keep all coord math here so training and pipeline.py stay in sync and avoid silent 4x stride bugs.
# Convention: point_out = A @ point_in + b

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# 2D affine transform: point_out = A @ point_in + b
@dataclass(frozen=True)
class AffineTransform:
    A: np.ndarray  # (2, 2)
    b: np.ndarray  # (2,)

    def __post_init__(self) -> None:
        if self.A.shape != (2, 2):
            raise ValueError(f"A must be (2, 2), got {self.A.shape}")
        if self.b.shape != (2,):
            raise ValueError(f"b must be (2,), got {self.b.shape}")

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        # apply transform to an (..., 2) array of (x, y) points
        return points @ self.A.T + self.b

    def apply_covariances(self, covariances: np.ndarray) -> np.ndarray:
        # update (..., 2, 2) Covariances via A @ Sigma @ A^T (translaton b doesn't affect variance)
        return self.A @ covariances @ self.A.T

    def inverse(self) -> "AffineTransform":
        # invert transform to map output coords back to input space
        a_inv = np.linalg.inv(self.A)
        return AffineTransform(A=a_inv, b=-a_inv @ self.b)

    def compose(self, other: "AffineTransform") -> "AffineTransform":
        # chain transforms: apply self first, then other
        return AffineTransform(A=other.A @ self.A, b=other.A @ self.b + other.b)


def letterbox_affine(
    orig_w: float,
    orig_h: float,
    target_size: int,
) -> AffineTransform:
    # map image coords to aspect-preserving square letterbox (compose after bbox_crop_affine if cropping first)
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"orig_w and orig_h must be positive, got {orig_w}, {orig_h}")

    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = orig_w * scale, orig_h * scale
    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0

    a = np.array([[scale, 0.0], [0.0, scale]])
    b = np.array([pad_x, pad_y])
    return AffineTransform(A=a, b=b)


def bbox_crop_affine(bbox_x: float, bbox_y: float, bbox_w: float, bbox_h: float, padding_frac: float) -> AffineTransform:
    # shift coords to padded bbox top-left origin without resizing (padding keeps fin tips from getting cliped)
    pad_x = bbox_w * padding_frac
    pad_y = bbox_h * padding_frac
    origin_x = bbox_x - pad_x
    origin_y = bbox_y - pad_y
    return AffineTransform(A=np.eye(2), b=-np.array([origin_x, origin_y]))


def crop_space_to_heatmap_space(stride: int) -> AffineTransform:
    #scale crop coords (384x384) down to Heatmap coords (96x96) using stride
    scale = 1.0 / stride
    return AffineTransform(A=np.eye(2) * scale, b=np.zeros(2))


def heatmap_space_to_crop_space(stride: int) -> AffineTransform:
    # scale heatmap coords back up to crop space
    return crop_space_to_heatmap_space(stride).inverse()