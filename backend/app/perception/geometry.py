"""Affine transforms between the three coordinate spaces in play.

Build Prompt v2 §4.4 (units contract) — read this docstring before touching
coordinates anywhere in the perception tier:

    heatmap space   (96x96, "stride" scale)   raw soft-argmax output
    crop space      (384x384)                 training loss, Sigma prediction
    original space   (variable per image)      evaluation metrics, GUM tier

This module is the *only* place that converts between them. `training/`
and `app/pipeline.py` both import from here rather than re-deriving an
affine transform independently — a divergence here is exactly the kind of
silent 4x bug Build Prompt v2 warns about in §4.4.

Convention: an affine transform is represented as (A, b) with
    point_out = A @ point_in + b
where A is a (2, 2) matrix and b is a (2,) vector. Composing transforms is
plain matrix algebra: applying (A1, b1) then (A2, b2) is
    A2 @ (A1 @ p + b1) + b2 = (A2 @ A1) @ p + (A2 @ b1 + b2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineTransform:
    """point_out = A @ point_in + b."""

    A: np.ndarray  # (2, 2)
    b: np.ndarray  # (2,)

    def __post_init__(self) -> None:
        if self.A.shape != (2, 2):
            raise ValueError(f"A must be (2, 2), got {self.A.shape}")
        if self.b.shape != (2,):
            raise ValueError(f"b must be (2,), got {self.b.shape}")

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        """Applies the transform to an (..., 2) array of (x, y) points."""
        return points @ self.A.T + self.b

    def apply_covariances(self, covariances: np.ndarray) -> np.ndarray:
        """Sigma_out = A @ Sigma_in @ A^T for an (..., 2, 2) array of covariances.

        Only the linear part A is used — a covariance matrix is invariant to
        translation, so `b` never enters this computation.
        """
        return self.A @ covariances @ self.A.T

    def inverse(self) -> "AffineTransform":
        """Returns the transform mapping `point_out` back to `point_in`."""
        a_inv = np.linalg.inv(self.A)
        return AffineTransform(A=a_inv, b=-a_inv @ self.b)

    def compose(self, other: "AffineTransform") -> "AffineTransform":
        """Returns the transform equivalent to applying `self` then `other`."""
        return AffineTransform(A=other.A @ self.A, b=other.A @ self.b + other.b)


def letterbox_affine(
    orig_w: float,
    orig_h: float,
    target_size: int,
) -> AffineTransform:
    """Affine mapping original-image pixel coords -> a `target_size`-square,
    aspect-ratio-preserving, letterbox-padded crop.

    This is used directly when no bounding box is available. When a fish_box
    crop is used instead, compose `bbox_crop_affine` with this function's
    output applied to the *cropped* image's own (width, height).

    Args:
        orig_w: width, in pixels, of the image being letterboxed.
        orig_h: height, in pixels, of the image being letterboxed.
        target_size: output side length in pixels (square).

    Returns:
        AffineTransform mapping (x, y) in the input image to (x, y) in the
        `target_size` x `target_size` output.
    """
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
    """Affine mapping original-image pixel coords -> coords relative to a
    padded crop of `bbox` (top-left origin, no resize).

    `padding_frac` grows the box by that fraction of its own size on every
    side before cropping (config key `image.bbox_padding`), so fin tips near
    the fish_box edge are not clipped.
    """
    pad_x = bbox_w * padding_frac
    pad_y = bbox_h * padding_frac
    origin_x = bbox_x - pad_x
    origin_y = bbox_y - pad_y
    return AffineTransform(A=np.eye(2), b=-np.array([origin_x, origin_y]))


def crop_space_to_heatmap_space(stride: int) -> AffineTransform:
    """Affine mapping crop-space (384x384) coords -> heatmap-space (96x96) coords."""
    scale = 1.0 / stride
    return AffineTransform(A=np.eye(2) * scale, b=np.zeros(2))


def heatmap_space_to_crop_space(stride: int) -> AffineTransform:
    """Inverse of `crop_space_to_heatmap_space`."""
    return crop_space_to_heatmap_space(stride).inverse()
