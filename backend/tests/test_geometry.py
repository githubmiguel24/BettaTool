"""Affine transform round-trips (Build Prompt v2 §4.4 units contract, §11)."""

from __future__ import annotations

import numpy as np
import pytest

from app.perception.geometry import (
    AffineTransform,
    bbox_crop_affine,
    crop_space_to_heatmap_space,
    heatmap_space_to_crop_space,
    letterbox_affine,
)


def test_letterbox_round_trip_within_1px() -> None:
    t = letterbox_affine(orig_w=800, orig_h=600, target_size=384)
    point = np.array([[401.3, 288.7]])
    mapped = t.apply_points(point)
    back = t.inverse().apply_points(mapped)
    assert np.abs(back - point).max() < 1.0


def test_letterbox_maps_corners_inside_target() -> None:
    t = letterbox_affine(orig_w=1000, orig_h=400, target_size=384)
    corners = np.array([[0.0, 0.0], [1000.0, 400.0]])
    mapped = t.apply_points(corners)
    assert np.all(mapped >= -1e-6) and np.all(mapped <= 384 + 1e-6)


def test_covariance_transform_scaling() -> None:
    """Sigma_orig = A Sigma A^T under a known scaling recovers the expected magnitude."""
    t = AffineTransform(A=np.eye(2) * 2.0, b=np.zeros(2))
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    scaled = t.apply_covariances(cov[None])[0]
    assert np.allclose(scaled, np.eye(2) * 4.0)


def test_covariance_transform_ignores_translation() -> None:
    t = AffineTransform(A=np.eye(2), b=np.array([100.0, -50.0]))
    cov = np.array([[3.0, 1.0], [1.0, 2.0]])
    out = t.apply_covariances(cov[None])[0]
    assert np.allclose(out, cov)


def test_compose_matches_manual_chain() -> None:
    bbox_t = bbox_crop_affine(bbox_x=50, bbox_y=60, bbox_w=200, bbox_h=150, padding_frac=0.15)
    lb_t = letterbox_affine(orig_w=200 * 1.3, orig_h=150 * 1.3, target_size=384)
    composed = bbox_t.compose(lb_t)

    point = np.array([[120.0, 100.0]])
    via_compose = composed.apply_points(point)
    via_manual = lb_t.apply_points(bbox_t.apply_points(point))
    assert np.allclose(via_compose, via_manual)


def test_heatmap_crop_space_round_trip() -> None:
    to_heatmap = crop_space_to_heatmap_space(stride=4)
    to_crop = heatmap_space_to_crop_space(stride=4)
    point = np.array([[200.0, 150.0]])
    assert np.allclose(to_heatmap.apply_points(point), [[50.0, 37.5]])
    assert np.allclose(to_crop.apply_points(to_heatmap.apply_points(point)), point)


def test_affine_transform_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError):
        AffineTransform(A=np.eye(3), b=np.zeros(2))
    with pytest.raises(ValueError):
        AffineTransform(A=np.eye(2), b=np.zeros(3))
