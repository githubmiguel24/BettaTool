"""Sanity checks for the Threshold Sensitivity Index."""

import numpy as np

from app.analytical.tsi import compute_tsi, is_confident


def test_tsi_grows_with_margin_from_threshold():
    jacobian = np.array([1.0, 0.0])
    near_threshold = compute_tsi(measurement=179.0, threshold=180.0, jacobian=jacobian)
    far_from_threshold = compute_tsi(measurement=150.0, threshold=180.0, jacobian=jacobian)
    assert far_from_threshold > near_threshold


def test_is_confident_boundary():
    assert is_confident(actual_rmse=1.0, tsi=2.0) is True
    assert is_confident(actual_rmse=2.0, tsi=2.0) is False
    assert is_confident(actual_rmse=3.0, tsi=2.0) is False
