"""Sanity checks for the ISO GUM propagation math."""

import numpy as np

from app.analytical.gum_propagation import (
    assemble_block_covariance,
    combined_uncertainty,
    expanded_uncertainty,
)


def test_combined_uncertainty_scales_with_covariance():
    jacobian = np.array([1.0, 0.0, 1.0, 0.0])  # sensitive to keypoint x-coordinates only
    small_cov = assemble_block_covariance(np.array([np.eye(2), np.eye(2)]))
    large_cov = assemble_block_covariance(np.array([4 * np.eye(2), 4 * np.eye(2)]))

    assert combined_uncertainty(jacobian, large_cov) > combined_uncertainty(jacobian, small_cov)


def test_expanded_uncertainty_uses_coverage_factor():
    assert expanded_uncertainty(1.0, k=2.0) == 2.0


def test_zero_sensitivity_gives_zero_uncertainty():
    jacobian = np.zeros(4)
    covariance = assemble_block_covariance(np.array([np.eye(2), np.eye(2)]))
    assert combined_uncertainty(jacobian, covariance) == 0.0
