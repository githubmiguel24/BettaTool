# =============================================================================
# Program Title: Threshold Sensitivity Index (TSI)
# Programmers: Daanay, Maraño, Trinidad, Verceles
# System Fit:  Part of the Analytical tier/module, 
#               its output is used by the decisional module.
# Written: September 11, 2026
# Revised: October 14, 2026 
# Purpose:  Computes the TSI, which is the largest tolerable
#           keypoint error and the predicted keypoint error
# Data Structure: NumPy arrays for: covariance matrix and jacobian matrix
# Algorithm:
# Control: 
# =============================================================================

 
# threshold sensitivity index (TSI) and the deferral rule built on it

from __future__ import annotations

import numpy as np

from app.analytical.gum_propagation import COVERAGE_FACTOR_K, assemble_block_covariance, combined_uncertainty


def compute_tsi(
    measurement: float,
    threshold: float,
    jacobian: np.ndarray,
    k: float = COVERAGE_FACTOR_K,
) -> float:
    # largest isotropic keypoint error (px)
    margin = abs(measurement - threshold)
    sensitivity_norm = float(np.sqrt(jacobian @ jacobian.T))
    
    #prevent div by zero
    if sensitivity_norm < 1e-12:
        return float("inf")

    return margin / (k * sensitivity_norm)


def predicted_keypoint_sigma(jacobian: np.ndarray, per_keypoint_covariances: np.ndarray) -> float:
    norm = float(np.sqrt(jacobian @ jacobian))
    if norm < 1e-12:
        return 0.0
    return combined_uncertainty(jacobian, assemble_block_covariance(per_keypoint_covariances)) / norm


def is_confident(sigma_hat: float, tsi: float) -> bool:
    # confident if the predicted keypoint uncertainty is below the criterion's TSI, otherwise defer
    return sigma_hat < tsi
