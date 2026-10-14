# =============================================================================
# Program Title: ISO GUM Uncertainty Propagation
# Programmers: Daanay, Maraño, Trinidad, Verceles
# System Fit:  Part of the Analytical tier/module
#              it receives the per-keypoint 2x2 covariances from 
#              the Perceptual module  (HRNet covariance head)
#              its results is used in the assessment report.
# Written : September 14, 2026
# Revised : October 14, 2026 
# Purpose : This turns keypoint uncertainty from perceptual tier 
#           and turns them into margin of error for each measurement, 
#           to do this, it follows the ISO GUM method.
# urpose : This turns keypoint uncertainty from perceptual tier 
#           and turns them into margin of error for each measurement, 
#           to do this, it follows the ISO GUM method.
# Data Structure: NumPy arrays for: 13 x 2 x 2 covariances, 26 x 26 block-diagonal matrix, and Jacobian 
# Algorithm: Uses Law of Propagation of Uncertainty 
# Control: Uses for loop once for sigma, the rest is sequential.
# =============================================================================

from __future__ import annotations
import numpy as np

COVERAGE_FACTOR_K = 2.0  # 95 perc confidnce per JCGM standard

def combined_uncertainty(jacobian: np.ndarray, covariance: np.ndarray) -> float:
    #calculte standard uncErt u_c = sqrt(J Sigma J^T) for scalr outpt
    variance = float(jacobian @ covariance @ jacobian.T)
    return float(np.sqrt(max(variance, 0.0)))


def expanded_uncertainty(u_c: float, k: float = COVERAGE_FACTOR_K) -> float:
    #get expanded uncertainty
    return k * u_c


def assemble_block_covariance(per_keypoint_covariances: np.ndarray) -> np.ndarray:
    #buid block-diag cov matrix, cross-lndmark errors are ignored
    n = per_keypoint_covariances.shape[0]
    sigma = np.zeros((2 * n, 2 * n))
    for i in range(n):
        sigma[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = per_keypoint_covariances[i]
    return sigma