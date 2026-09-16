from __future__ import annotations

import numpy as np

COVERAGE_FACTOR_K = 2.0  # 95 perc confidnce per JCGM std

def combined_uncertainty(jacobian: np.ndarray, covariance: np.ndarray) -> float:
    #calculte standard uncErt u_c = sqrt(J Sigma J^T) for scalr outpt
    variance = float(jacobian @ covariance @ jacobian.T)
    return float(np.sqrt(max(variance, 0.0)))


def expanded_uncertainty(u_c: float, k: float = COVERAGE_FACTOR_K) -> float:
    #get eXpnded uncertnty
    return k * u_c


def assemble_block_covariance(per_keypoint_covariances: np.ndarray) -> np.ndarray:
    #buid block-diag covar matrix, cross-lndmark errs r ignord here
    n = per_keypoint_covariances.shape[0]
    sigma = np.zeros((2 * n, 2 * n))
    for i in range(n):
        sigma[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = per_keypoint_covariances[i]
    return sigma