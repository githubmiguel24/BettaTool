"""ISO GUM (JCGM 100:2008) uncertainty propagation via the Jacobian method.

Implements u_c(y) = sqrt(J * Sigma * J^T) (thesis Ch. 3, "Uncertainty
Propagation") and the expanded uncertainty U(y) = k * u_c(y), using the
thesis's chosen coverage factor k = 2 (~95% confidence).
"""

from __future__ import annotations

import numpy as np

COVERAGE_FACTOR_K = 2.0  # ~95% confidence, per JCGM (2008) standard practice


def combined_uncertainty(jacobian: np.ndarray, covariance: np.ndarray) -> float:
    """u_c(y) = sqrt(J Sigma J^T) for a scalar output y.

    Args:
        jacobian: (2*N,) row vector, d(measurement)/d(keypoint coordinates).
        covariance: (2*N, 2*N) covariance matrix assembled from each
            keypoint's per-landmark 2x2 heatmap covariance (see
            `assemble_block_covariance`).

    Returns:
        The standard uncertainty u_c(y), in the same units as the measurement.
    """
    variance = float(jacobian @ covariance @ jacobian.T)
    return float(np.sqrt(max(variance, 0.0)))


def expanded_uncertainty(u_c: float, k: float = COVERAGE_FACTOR_K) -> float:
    """U(y) = k * u_c(y)."""
    return k * u_c


def assemble_block_covariance(per_keypoint_covariances: np.ndarray) -> np.ndarray:
    """Builds the (2*N, 2*N) block-diagonal covariance matrix Sigma from N
    per-keypoint (2, 2) covariance matrices.

    Follows the study's scope limitation: cross-landmark correlated errors
    are not modeled, so off-diagonal blocks between different keypoints are
    zero (Scope and Limitations, "Uncertainty Quantification Boundaries").
    """
    n = per_keypoint_covariances.shape[0]
    sigma = np.zeros((2 * n, 2 * n))
    for i in range(n):
        sigma[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = per_keypoint_covariances[i]
    return sigma
