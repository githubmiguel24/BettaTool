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
    # largest isotropic keypoint error (px) for which k * sigma * ||J|| still stays inside |y_hat - tau|
    margin = abs(measurement - threshold)
    sensitivity_norm = float(np.sqrt(jacobian @ jacobian.T))

    if sensitivity_norm < 1e-12:
        # insensitive to keypont error here
        return float("inf")

    return margin / (k * sensitivity_norm)


def predicted_keypoint_sigma(jacobian: np.ndarray, per_keypoint_covariances: np.ndarray) -> float:
    # sigma_hat: the isotropic keypoint error (px) that would produce the same measurement uncertainty as the predicted
    # covariances, sqrt(J Sigma J^T) / ||J||. It weights each keypoint by how much the criterion depends on it and keeps
    # the direction of each covariance, which a plain average of the predicted sigmas does not.
    norm = float(np.sqrt(jacobian @ jacobian))
    if norm < 1e-12:
        return 0.0
    return combined_uncertainty(jacobian, assemble_block_covariance(per_keypoint_covariances)) / norm


def is_confident(sigma_hat: float, tsi: float) -> bool:
    # confident if the predicted keypoint uncertainty is below the criterion's TSI, otherwise defer
    return sigma_hat < tsi
