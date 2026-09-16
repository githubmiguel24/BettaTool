"""Threshold Sensitivity Index (TSI) — thesis Ch. 1 / Ch. 3.

    TSI = |y_hat - tau| / (k * sqrt(J J^T))

The critical keypoint RMSE (in pixels) at which the 95% expanded
uncertainty interval becomes just wide enough to reach the IBC decision
boundary tau. Derived under the isotropic-noise simplifying assumption
Sigma = RMSE^2 * I (thesis Ch. 3, "Threshold Sensitivity Index").
"""

from __future__ import annotations

import numpy as np

from app.analytical.gum_propagation import COVERAGE_FACTOR_K


def compute_tsi(
    measurement: float,
    threshold: float,
    jacobian: np.ndarray,
    k: float = COVERAGE_FACTOR_K,
) -> float:
    """Critical RMSE (pixels) beyond which classification at `threshold` is unreliable.

    Args:
        measurement: the predicted geometric measurement y_hat.
        threshold: the IBC numerical boundary tau for this criterion.
        jacobian: (2*N,) sensitivity vector for this measurement.
        k: coverage factor (default 2, ~95% confidence).

    Returns:
        TSI in pixels. Compare against the model's actual measured RMSE:
        RMSE < TSI -> confident classification; RMSE > TSI -> defer to judge.
    """
    margin = abs(measurement - threshold)
    sensitivity_norm = float(np.sqrt(jacobian @ jacobian.T))
    if sensitivity_norm < 1e-12:
        return float("inf")  # measurement insensitive to keypoint error at this point
    return margin / (k * sensitivity_norm)


def is_confident(actual_rmse: float, tsi: float) -> bool:
    """True when the system may issue a confident Pass/Fault decision."""
    return actual_rmse < tsi
