# Threshold Sensiivity index (TSI)
# TSI = |y_hat - tau| / (k * sqrt(J J^T))
# crtical keypoint RMSE (pixels) where 95% uncerainty reaches IBC boundry tau
# assumes isotropic noise Sigma = RMSE^2 * I

from __future__ import annotations

import numpy as np

from app.analytical.gum_propagation import COVERAGE_FACTOR_K


def compute_tsi(
    measurement: float,
    threshold: float,
    jacobian: np.ndarray,
    k: float = COVERAGE_FACTOR_K,
) -> float:
    # returns TSI in pxls. if RMSE < TSI, confident , else defer to judge
    margin = abs(measurement - threshold)
    sensitivity_norm = float(np.sqrt(jacobian @ jacobian.T))
    
    if sensitivity_norm < 1e-12:
        # insensitive to keypont error here
        return float("inf") 
        
    return margin / (k * sensitivity_norm)


def is_confident(actual_rmse: float, tsi: float) -> bool:
    # true when we can safely pass/Fault
    return actual_rmse < tsi