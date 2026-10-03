# threshold sensitivity index and confidence gates for uncertainty evaluation

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


def is_confident_isotropic(actual_rmse: float, tsi: float) -> bool:
    # legacy isotropic baseline decision rule for ablations
    return actual_rmse < tsi


# Backwards-compatible alias. Do not add new callers
is_confident = is_confident_isotropic


def is_confident_gum(measurement: float, threshold: float, expanded_uncertainty: float) -> bool:
    # confident if expanded uncertainty doesn't reach the IBC threshold boundary
    return abs(measurement - threshold) > expanded_uncertainty