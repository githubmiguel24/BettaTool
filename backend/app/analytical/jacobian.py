# =============================================================================
# Program Title : jacobian.py - Numerical Jacobian for ISO GUM Propagation
# Programmers   : Trinidad, C. M. L.; Daanay, K. M. M.; Verceles, R. C. S.
# System Fit    : Betta Fish Quality Assessment Model > Analytical Tier.
#                 Called by app/pipeline.py for every IBC criterion; its output
#                 (the Jacobian J) is used by gum_propagation.py and tsi.py.
# Written       : September 10, 2026
# Revised       : October 14, 2026 (header block and comments for documentation)
# Purpose       : Computes how much a morphometric measurement (an angle or a
#                 ratio) changes when each keypoint coordinate moves slightly.
# Data / Algo   : Input x is a flat vector of 26 values (13 keypoints x 2).
#                 Algorithm: central finite differences,
#                 J[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2 * eps).
#                 Control: one loop over the 26 coordinates; no branching.
# =============================================================================

from __future__ import annotations

from typing import Callable

import numpy as np

# Type alias: a morphometric function maps flat keypoints to one measurement.
MeasurementFn = Callable[[np.ndarray], float]


def numerical_jacobian(fn: MeasurementFn, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Return the central-difference Jacobian (gradient) of fn at point x."""
    x = np.asarray(x, dtype=float)
    jac = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        # Partial derivative of the measurement with respect to coordinate i.
        jac[i] = (fn(x + step) - fn(x - step)) / (2 * eps)
    return jac