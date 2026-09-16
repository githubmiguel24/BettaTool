"""Numerical Jacobian utilities for GUM uncertainty propagation.

Each morphometric function in `morphometrics.py` maps a flat keypoint
vector x (length 2*N, alternating x/y) to a single scalar measurement y.
The Jacobian J = df/dx is obtained here via central finite differences so
new morphometric criteria can be added without hand-deriving a closed-form
derivative for each one.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

MeasurementFn = Callable[[np.ndarray], float]


def numerical_jacobian(fn: MeasurementFn, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Central-difference Jacobian of a scalar function `fn` at point `x`.

    Args:
        fn: maps a flat (2*N,) keypoint vector to a scalar measurement.
        x: the flat keypoint vector (x1, y1, x2, y2, ...) at which to evaluate.
        eps: finite-difference step size, in pixels.

    Returns:
        J: (2*N,) row vector of partial derivatives d(measurement)/d(x_i).
    """
    x = np.asarray(x, dtype=float)
    jacobian = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        jacobian[i] = (fn(x + step) - fn(x - step)) / (2 * eps)
    return jacobian
