#jacobian utils for iso gum

from __future__ import annotations

from typing import Callable

import numpy as np

MeasurementFn = Callable[[np.ndarray], float]


def numerical_jacobian(fn: MeasurementFn, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    # get central-difference jAcobian of fn at pOint x
    x = np.asarray(x, dtype=float)
    jac = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        # calc partial deriv
        jac[i] = (fn(x + step) - fn(x - step)) / (2 * eps)
    return jac