# =============================================================================
# Program Title : Main Analytical Engine

# Programmers   : Daanay, Kurt Michael M.
#                 Trinidad, Charles Miguel L.
#                 Verceles, Rogel Cristian S.

# System Fit    : Serves as the main file for the Analytical module. 
#                 It receives flat keypoint arrays and 2x2 covariances 
#                 from the Perceptual Tier, calculates geometric measurements 
#                 and their uncertainty bounds via ISO GUM propagation, 
#                 and derives the Threshold Sensitivity Index (TSI) for the Decisional 
#                 Tier.

# Written       : September 10, 2026
# Revised       : October 14, 2026 

# Purpose       : Transforms perceptual keypoint uncertainty from the Perceptual Tier 
#                 into geometric margins of error using the ISO GUM method. Computes 
#                 measurement sensitivity (Jacobians), calculates the maximum tolerable 
#                 pixel error (TSI) for safe classification against strict IBC thresholds, 
#                 and loads conformal prediction scaling factors.

# Data Structure: NumPy arrays (13x2x2 covariances, 26x26 block-diagonal matrix, 1x26 
#                 Jacobian vectors) and Path objects.

# Algorithm     : Law of Propagation of Uncertainty, central finite differences for numerical 
#                 Jacobians, and conformal prediction scaling.

# Control       : Sequential execution, vectorized NumPy array operations, for-loops for 
#                 matrix assembly  and partial derivatives, and conditionals 
#                 for configuration loading and zero-division prevention.
# =============================================================================

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

# CONSTANTS & TYPES

COVERAGE_FACTOR_K = 2.0  # 95% confidence interval per JCGM standard

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "training" / "configs" / "hrnet_w32.yaml"
CONFIG_ENV_VAR = "BETTA_CONFIG_PATH"
CONFIG_KEY = "measurement_factor"

# A morphometric function maps flat keypoints to one measurement
MeasurementFn = Callable[[np.ndarray], float]


# JACOBIAN MATRIX (senstivity)

def numerical_jacobian(fn: MeasurementFn, x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    # get central-difference jacobian of fn at point x
    x = np.asarray(x, dtype=float)
    jac = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        # calculates partial deriv of the measurement with respect to coordinate i
        jac[i] = (fn(x + step) - fn(x - step)) / (2 * eps)
    return jac


# ISO GUM UNCERTAINTY PROPAGATION

def assemble_block_covariance(per_keypoint_covariances: np.ndarray) -> np.ndarray:
    # Build a block-diagonal covariance matrix; cross-landmark errors are treated as independent
    n = per_keypoint_covariances.shape[0]
    sigma = np.zeros((2 * n, 2 * n))
    for i in range(n):
        sigma[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = per_keypoint_covariances[i]
    return sigma


def combined_uncertainty(jacobian: np.ndarray, covariance: np.ndarray) -> float:
    # Calculate standard uncertainty u_c = sqrt(J * Sigma * J^T) for a scalar output
    variance = float(jacobian @ covariance @ jacobian.T)
    return float(np.sqrt(max(variance, 0.0)))


def expanded_uncertainty(u_c: float, k: float = COVERAGE_FACTOR_K) -> float:
    # Calculate the expanded uncertainty
    return k * u_c


# THRESHOLD SENSITIVITY INDEX 

def compute_tsi(
    measurement: float,
    threshold: float,
    jacobian: np.ndarray,
    k: float = COVERAGE_FACTOR_K,
) -> float:
    # Largest keypoint error in pixels
    margin = abs(measurement - threshold)
    sensitivity_norm = float(np.sqrt(jacobian @ jacobian.T))
    
    # Prevents division by zero
    if sensitivity_norm < 1e-12:
        return float("inf")

    return margin / (k * sensitivity_norm)


def predicted_keypoint_sigma(jacobian: np.ndarray, per_keypoint_covariances: np.ndarray) -> float:
    norm = float(np.sqrt(jacobian @ jacobian.T))
    if norm < 1e-12:
        return 0.0
    return combined_uncertainty(jacobian, assemble_block_covariance(per_keypoint_covariances)) / norm


def is_confident(sigma_hat: float, tsi: float) -> bool:
    # confident if the predicted keypoint uncertainty is below the criterion's TSI, otherwise defer
    return sigma_hat < tsi


# MEASUREMENT FACTORS

def conformal_scale(abs_errors: np.ndarray, uncertainties: np.ndarray, coverage: float = 0.95) -> float:
    # s such that s*U covers `coverage` of cases; level = ceil((n+1)*coverage)/n, capped at 1
    ratios = np.asarray(abs_errors, dtype=float) / np.maximum(np.asarray(uncertainties, dtype=float), 1e-12)
    n = len(ratios)
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    return float(np.quantile(ratios, level, method="higher"))


def _load_criterion_field(field: str, path: str | Path | None) -> dict[str, float]:
    path = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    if not path.is_file():
        return {}
    block = (yaml.safe_load(path.read_text()) or {}).get(CONFIG_KEY) or {}
    return {key: float(entry[field]) for key, entry in block.items()}
