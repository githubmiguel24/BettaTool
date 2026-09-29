# Threshold Sensitivity Index (TSI) -- legacy isotropic formulation, kept
# for the ablation baseline only (see is_confident_isotropic below).
# TSI = |y_hat - tau| / (k * sqrt(J J^T))
# critical keypoint RMSE (pixels) where k-sigma uncertainty reaches IBC
# boundary tau, assuming isotropic noise Sigma = I (see docstring below --
# `compute_tsi` never actually receives a covariance matrix).
#
# The pipeline's real decision rule is `is_confident_gum`, further down,
# which compares directly against the correctly-propagated anisotropic
# uncertainty (u_c = sqrt(J Sigma J^T), the real per-keypoint covariance)
# instead of this pixel-space proxy.

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
    """Legacy/baseline decision rule -- kept ONLY so
    `training/ablation_tsi_gate.py` can reconstruct it for comparison. Not
    used by the shipped pipeline anymore; see `is_confident_gum` below.

    This implicitly assumes Sigma = I (unit, isotropic, uncorrelated
    per-keypoint noise): `compute_tsi` above never receives a covariance
    matrix at all, only the Jacobian. And `actual_rmse` (see
    app/pipeline.py) is `sqrt(mean(diag(block_covariance)))` -- a single
    number blended across ALL 13 keypoints' variances, regardless of which
    keypoints this specific criterion's Jacobian is actually nonzero for.
    A criterion depending on 2-3 well-localized keypoints could get
    deferred because of noise in an unrelated keypoint (e.g. eye_center,
    which feeds no criterion at all); a criterion depending on genuinely
    poorly-localized keypoints could get waved through because other,
    irrelevant keypoints in the average were well-localized. Neither
    failure mode is visible from the outside, because this decision was
    never the same number reported to the user as the measurement's +/-
    margin of error (that number, `uncertainty` in
    app/decisional/abstention_gate.py, was already computed correctly from
    the real covariance -- see `is_confident_gum`).
    """
    return actual_rmse < tsi


# Backwards-compatible alias. Do not add new callers -- use
# `is_confident_gum` instead.
is_confident = is_confident_isotropic


def is_confident_gum(measurement: float, threshold: float, expanded_uncertainty: float) -> bool:
    """Confident iff the k-sigma expanded uncertainty U(y) = k*sqrt(J Sigma
    J^T) -- the SAME number already reported to the user as this
    measurement's +/- margin of error -- does not reach the IBC threshold.

    This is the corrected replacement for `is_confident_isotropic`. It uses
    the model's actual per-keypoint anisotropic covariance (including the
    correlation term rho, and restricted to the keypoints this criterion's
    Jacobian actually depends on, since the Jacobian is zero everywhere
    else in `combined_uncertainty`'s J*Sigma*J^T) instead of a Jacobian-only
    TSI compared against an RMSE blended across every annotated keypoint.
    It is also simpler: no separate pixel-space TSI/RMSE detour, just a
    direct comparison in the criterion's own units (degrees or a
    dimensionless ratio).

    See training/ablation_tsi_gate.py for the empirical comparison between
    this rule and the legacy isotropic one on the held-out test set.

    Strict `>`, not `>=`: when the margin exactly equals the expanded
    uncertainty, the confidence interval [measurement-U, measurement+U]
    just touches tau -- tau is not strictly excluded, so this defers rather
    than resolving the tie in the model's favor.
    """
    return abs(measurement - threshold) > expanded_uncertainty