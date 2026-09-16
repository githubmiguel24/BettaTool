"""Calibration metrics: NLL, Mahalanobis coverage, sigma-vs-error (Build Prompt v2 §10.2).

First-class, not optional — the downstream TSI gate assumes predicted
uncertainty is trustworthy, so these numbers are what justify that
assumption (or reveal that it does not hold).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from training.metrics.localization import wilson_score_interval


def mahalanobis_distance_sq(pred_mu: np.ndarray, pred_cov: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(N, K) squared Mahalanobis distance d^2 = (gt-mu)^T Sigma^-1 (gt-mu).

    Under a correctly-calibrated 2D Gaussian this follows a chi-squared
    distribution with 2 degrees of freedom (Build Prompt v2 §10.2).
    """
    residual = gt - pred_mu  # (N, K, 2)
    inv_cov = np.linalg.inv(pred_cov)  # (N, K, 2, 2)
    # d2[n, k] = residual[n, k] @ inv_cov[n, k] @ residual[n, k]
    tmp = np.einsum("nkij,nkj->nki", inv_cov, residual)  # (N, K, 2)
    d2 = np.einsum("nki,nki->nk", residual, tmp)  # (N, K)
    return d2


def mahalanobis_coverage(
    pred_mu: np.ndarray, pred_cov: np.ndarray, gt: np.ndarray, visible_mask: np.ndarray, coverage_level: float
) -> dict[str, float]:
    """Fraction of visible ground-truth points inside the predicted confidence ellipse.

    Args:
        pred_mu, pred_cov, gt: (N, K, 2) / (N, K, 2, 2) / (N, K, 2).
        visible_mask: (N, K) bool.
        coverage_level: nominal coverage, e.g. 0.68 or 0.95.

    Returns:
        dict with "empirical_coverage", "nominal_coverage", 95% Wilson CI
        ("ci_low", "ci_high"), and "n" (number of points evaluated).
    """
    d2 = mahalanobis_distance_sq(pred_mu, pred_cov, gt)
    threshold = stats.chi2.ppf(coverage_level, df=2)

    mask = visible_mask.astype(bool)
    inside = (d2 <= threshold) & mask
    n_total = int(mask.sum())
    n_inside = int(inside.sum())
    empirical = n_inside / n_total if n_total > 0 else float("nan")
    ci_low, ci_high = wilson_score_interval(n_inside, n_total)

    return {
        "empirical_coverage": empirical,
        "nominal_coverage": coverage_level,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n_total,
    }


def gaussian_nll_numpy(pred_mu: np.ndarray, pred_cov: np.ndarray, gt: np.ndarray, visible_mask: np.ndarray) -> float:
    """Mean Gaussian NLL over visible (sample, keypoint) pairs — NumPy mirror
    of `training/losses/gaussian_nll.py`'s formula, for evaluation-time
    reporting where a torch dependency is not otherwise needed."""
    det = np.linalg.det(pred_cov)
    d2 = mahalanobis_distance_sq(pred_mu, pred_cov, gt)
    nll = 0.5 * np.log(np.clip(det, 1e-12, None)) + 0.5 * d2
    mask = visible_mask.astype(bool)
    return float(nll[mask].mean()) if mask.any() else float("nan")


def fixed_sigma_baseline_nll(gt: np.ndarray, pred_mu: np.ndarray, visible_mask: np.ndarray, fixed_sigma_px: float) -> float:
    """NLL under an isotropic Gaussian of fixed sigma (context baseline, Build Prompt v2 §10.2)."""
    cov = np.tile((fixed_sigma_px**2) * np.eye(2), gt.shape[:-1] + (1, 1))
    return gaussian_nll_numpy(pred_mu, cov, gt, visible_mask)


def sigma_error_spearman_correlation(pred_cov: np.ndarray, radial_error: np.ndarray, visible_mask: np.ndarray) -> dict[str, float]:
    """Spearman correlation between predicted sigma (mean eigenvalue of Sigma)
    and actual radial error, overall — the direct evidence the covariance
    head is informative rather than decorative (Build Prompt v2 §10.2).

    `pred_cov`: (N, K, 2, 2). `radial_error`: (N, K), NaN for masked entries
    (as returned by `training.metrics.localization.radial_errors`).
    """
    eigvals = np.linalg.eigvalsh(pred_cov)  # (N, K, 2), ascending
    mean_sigma = np.sqrt(eigvals.mean(axis=-1))  # geometric-ish scale proxy, in px

    mask = visible_mask.astype(bool) & ~np.isnan(radial_error)
    if mask.sum() < 2:
        return {"spearman_r": float("nan"), "p_value": float("nan"), "n": int(mask.sum())}

    rho, p_value = stats.spearmanr(mean_sigma[mask], radial_error[mask])
    return {"spearman_r": float(rho), "p_value": float(p_value), "n": int(mask.sum())}


def reliability_bins(pred_cov: np.ndarray, radial_error: np.ndarray, visible_mask: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Bins predictions by predicted sigma; returns (mean predicted sigma, mean
    observed error) per bin, for the reliability plot (Build Prompt v2 §10.2)."""
    eigvals = np.linalg.eigvalsh(pred_cov)
    mean_sigma = np.sqrt(eigvals.mean(axis=-1))

    mask = visible_mask.astype(bool) & ~np.isnan(radial_error)
    sigma_vals, err_vals = mean_sigma[mask], radial_error[mask]
    if sigma_vals.size == 0:
        return []

    edges = np.quantile(sigma_vals, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6  # include the max value in the last bin
    bin_idx = np.digitize(sigma_vals, edges[1:-1])

    rows = []
    for b in range(n_bins):
        in_bin = bin_idx == b
        if in_bin.sum() == 0:
            continue
        rows.append(
            {
                "mean_predicted_sigma": float(sigma_vals[in_bin].mean()),
                "mean_observed_error": float(err_vals[in_bin].mean()),
                "n": int(in_bin.sum()),
            }
        )
    return rows
