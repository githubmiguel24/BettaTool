# metrics for evaluating uncertainty and coverage (nll, mahalanobis, spearman)

from __future__ import annotations

import numpy as np
from scipy import stats

from training.metrics.localization import wilson_score_interval


# squared mahalanobis distance d^2 = (gt-mu)^T Sigma^-1 (gt-mu), follows chi2(df=2)
def mahalanobis_distance_sq(pred_mu: np.ndarray, pred_cov: np.ndarray, gt: np.ndarray) -> np.ndarray:
    residual = gt - pred_mu  # (N, K, 2)
    inv_cov = np.linalg.inv(pred_cov)  # (N, K, 2, 2)
    # d2[n, k] = residual[n, k] @ inv_cov[n, k] @ residual[n, k]
    tmp = np.einsum("nkij,nkj->nki", inv_cov, residual)  # (N, K, 2)
    d2 = np.einsum("nki,nki->nk", residual, tmp)  # (N, K)
    return d2


# fraction of visible ground truth points falling inside the confidence elipse
def mahalanobis_coverage(
    pred_mu: np.ndarray, pred_cov: np.ndarray, gt: np.ndarray, visible_mask: np.ndarray, coverage_level: float
) -> dict[str, float]:
    d2 = mahalanobis_distance_sq(pred_mu, pred_cov, gt)
    threshold = stats.chi2.ppf(coverage_level, df=2)  # chi2 cutoff for 2d gaussian

    mask = visible_mask.astype(bool)
    inside = (d2 <= threshold) & mask
    n_total = int(mask.sum())
    n_inside = int(inside.sum())
    empirical = n_inside / n_total if n_total > 0 else float("nan")
    ci_low, ci_high = wilson_score_interval(n_inside, n_total)  # 95% wilson CI

    return {
        "empirical_coverage": empirical,
        "nominal_coverage": coverage_level,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n_total,
    }


# mean gaussian NLL over visible pairs in numpy so we dont need torch at eval
def gaussian_nll_numpy(pred_mu: np.ndarray, pred_cov: np.ndarray, gt: np.ndarray, visible_mask: np.ndarray) -> float:
    det = np.linalg.det(pred_cov)
    d2 = mahalanobis_distance_sq(pred_mu, pred_cov, gt)
    nll = 0.5 * np.log(np.clip(det, 1e-12, None)) + 0.5 * d2  # clip det to avoid log(0)
    mask = visible_mask.astype(bool)
    return float(nll[mask].mean()) if mask.any() else float("nan")


# baseline NLL using a fixed isotropic sigma for comparison
def fixed_sigma_baseline_nll(gt: np.ndarray, pred_mu: np.ndarray, visible_mask: np.ndarray, fixed_sigma_px: float) -> float:
    cov = np.tile((fixed_sigma_px**2) * np.eye(2), gt.shape[:-1] + (1, 1))  # build isotropic cov
    return gaussian_nll_numpy(pred_mu, cov, gt, visible_mask)


# spearman corr between predicted sigma and radial error to check if cov head is useful
def sigma_error_spearman_correlation(pred_cov: np.ndarray, radial_error: np.ndarray, visible_mask: np.ndarray) -> dict[str, float]:
    eigvals = np.linalg.eigvalsh(pred_cov)  # (N, K, 2), ascending
    mean_sigma = np.sqrt(eigvals.mean(axis=-1))  # scale proxy in px

    mask = visible_mask.astype(bool) & ~np.isnan(radial_error)  # drop masked and NaNs
    if mask.sum() < 2:
        return {"spearman_r": float("nan"), "p_value": float("nan"), "n": int(mask.sum())}

    rho, p_value = stats.spearmanr(mean_sigma[mask], radial_error[mask])
    return {"spearman_r": float(rho), "p_value": float(p_value), "n": int(mask.sum())}


# bin predictions by sigma to get mean predicted sigma vs observed err for reliability plot
def reliability_bins(pred_cov: np.ndarray, radial_error: np.ndarray, visible_mask: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
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