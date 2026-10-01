# localization metrics (rmse, mre, pck, and confidence intervals) using pure numpy and scipy

from __future__ import annotations

import numpy as np
from scipy import stats


def radial_errors(pred: np.ndarray, gt: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    # get euclidean distance in pixels per keypoint and set invisible ones to nan
    err = np.linalg.norm(pred - gt, axis=-1)
    return np.where(visible_mask.astype(bool), err, np.nan)


def per_keypoint_rmse(pred: np.ndarray, gt: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    # rmse for each keypoint (K,) across visible samples only
    err = radial_errors(pred, gt, visible_mask)
    return np.sqrt(np.nanmean(err**2, axis=0))


def overall_rmse(pred: np.ndarray, gt: np.ndarray, visible_mask: np.ndarray) -> float:
    # single rmse value over all visible keypoints
    err = radial_errors(pred, gt, visible_mask)
    return float(np.sqrt(np.nanmean(err**2)))


def mean_and_median_radial_error(pred: np.ndarray, gt: np.ndarray, visible_mask: np.ndarray) -> tuple[float, float]:
    # returns both mean and median radial error in pixels
    err = radial_errors(pred, gt, visible_mask)
    return float(np.nanmean(err)), float(np.nanmedian(err))


def body_length(gt: np.ndarray, snout_idx: int, peduncle_top_idx: int, peduncle_bottom_idx: int) -> np.ndarray:
    # distance from snout tip to the peduncle midpoint to normalize pck by fish size
    peduncle_mid = 0.5 * (gt[:, peduncle_top_idx] + gt[:, peduncle_bottom_idx])
    return np.linalg.norm(gt[:, snout_idx] - peduncle_mid, axis=-1)


def pck_at_alpha(
    pred: np.ndarray,
    gt: np.ndarray,
    visible_mask: np.ndarray,
    alpha: float,
    snout_idx: int,
    peduncle_top_idx: int,
    peduncle_bottom_idx: int,
) -> dict[str, float]:
    # caclulate pck at alpha * body_length along with 95% wilson confidence bounds
    lengths = body_length(gt, snout_idx, peduncle_top_idx, peduncle_bottom_idx)  # (N,)
    threshold = alpha * lengths  # (N,)

    err = radial_errors(pred, gt, visible_mask)  # (N, K) with nan if not visible
    correct = err <= threshold[:, None]  # nan comparisons evaluate to False automatically
    valid = ~np.isnan(err)

    n_correct = int(np.sum(correct & valid))
    n_total = int(np.sum(valid))
    pck = n_correct / n_total if n_total > 0 else float("nan")
    ci_low, ci_high = wilson_score_interval(n_correct, n_total)
    return {"pck": pck, "n_correct": n_correct, "n_total": n_total, "ci_low": ci_low, "ci_high": ci_high}


def wilson_score_interval(n_correct: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    # wilson score interval for binomial proportions
    if n_total == 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    half_width = (z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denom
    # cast to standard python float so json.dumps doesnt complain later
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def clopper_pearson_interval(n_correct: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    # exact clopper-pearson Interval for when pck is close to 0% or 100%
    if n_total == 0:
        return float("nan"), float("nan")
    alpha = 1 - confidence
    low = 0.0 if n_correct == 0 else stats.beta.ppf(alpha / 2, n_correct, n_total - n_correct + 1)
    high = 1.0 if n_correct == n_total else stats.beta.ppf(1 - alpha / 2, n_correct + 1, n_total - n_correct)
    return float(low), float(high)


def bootstrap_ci(values: np.ndarray, statistic_fn=np.nanmean, n_resamples: int = 2000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    # percentile bootstrap confidence intervl for continuous metrics like rmse and mre
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    resampled = rng.choice(values, size=(n_resamples, values.size), replace=True)
    stat_dist = statistic_fn(resampled, axis=1)
    alpha = 1 - confidence
    return float(np.percentile(stat_dist, 100 * alpha / 2)), float(np.percentile(stat_dist, 100 * (1 - alpha / 2)))