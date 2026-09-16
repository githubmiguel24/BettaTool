"""Localization metrics: RMSE, MRE, PCK (Build Prompt v2 §10.1).

Pure NumPy + SciPy — no torch dependency, so these are testable and usable
standalone in a notebook when writing up Chapter 3 results.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def radial_errors(pred: np.ndarray, gt: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    """Per-sample, per-keypoint Euclidean distance between prediction and ground truth.

    Args:
        pred: (N, K, 2) predicted coordinates, ORIGINAL-image pixels.
        gt: (N, K, 2) ground-truth coordinates, ORIGINAL-image pixels.
        visible_mask: (N, K) bool/float, True/1.0 for keypoints that have a
            valid ground-truth location (visibility flag 2 or 1).

    Returns:
        (N, K) array of radial errors in pixels; entries where
        `visible_mask` is falsy are NaN (so downstream `np.nanmean` etc.
        naturally skip them, rather than treating a masked keypoint as a
        zero-error observation).
    """
    err = np.linalg.norm(pred - gt, axis=-1)
    return np.where(visible_mask.astype(bool), err, np.nan)


def per_keypoint_rmse(pred: np.ndarray, gt: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    """(K,) RMSE per keypoint, in original-image pixels, over visible instances only.

    RMSE here is defined per the manuscript's Statistical Treatment section
    convention: RMSE_k = sqrt(mean over visible samples of squared radial
    error for keypoint k). See training/README.md for the exact page
    citation to keep in Chapter 3.
    """
    err = radial_errors(pred, gt, visible_mask)
    return np.sqrt(np.nanmean(err**2, axis=0))


def overall_rmse(pred: np.ndarray, gt: np.ndarray, visible_mask: np.ndarray) -> float:
    """Scalar RMSE across all visible (sample, keypoint) pairs."""
    err = radial_errors(pred, gt, visible_mask)
    return float(np.sqrt(np.nanmean(err**2)))


def mean_and_median_radial_error(pred: np.ndarray, gt: np.ndarray, visible_mask: np.ndarray) -> tuple[float, float]:
    """(mean, median) radial error in pixels, over all visible (sample, keypoint) pairs."""
    err = radial_errors(pred, gt, visible_mask)
    return float(np.nanmean(err)), float(np.nanmedian(err))


def body_length(gt: np.ndarray, snout_idx: int, peduncle_top_idx: int, peduncle_bottom_idx: int) -> np.ndarray:
    """(N,) body length per sample: snout_tip -> midpoint(peduncle_top, peduncle_bottom).

    Used to normalize PCK@alpha so the threshold scales with fish size in
    the photo rather than being a fixed pixel count (Build Prompt v2 §10.1).
    """
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
    """Percentage of Correct Keypoints at threshold `alpha * body_length`.

    Returns a dict with "pck" (float in [0, 1]) and a Wilson-score
    `"ci_low"`/`"ci_high"` 95% confidence interval (Build Prompt v2 §10.4).
    """
    lengths = body_length(gt, snout_idx, peduncle_top_idx, peduncle_bottom_idx)  # (N,)
    threshold = alpha * lengths  # (N,)

    err = radial_errors(pred, gt, visible_mask)  # (N, K), NaN where invisible
    correct = err <= threshold[:, None]  # broadcasts; NaN <= x is False, which is correct (excluded)
    valid = ~np.isnan(err)

    n_correct = int(np.sum(correct & valid))
    n_total = int(np.sum(valid))
    pck = n_correct / n_total if n_total > 0 else float("nan")
    ci_low, ci_high = wilson_score_interval(n_correct, n_total)
    return {"pck": pck, "n_correct": n_correct, "n_total": n_total, "ci_low": ci_low, "ci_high": ci_high}


def wilson_score_interval(n_correct: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (Build Prompt v2 §10.4).

    Preferred over the naive normal-approximation interval near 0 or 1. At
    n=225 (the target test-set size) and p~0.90, the half-width is
    approximately +-3.9 percentage points, as noted in the spec — smaller
    differences between two models/ablations are not distinguishable at
    that sample size.
    """
    if n_total == 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    half_width = (z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denom
    # Explicit float() casts: z (and everything derived from it) is a plain
    # Python float once cast above, so center/half_width stay plain floats
    # too — this keeps every metrics function's output JSON-serializable
    # (training/evaluate.py writes these straight into metrics.json)
    # without a numpy.float64 tripping json.dumps.
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def clopper_pearson_interval(n_correct: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson exact interval — use near the 0%/100% boundary where
    Wilson can behave poorly (Build Prompt v2 §10.4)."""
    if n_total == 0:
        return float("nan"), float("nan")
    alpha = 1 - confidence
    low = 0.0 if n_correct == 0 else stats.beta.ppf(alpha / 2, n_correct, n_total - n_correct + 1)
    high = 1.0 if n_correct == n_total else stats.beta.ppf(1 - alpha / 2, n_correct + 1, n_total - n_correct)
    return float(low), float(high)


def bootstrap_ci(values: np.ndarray, statistic_fn=np.nanmean, n_resamples: int = 2000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a continuous metric (e.g. RMSE, MRE) — Build Prompt v2 §10.4."""
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    resampled = rng.choice(values, size=(n_resamples, values.size), replace=True)
    stat_dist = statistic_fn(resampled, axis=1)
    alpha = 1 - confidence
    return float(np.percentile(stat_dist, 100 * alpha / 2)), float(np.percentile(stat_dist, 100 * (1 - alpha / 2)))
