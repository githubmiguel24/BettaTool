"""Heatmap target rendering + localization/calibration metrics (Build Prompt v2 §11)."""

from __future__ import annotations

import numpy as np

from training.metrics.calibration import (
    fixed_sigma_baseline_nll,
    gaussian_nll_numpy,
    mahalanobis_coverage,
    sigma_error_spearman_correlation,
)
from training.metrics.localization import (
    clopper_pearson_interval,
    mean_and_median_radial_error,
    pck_at_alpha,
    per_keypoint_rmse,
    radial_errors,
    wilson_score_interval,
)
from training.targets import render_all_targets, render_gaussian_heatmap


def test_gaussian_heatmap_peak_at_center() -> None:
    heatmap = render_gaussian_heatmap(heatmap_size=64, center_heatmap_space=(31.6, 20.2), sigma_px=2.0)
    peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    assert (peak_x, peak_y) == (32, 20)
    assert heatmap.max() <= 1.0 + 1e-6


def test_render_all_targets_masks_invisible_keypoints() -> None:
    centers = np.array([[10.0, 10.0], [50.0, 50.0]])
    visible = np.array([True, False])
    targets = render_all_targets(heatmap_size=64, centers_heatmap_space=centers, sigma_px=2.0, visibility_mask=visible)
    assert targets[0].max() > 0.9
    assert targets[1].max() == 0.0


def test_rmse_matches_known_gaussian_noise() -> None:
    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 100, size=(300, 2, 2))
    pred = gt + rng.normal(0, 3.0, size=(300, 2, 2))
    vis = np.ones((300, 2))
    rmse = per_keypoint_rmse(pred, gt, vis)
    # E[radial_error^2] = 2 * sigma^2 for isotropic 2D Gaussian noise.
    assert np.allclose(rmse, np.sqrt(2) * 3.0, atol=0.5)


def test_masked_keypoints_excluded_from_rmse() -> None:
    gt = np.zeros((5, 1, 2))
    pred = np.ones((5, 1, 2)) * 1000  # huge error
    vis = np.zeros((5, 1))  # all masked
    err = radial_errors(pred, gt, vis)
    assert np.all(np.isnan(err))


def test_pck_extremes() -> None:
    rng = np.random.default_rng(1)
    gt = rng.uniform(0, 100, size=(50, 3, 2))
    pred = gt.copy()
    vis = np.ones((50, 3))
    result = pck_at_alpha(pred, gt, vis, alpha=0.1, snout_idx=0, peduncle_top_idx=1, peduncle_bottom_idx=2)
    assert result["pck"] == 1.0  # exact match -> always within any positive threshold


def test_wilson_interval_contains_point_estimate() -> None:
    low, high = wilson_score_interval(90, 100)
    assert low < 0.9 < high


def test_clopper_pearson_boundary_cases_do_not_crash() -> None:
    assert clopper_pearson_interval(0, 100)[0] == 0.0
    assert clopper_pearson_interval(100, 100)[1] == 1.0


def test_mahalanobis_coverage_matches_nominal_for_correct_sigma() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    gt = rng.uniform(0, 100, size=(n, 1, 2))
    sigma = 4.0
    pred = gt + rng.normal(0, sigma, size=(n, 1, 2))
    cov = np.tile((sigma**2) * np.eye(2), (n, 1, 1, 1))
    vis = np.ones((n, 1))

    result = mahalanobis_coverage(pred, cov, gt, vis, coverage_level=0.95)
    assert abs(result["empirical_coverage"] - 0.95) < 0.02


def test_gaussian_nll_numpy_matches_fixed_sigma_baseline_when_cov_is_fixed() -> None:
    rng = np.random.default_rng(3)
    gt = rng.uniform(0, 50, size=(20, 1, 2))
    pred = gt + rng.normal(0, 2.0, size=(20, 1, 2))
    cov = np.tile((2.0**2) * np.eye(2), (20, 1, 1, 1))
    vis = np.ones((20, 1))

    nll = gaussian_nll_numpy(pred, cov, gt, vis)
    baseline = fixed_sigma_baseline_nll(gt, pred, vis, fixed_sigma_px=2.0)
    assert np.isclose(nll, baseline)


def test_sigma_error_correlation_is_positive_for_heteroscedastic_noise() -> None:
    rng = np.random.default_rng(4)
    n = 500
    true_sigma = rng.uniform(1, 8, size=(n, 1))
    gt = rng.uniform(0, 200, size=(n, 1, 2))
    pred = gt + rng.normal(0, 1, size=(n, 1, 2)) * true_sigma[..., None]
    cov = np.zeros((n, 1, 2, 2))
    cov[..., 0, 0] = true_sigma**2
    cov[..., 1, 1] = true_sigma**2
    vis = np.ones((n, 1))

    err = radial_errors(pred, gt, vis)
    result = sigma_error_spearman_correlation(cov, err, vis)
    assert result["spearman_r"] > 0.5
