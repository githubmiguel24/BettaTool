"""RQ1 — Average initial localization uncertainty, propagated geometric
margin of error, and per-criterion Threshold Sensitivity Index (TSI).

Populates Table 1 and Table 2 (Appendix 1.1). Requires a trained model and
an annotated test split; see `training/dataset.py`.
"""

from __future__ import annotations

import numpy as np

from app.analytical.gum_propagation import (
    assemble_block_covariance,
    combined_uncertainty,
    expanded_uncertainty,
)
from app.analytical.jacobian import numerical_jacobian
from app.analytical.morphometrics import MORPHOMETRIC_FUNCTIONS
from app.analytical.tsi import compute_tsi
from app.decisional.ibc_standards import CRITERION_THRESHOLDS


def compute_rmse(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """RMSE (pixels) between predicted and ground-truth (N, 2) keypoint arrays."""
    return float(np.sqrt(np.mean(np.sum((predicted - ground_truth) ** 2, axis=1))))


def run(test_set) -> dict:
    """Aggregates Table 1 & Table 2 values across the test set.

    Args:
        test_set: an iterable of (predicted_keypoints, covariances, ground_truth_keypoints),
            each shaped (N, 2), (N, 2, 2), (N, 2).

    TODO: wire `test_set` to the real test split once the model is trained.
    """
    rows = {key: {"rmse": [], "margin": [], "tsi": []} for key in MORPHOMETRIC_FUNCTIONS}

    for predicted, covariances, ground_truth in test_set:
        flat_keypoints = predicted.reshape(-1)
        block_covariance = assemble_block_covariance(covariances)
        rmse = compute_rmse(predicted, ground_truth)

        for criterion_key, measurement_fn in MORPHOMETRIC_FUNCTIONS.items():
            jacobian = numerical_jacobian(measurement_fn, flat_keypoints)
            measurement = measurement_fn(flat_keypoints)
            margin = expanded_uncertainty(combined_uncertainty(jacobian, block_covariance))
            tsi = compute_tsi(measurement, CRITERION_THRESHOLDS[criterion_key], jacobian)

            rows[criterion_key]["rmse"].append(rmse)
            rows[criterion_key]["margin"].append(margin)
            rows[criterion_key]["tsi"].append(tsi)

    return {
        key: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for key, metrics in rows.items()
    }


if __name__ == "__main__":
    raise SystemExit("Provide a real test_set iterable before running — see run() docstring.")
