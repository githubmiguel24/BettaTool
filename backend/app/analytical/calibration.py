# post-hoc per-criterion scaling of the expanded uncertainty (split-conformal style)
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[2] / "training" / "runs" / "kaggle_run" / "calibration.json"
)
CALIBRATION_ENV_VAR = "BETTA_CALIBRATION_PATH"


def conformal_scale(abs_errors: np.ndarray, uncertainties: np.ndarray, coverage: float = 0.95) -> float:
    # s such that s*U covers `coverage` of cases; level = ceil((n+1)*coverage)/n, capped at 1
    ratios = np.asarray(abs_errors, dtype=float) / np.maximum(np.asarray(uncertainties, dtype=float), 1e-12)
    n = len(ratios)
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    return float(np.quantile(ratios, level, method="higher"))


def load_uncertainty_scales(path: str | Path | None = None) -> dict[str, float]:
    # per-criterion scales from calibration.json; empty dict (=> all 1.0) if the file is absent
    path = Path(path or os.environ.get(CALIBRATION_ENV_VAR) or DEFAULT_CALIBRATION_PATH)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {key: float(entry["scale"]) for key, entry in data.get("criteria", {}).items()}
