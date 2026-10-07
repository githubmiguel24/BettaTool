# per-criterion measurement factors 
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "training" / "configs" / "hrnet_w32.yaml"
CONFIG_ENV_VAR = "BETTA_CONFIG_PATH"
CONFIG_KEY = "measurement_factor"


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


def load_uncertainty_scales(path: str | Path | None = None) -> dict[str, float]:
    # per-criterion scales on the reported U(y); empty dict (=> all 1.0) if the config has no measurement_factor block
    return _load_criterion_field("scale", path)


def load_tsi_scales(path: str | Path | None = None) -> dict[str, float]:
    # per-criterion scales on sigma_hat for the TSI gate; empty dict (=> all 1.0) if the config has no measurement_factor block
    return _load_criterion_field("tsi_scale", path)
