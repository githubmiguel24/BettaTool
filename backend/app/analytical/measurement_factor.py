# =============================================================================
# Program Title : measurement_factor.py - Per-Criterion Calibration Factors
# Programmers   : Trinidad, C. M. L.; Daanay, K. M. M.; Verceles, R. C. S.
# System Fit    : Betta Fish Quality Assessment Model > Analytical Tier (support).
#                 app/pipeline.py loads the scales when AssessmentPipeline is
#                 created: one scale for the reported margin of error U(y) and
#                 one for sigma_hat in the TSI gate (tsi.py). Both are stored in
#                 the "measurement_factor" block of training/configs/hrnet_w32.yaml.
# Written       : September 10, 2026
# Revised       : October 6, 2026 (last saved);
#                 October 14, 2026 (header block and comments for documentation)
# Purpose       : Loads, and on validation data fits, the per-criterion scales
#                 that make the reported uncertainty and the TSI gate match the
#                 errors actually observed. If no scale is configured, a
#                 criterion keeps a scale of 1.0 (no correction).
# Data / Algo   : Data: dict[str, float] (criterion key -> scale); the parsed
#                 YAML dictionary; Path objects; NumPy arrays of absolute errors
#                 and uncertainties; constants DEFAULT_CONFIG_PATH,
#                 CONFIG_ENV_VAR and CONFIG_KEY.
#                 Algorithm: conformal calibration. For each case, ratio =
#                 |error| / U; the scale is the quantile of the ratios at level
#                 ceil((n + 1) * coverage) / n (capped at 1), so scale * U covers
#                 about 95% of cases. Config path: argument, then the
#                 BETTA_CONFIG_PATH environment variable, then the default file.
#                 Control: sequence; a guard clause when the file is missing;
#                 one dictionary comprehension.
# =============================================================================

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "training" / "configs" / "hrnet_w32.yaml"
)
CONFIG_ENV_VAR = "BETTA_CONFIG_PATH"
CONFIG_KEY = "measurement_factor"

# Smallest uncertainty used as a divisor, to avoid division by zero.
_MIN_UNCERTAINTY = 1e-12


def conformal_scale(
    abs_errors: np.ndarray,
    uncertainties: np.ndarray,
    coverage: float = 0.95,
) -> float:
    """Return the scale s such that s * U covers `coverage` of the cases.

    The quantile level is ceil((n + 1) * coverage) / n, capped at 1.
    """
    errors = np.asarray(abs_errors, dtype=float)
    uncertain = np.maximum(np.asarray(uncertainties, dtype=float), _MIN_UNCERTAINTY)
    ratios = errors / uncertain
    n = len(ratios)
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    return float(np.quantile(ratios, level, method="higher"))


def _load_criterion_field(field: str, path: str | Path | None) -> dict[str, float]:
    """Read one field (e.g. "scale") of every criterion in the config file."""
    path = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    if not path.is_file():
        # No config file: callers then use a scale of 1.0 for every criterion.
        return {}
    block = (yaml.safe_load(path.read_text()) or {}).get(CONFIG_KEY) or {}
    return {key: float(entry[field]) for key, entry in block.items()}


def load_uncertainty_scales(path: str | Path | None = None) -> dict[str, float]:
    """Return the per-criterion scales applied to the reported U(y).

    An empty dict (all scales 1.0) is returned if the config has no
    measurement_factor block.
    """
    return _load_criterion_field("scale", path)