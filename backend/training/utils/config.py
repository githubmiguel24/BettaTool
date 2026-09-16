"""Single YAML config system: load, deep-merge, and resolve dotted keys.

Every hyperparameter, path, and convention used anywhere in `training/` or
`app/perception/` traces back to a key in `training/configs/base.yaml` or an
experiment override — never a hardcoded literal in code (Build Prompt v2 §1).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_BASE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merges `override` into `base`, returning a new dict.

    Scalars and lists in `override` replace the corresponding value in
    `base` outright; nested dicts are merged key-by-key.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(experiment_config_path: str | Path, base_config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads `base.yaml`, deep-merges the experiment config on top, and returns it.

    Args:
        experiment_config_path: path to an experiment YAML (e.g.
            `training/configs/hrnet_w32.yaml`).
        base_config_path: override for the shared-defaults file; defaults to
            `training/configs/base.yaml` next to this module.

    Returns:
        The fully resolved config as a plain nested dict.

    Raises:
        FileNotFoundError: if either file does not exist.
        ValueError: if either file does not parse to a mapping.
    """
    base_path = Path(base_config_path) if base_config_path is not None else _BASE_CONFIG_PATH
    exp_path = Path(experiment_config_path)

    if not base_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    if not exp_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")

    base_cfg = yaml.safe_load(base_path.read_text())
    exp_cfg = yaml.safe_load(exp_path.read_text())

    if not isinstance(base_cfg, dict):
        raise ValueError(f"{base_path} did not parse to a mapping.")
    if not isinstance(exp_cfg, dict):
        raise ValueError(f"{exp_path} did not parse to a mapping.")

    resolved = _deep_merge(base_cfg, exp_cfg)
    resolved["_resolved_from"] = {"base": str(base_path), "experiment": str(exp_path)}
    return resolved


def get(config: dict[str, Any], dotted_key: str, default: Any = ...) -> Any:
    """Reads a nested config value using a dotted path, e.g. `get(cfg, "training.batch_size")`."""
    node: Any = config
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            if default is ...:
                raise KeyError(f"Config key not found: {dotted_key!r}")
            return default
    return node
