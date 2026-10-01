# yaml config loader that merges experiment configs into base.yaml and handles doted key lookups

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# default path to the base config file
_BASE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    # recursively merge override into base, Lists and scalars get replaced outright
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(experiment_config_path: str | Path, base_config_path: str | Path | None = None) -> dict[str, Any]:
    # loads base.yaml and merges the expriment config on top
    base_path = Path(base_config_path) if base_config_path is not None else _BASE_CONFIG_PATH
    exp_path = Path(experiment_config_path)

    # make sure both config files actually exist before reading
    if not base_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    if not exp_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")

    base_cfg = yaml.safe_load(base_path.read_text())
    exp_cfg = yaml.safe_load(exp_path.read_text())

    # sanity check that the yaml files parsed into dicts
    if not isinstance(base_cfg, dict):
        raise ValueError(f"{base_path} did not parse to a mapping.")
    if not isinstance(exp_cfg, dict):
        raise ValueError(f"{exp_path} did not parse to a mapping.")

    # combine configs and track which files were used
    resolved = _deep_merge(base_cfg, exp_cfg)
    resolved["_resolved_from"] = {"base": str(base_path), "experiment": str(exp_path)}
    return resolved


def get(config: dict[str, Any], dotted_key: str, default: Any = ...) -> Any:
    # grab a nested config value using dot notation like "training.batch_size"
    node: Any = config
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            # raise an error if no default was passed in
            if default is ...:
                raise KeyError(f"Config key not found: {dotted_key!r}")
            return default
    return node