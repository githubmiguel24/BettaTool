"""Grouped, stratified train/val/test splitting (Build Prompt v2 §5.1).

Pure Python + NumPy — no torch/albumentations dependency, so this module is
fully testable in any environment. Splits are computed once and written to
disk as explicit JSON ID lists (`write_splits` / `load_splits`); nothing in
`training/dataset.py` ever regenerates a split at runtime.

Grouping is mandatory: multiple photographs of the same physical fish
(`specimen_id`) must land in exactly one partition, never split across
train/val/test, or identity leaks across partitions and inflates reported
accuracy (Yagis et al., 2021; Bussola et al., 2021 — cited in Build Prompt
v2 §5.1). Stratification is applied on the group level using each group's
*first* sample's strata values, joined into one composite key, so groups
with many samples don't get a size-1 "stratum" that can't be split evenly.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def compute_grouped_stratified_split(
    samples: list[dict[str, Any]],
    group_key: str,
    stratify_keys: list[str],
    split_fractions: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    """Splits `samples` into named partitions, grouped by `group_key`.

    Args:
        samples: one dict per annotated image, each with at least an
            `"image_id"` key, the `group_key` field, and every field named
            in `stratify_keys`.
        group_key: field name whose value must not appear in more than one
            partition (e.g. "specimen_id").
        stratify_keys: field names combined into one composite stratum key
            per group (e.g. ["source", "color_morph", "has_occlusion"]).
        split_fractions: partition name -> fraction, e.g.
            `{"train": 0.70, "val": 0.15, "test": 0.15}`. Fractions need not
            sum to exactly 1.0 (they are renormalized), and any number of
            partitions is supported — this is what lets a future 4th
            "calib" partition be added by editing config alone.
        seed: RNG seed for the shuffle within each stratum, for reproducibility.

    Returns:
        Dict mapping each partition name (same keys as `split_fractions`)
        to a list of `image_id` strings.

    Raises:
        ValueError: if `samples` is empty, if `split_fractions` is empty, or
            if any sample is missing `group_key` or a stratify key.
    """
    if not samples:
        raise ValueError("compute_grouped_stratified_split received zero samples.")
    if not split_fractions:
        raise ValueError("split_fractions must not be empty.")

    partition_names = list(split_fractions.keys())
    total_fraction = sum(split_fractions.values())
    normalized_fractions = {k: v / total_fraction for k, v in split_fractions.items()}

    # 1. Group samples by group_key, and compute each group's stratum key
    #    from its FIRST sample (all samples in a group are assumed to share
    #    strata, e.g. the same fish has one color morph).
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if group_key not in sample:
            raise ValueError(f"Sample {sample.get('image_id', '?')} is missing group_key '{group_key}'.")
        groups[str(sample[group_key])].append(sample)

    strata: dict[str, list[str]] = defaultdict(list)  # stratum_key -> [group_id, ...]
    for group_id, group_samples in groups.items():
        first = group_samples[0]
        for stratify_key in stratify_keys:
            if stratify_key not in first:
                raise ValueError(f"Group '{group_id}' sample is missing stratify key '{stratify_key}'.")
        stratum_key = "|".join(str(first[k]) for k in stratify_keys)
        strata[stratum_key].append(group_id)

    # 2. Within each stratum, shuffle groups deterministically and cut into
    #    partitions proportional to normalized_fractions. Using groups (not
    #    raw samples) as the unit being cut is what keeps a specimen's
    #    photos together.
    rng = random.Random(seed)
    partition_group_ids: dict[str, list[str]] = {name: [] for name in partition_names}

    for stratum_key in sorted(strata.keys()):  # sorted: deterministic across platforms
        group_ids = list(strata[stratum_key])
        rng.shuffle(group_ids)

        n = len(group_ids)
        cursor = 0
        cumulative = 0.0
        for i, name in enumerate(partition_names):
            cumulative += normalized_fractions[name]
            end = n if i == len(partition_names) - 1 else round(cumulative * n)
            partition_group_ids[name].extend(group_ids[cursor:end])
            cursor = end

    # 3. Expand group ids back to image ids.
    partitions: dict[str, list[str]] = {name: [] for name in partition_names}
    for name, group_ids in partition_group_ids.items():
        for group_id in group_ids:
            partitions[name].extend(str(s["image_id"]) for s in groups[group_id])

    return partitions


def assert_no_group_leakage(partitions: dict[str, list[str]], samples: list[dict[str, Any]], group_key: str) -> None:
    """Raises AssertionError if any group_key value appears in more than one partition."""
    image_id_to_group = {str(s["image_id"]): str(s[group_key]) for s in samples}
    group_to_partitions: dict[str, set[str]] = defaultdict(set)
    for partition_name, image_ids in partitions.items():
        for image_id in image_ids:
            group_to_partitions[image_id_to_group[image_id]].add(partition_name)

    leaked = {g: p for g, p in group_to_partitions.items() if len(p) > 1}
    if leaked:
        raise AssertionError(f"{group_key} values split across partitions (leakage): {leaked}")


def write_splits(splits_dir: str | Path, partitions: dict[str, list[str]]) -> None:
    """Writes each partition to `<splits_dir>/<name>.json` as a sorted ID list."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, image_ids in partitions.items():
        (splits_dir / f"{name}.json").write_text(json.dumps(sorted(image_ids), indent=2))


def load_splits(splits_dir: str | Path, partition_names: list[str]) -> dict[str, list[str]]:
    """Reads back the JSON ID lists written by `write_splits`.

    Raises FileNotFoundError with a clear message if a split has not been
    generated yet — splits are never silently regenerated at runtime
    (Build Prompt v2 §5.1).
    """
    splits_dir = Path(splits_dir)
    partitions = {}
    for name in partition_names:
        path = splits_dir / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Split file not found: {path}. Splits are computed once and stored on "
                "disk, never regenerated at runtime — run "
                "`python -m training.splitting --ann <file>` (or the equivalent step in "
                "your data-prep script) first."
            )
        partitions[name] = json.loads(path.read_text())
    return partitions


def composition_table(
    partitions: dict[str, list[str]], samples: list[dict[str, Any]], stratify_keys: list[str]
) -> list[dict[str, Any]]:
    """Builds the split-composition table (counts per split x stratum) for
    Chapter 3 of the manuscript (Build Prompt v2 §5.1).

    Returns a list of rows, each a dict with keys "partition", one column
    per `stratify_keys` entry (or "*" if not stratified within this row —
    here every row is a single stratum combination), and "count".
    """
    by_image_id = {str(s["image_id"]): s for s in samples}
    rows: list[dict[str, Any]] = []
    for partition_name, image_ids in partitions.items():
        counts: dict[tuple[str, ...], int] = defaultdict(int)
        for image_id in image_ids:
            sample = by_image_id[image_id]
            key = tuple(str(sample[k]) for k in stratify_keys)
            counts[key] += 1
        for key, count in sorted(counts.items()):
            row = {"partition": partition_name, "count": count}
            row.update(dict(zip(stratify_keys, key)))
            rows.append(row)
    return rows


def main() -> None:
    """CLI: computes and writes the train/val/test split once, from config.

        python -m training.splitting --ann data/annotations/annotations.json \\
            --config training/configs/hrnet_w32.yaml

    Refuses to overwrite an existing split unless `--force` is passed —
    splits are meant to be computed once and then frozen (Build Prompt v2 §5.1).
    """
    from training.utils.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite existing split files.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits_dir = Path(cfg["paths"]["splits_dir"])
    if not args.force and any((splits_dir / f"{name}.json").exists() for name in cfg["dataset"]["splits"]):
        raise SystemExit(
            f"Split files already exist under {splits_dir}. Splits are frozen once written "
            "(Build Prompt v2 §5.1) — pass --force to intentionally regenerate them."
        )

    samples = json.loads(Path(args.ann).read_text())["annotations"]
    partitions = compute_grouped_stratified_split(
        samples,
        group_key=cfg["dataset"]["group_by"],
        stratify_keys=cfg["dataset"]["stratify_by"],
        split_fractions=cfg["dataset"]["splits"],
        seed=cfg["seed"],
    )
    assert_no_group_leakage(partitions, samples, cfg["dataset"]["group_by"])
    write_splits(splits_dir, partitions)

    print(f"Wrote splits to {splits_dir}: " + ", ".join(f"{k}={len(v)}" for k, v in partitions.items()))
    for row in composition_table(partitions, samples, cfg["dataset"]["stratify_by"]):
        print(" ", row)


if __name__ == "__main__":
    main()
