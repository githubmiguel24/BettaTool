# handles grouped and stratified dataset splitting into train/val/test json files so same fish or multi-fish images dont leak across splits

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


# union-find to group samples that share a specimen_id or show up in the same image_id
def _union_find_groups(samples: list[dict[str, Any]], group_key: str) -> dict[int, int]:
    parent = list(range(len(samples)))

    # path compression find
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    first_seen_by_group: dict[str, int] = {}
    first_seen_by_image: dict[str, int] = {}
    for i, sample in enumerate(samples):
        if group_key not in sample:
            raise ValueError(f"Sample {sample.get('image_id', '?')} is missing group_key '{group_key}'.")
        gk = str(sample[group_key])
        # merge if we already saw this specimen
        if gk in first_seen_by_group:
            union(i, first_seen_by_group[gk])
        else:
            first_seen_by_group[gk] = i

        img = str(sample["image_id"])
        # also merge annotations from the same photo so backgrounds dont leak
        if img in first_seen_by_image:
            union(i, first_seen_by_image[img])
        else:
            first_seen_by_image[img] = i

    return {i: find(i) for i in range(len(samples))}


# splits samples into partitions while keeping groups together and balancing strata
def compute_grouped_stratified_split(
    samples: list[dict[str, Any]],
    group_key: str,
    stratify_keys: list[str],
    split_fractions: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    if not samples:
        raise ValueError("compute_grouped_stratified_split received zero samples.")
    if not split_fractions:
        raise ValueError("split_fractions must not be empty.")

    partition_names = list(split_fractions.keys())
    total_fraction = sum(split_fractions.values())
    # normalize fractions just in case they dont add up to 1
    normalized_fractions = {k: v / total_fraction for k, v in split_fractions.items()}

    # group samples by specimen/image and grab the stratum key from the first item in each group
    roots = _union_find_groups(samples, group_key)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[roots[i]].append(sample)

    strata: dict[str, list[str]] = defaultdict(list)  # maps composite stratum key to group ids
    for group_id, group_samples in groups.items():
        first = group_samples[0]
        for stratify_key in stratify_keys:
            if stratify_key not in first:
                raise ValueError(f"Group '{group_id}' sample is missing stratify key '{stratify_key}'.")
        stratum_key = "|".join(str(first[k]) for k in stratify_keys)
        strata[stratum_key].append(group_id)

    # shuffle groups inside each Stratum and slice into partitons
    rng = random.Random(seed)
    partition_group_ids: dict[str, list[str]] = {name: [] for name in partition_names}

    for stratum_key in sorted(strata.keys()):  # sort keys so rng stays deterministic
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

    # unpack group ids back into image_id lists
    partitions: dict[str, list[str]] = {name: [] for name in partition_names}
    for name, group_ids in partition_group_ids.items():
        for group_id in group_ids:
            partitions[name].extend(str(s["image_id"]) for s in groups[group_id])

    return partitions


# sanity check that no image_id or group_key ended up in multiple splits
def assert_no_group_leakage(partitions: dict[str, list[str]], samples: list[dict[str, Any]], group_key: str) -> None:
    image_id_to_partitions: dict[str, set[str]] = defaultdict(set)
    for partition_name, image_ids in partitions.items():
        for image_id in image_ids:
            image_id_to_partitions[image_id].add(partition_name)

    # check image leakage first
    leaked_images = {img: p for img, p in image_id_to_partitions.items() if len(p) > 1}
    if leaked_images:
        raise AssertionError(f"image_id values split across partitions (leakage): {leaked_images}")

    # now check if any specimen_id leaked across splits
    image_id_to_partition = {img: next(iter(p)) for img, p in image_id_to_partitions.items()}
    group_to_partitions: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        image_id = str(sample["image_id"])
        partition = image_id_to_partition.get(image_id)
        if partition is not None:
            group_to_partitions[str(sample[group_key])].add(partition)

    leaked_groups = {g: p for g, p in group_to_partitions.items() if len(p) > 1}
    if leaked_groups:
        raise AssertionError(f"{group_key} values split across partitions (leakage): {leaked_groups}")


# dump each split as a sorted json list of image ids
def write_splits(splits_dir: str | Path, partitions: dict[str, list[str]]) -> None:
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, image_ids in partitions.items():
        (splits_dir / f"{name}.json").write_text(json.dumps(sorted(image_ids), indent=2))


# load precomputed split json files from disk
def load_splits(splits_dir: str | Path, partition_names: list[str]) -> dict[str, list[str]]:
    splits_dir = Path(splits_dir)
    partitions = {}
    for name in partition_names:
        path = splits_dir / f"{name}.json"
        # fail fast if the split file hasn't been generated yet
        if not path.is_file():
            raise FileNotFoundError(
                f"Split file not found: {path}. Splits are computed once and stored on "
                "disk, never regenerated at runtime — run "
                "`python -m training.splitting --ann <file>` (or the equivalent step in "
                "your data-prep script) first."
            )
        partitions[name] = json.loads(path.read_text())
    return partitions


# get counts per split and stratum for logging and tables
def composition_table(
    partitions: dict[str, list[str]], samples: list[dict[str, Any]], stratify_keys: list[str]
) -> list[dict[str, Any]]:
    by_image_id = {str(s["image_id"]): s for s in samples}
    rows: list[dict[str, Any]] = []
    for partition_name, image_ids in partitions.items():
        counts: dict[tuple[str, ...], int] = defaultdict(int)
        for image_id in image_ids:
            sample = by_image_id[image_id]
            key = tuple(str(sample[k]) for k in stratify_keys)
            counts[key] += 1
        # pack counts into row dicts
        for key, count in sorted(counts.items()):
            row = {"partition": partition_name, "count": count}
            row.update(dict(zip(stratify_keys, key)))
            rows.append(row)
    return rows


# cli entry point to generate and save the dataset splits from config
def main() -> None:
    from training.utils.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite existing split files.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits_dir = Path(cfg["paths"]["splits_dir"])
    # dont overwrite existing splits unless --force is passed
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
    # verify no leakage before saving to disk
    assert_no_group_leakage(partitions, samples, cfg["dataset"]["group_by"])
    write_splits(splits_dir, partitions)

    print(f"Wrote splits to {splits_dir}: " + ", ".join(f"{k}={len(v)}" for k, v in partitions.items()))
    for row in composition_table(partitions, samples, cfg["dataset"]["stratify_by"]):
        print(" ", row)


if __name__ == "__main__":
    main()