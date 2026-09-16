"""Grouped, stratified splitting — no specimen appears in more than one
partition (Build Prompt v2 §11)."""

from __future__ import annotations

import random

from training.splitting import assert_no_group_leakage, compute_grouped_stratified_split


def _make_samples(n_specimens: int, photos_per_specimen: int) -> list[dict]:
    rng = random.Random(0)
    sources = ["roboflow", "kaggle", "breeder_contributed"]
    morphs = ["red", "blue", "yellow"]
    samples = []
    for s in range(n_specimens):
        specimen_id = f"specimen_{s}"
        morph = morphs[s % len(morphs)]
        source = sources[s % len(sources)]
        for p in range(photos_per_specimen):
            samples.append(
                {
                    "image_id": f"img_{s}_{p}",
                    "specimen_id": specimen_id,
                    "source": source,
                    "color_morph": morph,
                    "has_occlusion": rng.random() < 0.2,
                }
            )
    return samples


def test_no_specimen_appears_in_more_than_one_partition() -> None:
    samples = _make_samples(n_specimens=60, photos_per_specimen=4)
    partitions = compute_grouped_stratified_split(
        samples,
        group_key="specimen_id",
        stratify_keys=["source", "color_morph", "has_occlusion"],
        split_fractions={"train": 0.70, "val": 0.15, "test": 0.15},
        seed=42,
    )
    assert_no_group_leakage(partitions, samples, "specimen_id")  # raises on failure


def test_split_sizes_are_approximately_proportional() -> None:
    samples = _make_samples(n_specimens=90, photos_per_specimen=3)
    partitions = compute_grouped_stratified_split(
        samples,
        group_key="specimen_id",
        stratify_keys=["source"],
        split_fractions={"train": 0.70, "val": 0.15, "test": 0.15},
        seed=1,
    )
    total = sum(len(v) for v in partitions.values())
    assert total == len(samples)
    assert 0.60 < len(partitions["train"]) / total < 0.80
    assert 0.05 < len(partitions["val"]) / total < 0.25
    assert 0.05 < len(partitions["test"]) / total < 0.25


def test_splitting_is_deterministic_given_same_seed() -> None:
    samples = _make_samples(n_specimens=40, photos_per_specimen=2)
    kwargs = dict(
        group_key="specimen_id",
        stratify_keys=["source", "color_morph"],
        split_fractions={"train": 0.7, "val": 0.15, "test": 0.15},
        seed=7,
    )
    p1 = compute_grouped_stratified_split(samples, **kwargs)
    p2 = compute_grouped_stratified_split(samples, **kwargs)
    assert p1 == p2


def test_a_fourth_partition_can_be_added_via_fractions_alone() -> None:
    samples = _make_samples(n_specimens=50, photos_per_specimen=3)
    partitions = compute_grouped_stratified_split(
        samples,
        group_key="specimen_id",
        stratify_keys=["source"],
        split_fractions={"train": 0.60, "val": 0.15, "test": 0.15, "calib": 0.10},
        seed=3,
    )
    assert set(partitions.keys()) == {"train", "val", "test", "calib"}
    assert_no_group_leakage(partitions, samples, "specimen_id")
