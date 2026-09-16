"""Single source of truth check (Build Prompt v2 §4.2's spirit applied to
keypoint definitions): app/perception/keypoints.py and
training/configs/keypoints.yaml must never drift apart.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.perception.keypoints import FLIP_MAP, KEYPOINT_SHORT_CODES, NUM_KEYPOINTS, SKELETON_EDGES

_YAML_PATH = Path(__file__).resolve().parents[1] / "training" / "configs" / "keypoints.yaml"


def _load_yaml() -> dict:
    return yaml.safe_load(_YAML_PATH.read_text())


def test_num_keypoints_matches_yaml() -> None:
    assert _load_yaml()["num_keypoints"] == NUM_KEYPOINTS == 13


def test_names_and_order_match_yaml() -> None:
    yaml_names = [lm["name"] for lm in sorted(_load_yaml()["landmarks"], key=lambda lm: lm["index"])]
    assert yaml_names == KEYPOINT_SHORT_CODES


def test_flip_map_is_identity_in_both_places() -> None:
    assert FLIP_MAP == list(range(NUM_KEYPOINTS))
    assert _load_yaml()["flip_map"] == list(range(NUM_KEYPOINTS))


def test_flip_applied_twice_is_a_no_op() -> None:
    permuted_once = [FLIP_MAP[i] for i in range(NUM_KEYPOINTS)]
    permuted_twice = [FLIP_MAP[permuted_once[i]] for i in range(NUM_KEYPOINTS)]
    assert permuted_twice == list(range(NUM_KEYPOINTS))


def test_skeleton_edges_reference_valid_indices() -> None:
    for i, j in SKELETON_EDGES:
        assert 0 <= i < NUM_KEYPOINTS
        assert 0 <= j < NUM_KEYPOINTS

    yaml_edges = [tuple(e) for e in _load_yaml()["skeleton"]]
    assert yaml_edges == SKELETON_EDGES
