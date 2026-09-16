"""Dataset loading, validation, and coordinate round-trip tests (Build Prompt v2 §11).

These exercise the pure NumPy/PIL parts of `training/dataset.py` (crop,
letterbox, affine bookkeeping) without requiring torch — only
`BettaKeypointDataset.__getitem__`'s final tensor conversion needs torch,
and that step is exercised separately in test_smoke_dummy.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.dataset import AnnotationValidationError, BettaKeypointDataset, load_and_validate_annotations
from training.make_dummy_dataset import generate_dummy_dataset


@pytest.fixture()
def dummy_dataset_dir(tmp_path: Path) -> Path:
    generate_dummy_dataset(tmp_path, n_images=12, img_size=256, seed=1)
    return tmp_path


def test_load_and_validate_annotations_succeeds_on_dummy_data(dummy_dataset_dir: Path) -> None:
    anns = load_and_validate_annotations(dummy_dataset_dir / "annotations.json", dummy_dataset_dir / "images")
    assert len(anns) == 12
    for sample in anns:
        assert "specimen_id" in sample
        assert "fish_box" in sample
        assert len(sample["keypoints"]) == 13 * 3


def test_validation_fails_loudly_on_wrong_keypoint_count(dummy_dataset_dir: Path, tmp_path: Path) -> None:
    import json

    ann_path = dummy_dataset_dir / "annotations.json"
    coco = json.loads(ann_path.read_text())
    coco["annotations"][0]["keypoints"] = coco["annotations"][0]["keypoints"][:-3]  # drop one landmark
    ann_path.write_text(json.dumps(coco))

    with pytest.raises(AnnotationValidationError):
        load_and_validate_annotations(ann_path, dummy_dataset_dir / "images")


def test_validation_fails_loudly_on_missing_image_file(dummy_dataset_dir: Path) -> None:
    import json

    ann_path = dummy_dataset_dir / "annotations.json"
    coco = json.loads(ann_path.read_text())
    (dummy_dataset_dir / "images" / coco["images"][0]["file_name"]).unlink()

    with pytest.raises(AnnotationValidationError):
        load_and_validate_annotations(ann_path, dummy_dataset_dir / "images")


def test_crop_round_trip_within_1px_for_in_frame_landmarks(dummy_dataset_dir: Path) -> None:
    anns = load_and_validate_annotations(dummy_dataset_dir / "annotations.json", dummy_dataset_dir / "images")
    ds = BettaKeypointDataset(
        images_dir=dummy_dataset_dir / "images",
        annotations_path=dummy_dataset_dir / "annotations.json",
        split_image_ids=[a["image_id"] for a in anns],
        input_size=384,
        heatmap_size=96,
        bbox_padding=0.15,
        heatmap_target_sigma_px=2.0,
        imagenet_mean=[0.485, 0.456, 0.406],
        imagenet_std=[0.229, 0.224, 0.225],
    )

    for sample in ds.samples:
        crop_image, kp_crop, visibility, affine = ds._load_crop(sample)
        assert crop_image.shape == (384, 384, 3)

        orig_kp = np.array(sample["keypoints"], dtype=np.float64).reshape(-1, 3)[:, :2]
        recovered = affine.inverse().apply_points(kp_crop)

        in_frame = visibility != -1
        err = np.abs(recovered[in_frame] - orig_kp[in_frame])
        assert err.max() < 1.0


def test_out_of_frame_landmarks_land_outside_the_crop_canvas(dummy_dataset_dir: Path) -> None:
    anns = load_and_validate_annotations(dummy_dataset_dir / "annotations.json", dummy_dataset_dir / "images")
    ds = BettaKeypointDataset(
        images_dir=dummy_dataset_dir / "images",
        annotations_path=dummy_dataset_dir / "annotations.json",
        split_image_ids=[a["image_id"] for a in anns],
        input_size=384,
        heatmap_size=96,
        bbox_padding=0.15,
        heatmap_target_sigma_px=2.0,
        imagenet_mean=[0.485, 0.456, 0.406],
        imagenet_std=[0.229, 0.224, 0.225],
    )
    found_any_oof = False
    for sample in ds.samples:
        _, kp_crop, visibility, _ = ds._load_crop(sample)
        oof = visibility == -1
        if oof.any():
            found_any_oof = True
            oof_points = kp_crop[oof]
            assert np.any((oof_points < 0) | (oof_points > 384))
    assert found_any_oof, "dummy dataset seed should produce at least one out-of-frame landmark"
