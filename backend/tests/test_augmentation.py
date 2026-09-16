"""Augmentation tests (Build Prompt v2 §11). SKIPPED if albumentations is
not importable — see training/README.md, "Known limitations": it could not
be installed in the sandbox this codebase was authored in.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("albumentations")

from app.perception.keypoints import NUM_KEYPOINTS  # noqa: E402
from training.transforms import build_eval_transform, build_train_transform  # noqa: E402

_AUG_CONFIG = {
    "hflip_p": 1.0,  # deterministic for the flip-consistency test below
    "rotate_deg": 0,
    "scale_range": [1.0, 1.0],
    "translate_frac": 0.0,
    "color_jitter": {"brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0},
    "blur_p": 0.0,
    "jpeg_artifact_p": 0.0,
    "coarse_dropout_p": 0.0,
    "coarse_dropout_max_holes": 1,
    "coarse_dropout_hole_size_frac": 0.05,
}


def test_eval_transform_is_identity() -> None:
    transform = build_eval_transform()
    image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    keypoints = np.random.uniform(0, 64, size=(NUM_KEYPOINTS, 2))
    visibility = np.full(NUM_KEYPOINTS, 2)

    out_image, out_kp, out_vis = transform(image, keypoints, visibility)
    assert np.array_equal(out_image, image)
    assert np.allclose(out_kp, keypoints)
    assert np.array_equal(out_vis, visibility)


def test_horizontal_flip_moves_keypoints_consistently_with_the_image() -> None:
    """With hflip_p=1.0 and every other augmentation disabled, a keypoint at
    x should land at (width - 1 - x), matching where the pixel itself moved."""
    transform = build_train_transform(_AUG_CONFIG, input_size=64)
    width = 64
    image = np.zeros((width, width, 3), dtype=np.uint8)
    image[:, 10, :] = 255  # a vertical white stripe at x=10

    keypoints = np.array([[10.0, 32.0]])
    visibility = np.array([2])

    out_image, out_kp, _ = transform(image, keypoints, visibility)

    # The stripe should have moved to x = width - 1 - 10 = 53.
    stripe_x = np.argmax(out_image[32, :, 0])
    assert abs(int(stripe_x) - (width - 1 - 10)) <= 1

    expected_kp_x = width - 1 - 10
    assert abs(out_kp[0, 0] - expected_kp_x) <= 1.5
