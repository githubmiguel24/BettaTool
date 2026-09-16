"""Synthesizes a small procedurally-drawn dataset so the entire pipeline
(split -> validate -> dataset -> train -> evaluate -> predict) can run
end-to-end before any real Roboflow annotations exist (Build Prompt v2 §5.5).

Each image is a simple fish-like silhouette (an ellipse body + triangular
fins) drawn with known landmark positions, so ground truth is exact and the
"model" can plausibly learn something on it even though it looks nothing
like a real Betta photograph — the point is pipeline verification, not
representativeness.

Usage:
    python -m training.make_dummy_dataset --out data/raw_dummy --n 64
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app.perception.keypoints import KEYPOINT_SHORT_CODES, NUM_KEYPOINTS, Visibility

_SOURCES = ["roboflow", "kaggle", "breeder_contributed"]
_COLOR_MORPHS = ["red", "blue", "yellow", "marble", "black"]
_COLOR_RGB = {
    "red": (180, 40, 40),
    "blue": (40, 70, 180),
    "yellow": (210, 190, 50),
    "marble": (160, 150, 160),
    "black": (40, 40, 45),
}


def _procedural_landmarks(body_cx: float, body_cy: float, body_w: float, body_h: float, rng: random.Random) -> np.ndarray:
    """Returns a (13, 2) array of landmark (x, y) positions for one synthetic
    fish, in the same order as `app.perception.keypoints.Keypoint`.

    Positions are simple geometric offsets from the body ellipse, jittered
    slightly per-sample so the "dataset" is not 64 identical fish.
    """
    jitter = lambda scale: rng.uniform(-scale, scale)  # noqa: E731

    half_w, half_h = body_w / 2, body_h / 2
    pts = np.array(
        [
            (body_cx - half_w - 0.05 * body_w, body_cy + jitter(0.03 * body_h)),                # snout_tip
            (body_cx - half_w * 0.55, body_cy - half_h * 0.15 + jitter(0.02 * body_h)),          # eye_center
            (body_cx - half_w * 0.1, body_cy - half_h * 1.05 + jitter(0.02 * body_h)),           # dorsal_base_ant
            (body_cx + half_w * 0.35, body_cy - half_h * 1.05 + jitter(0.02 * body_h)),          # dorsal_base_post
            (body_cx + half_w * 0.30, body_cy - half_h * 1.9 + jitter(0.05 * body_h)),           # dorsal_tip
            (body_cx + half_w * 0.85, body_cy - half_h * 0.5 + jitter(0.02 * body_h)),           # peduncle_top
            (body_cx + half_w * 0.85, body_cy + half_h * 0.5 + jitter(0.02 * body_h)),           # peduncle_bottom
            (body_cx + half_w * 2.0, body_cy - half_h * 1.6 + jitter(0.08 * body_h)),            # caudal_tip_upper
            (body_cx + half_w * 2.0, body_cy + half_h * 1.6 + jitter(0.08 * body_h)),            # caudal_tip_lower
            (body_cx + half_w * 2.2, body_cy + jitter(0.05 * body_h)),                           # caudal_center
            (body_cx + half_w * 0.35, body_cy + half_h * 1.05 + jitter(0.02 * body_h)),          # anal_base_ant
            (body_cx - half_w * 0.1, body_cy + half_h * 1.05 + jitter(0.02 * body_h)),           # anal_base_post
            (body_cx - half_w * 0.05, body_cy + half_h * 1.9 + jitter(0.05 * body_h)),           # anal_tip
        ],
        dtype=np.float64,
    )
    assert pts.shape == (NUM_KEYPOINTS, 2)
    return pts


def _draw_fish(img_size: int, landmarks: np.ndarray, body_cx: float, body_cy: float, body_w: float, body_h: float, color: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (img_size, img_size), color=(230, 235, 240))
    draw = ImageDraw.Draw(img)

    body_bbox = [body_cx - body_w / 2, body_cy - body_h / 2, body_cx + body_w / 2, body_cy + body_h / 2]
    draw.ellipse(body_bbox, fill=color)

    dorsal = [tuple(landmarks[2]), tuple(landmarks[4]), tuple(landmarks[3])]
    anal = [tuple(landmarks[10]), tuple(landmarks[12]), tuple(landmarks[11])]
    caudal = [tuple(landmarks[5]), tuple(landmarks[7]), tuple(landmarks[9]), tuple(landmarks[8]), tuple(landmarks[6])]
    for fin in (dorsal, anal, caudal):
        draw.polygon(fin, fill=tuple(int(c * 0.75) for c in color))

    eye_xy = landmarks[1]
    r = body_h * 0.05
    draw.ellipse([eye_xy[0] - r, eye_xy[1] - r, eye_xy[0] + r, eye_xy[1] + r], fill=(10, 10, 10))
    return img


def _fish_box_from_landmarks(landmarks: np.ndarray, img_size: int, pad: float = 8.0) -> list[float]:
    x_min, y_min = landmarks.min(axis=0) - pad
    x_max, y_max = landmarks.max(axis=0) + pad
    x_min, y_min = max(0.0, x_min), max(0.0, y_min)
    x_max, y_max = min(float(img_size), x_max), min(float(img_size), y_max)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def generate_dummy_dataset(
    output_dir: str | Path,
    n_images: int,
    img_size: int = 640,
    n_specimens: int | None = None,
    seed: int = 42,
) -> Path:
    """Generates `n_images` synthetic fish photographs + a COCO Keypoints JSON.

    Args:
        output_dir: directory to write `images/` and `annotations.json` into
            (created if missing).
        n_images: number of images to generate.
        img_size: side length of each square generated image, pixels.
        n_specimens: number of distinct `specimen_id`s to draw images from
            (with repeats), exercising grouped splitting. Defaults to
            `max(1, n_images // 3)`, i.e. on average 3 photos per specimen.
        seed: RNG seed, for reproducibility.

    Returns:
        Path to the written `annotations.json`.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    n_specimens = n_specimens if n_specimens is not None else max(1, n_images // 3)
    specimen_ids = [f"specimen_{i:04d}" for i in range(n_specimens)]
    specimen_morph = {sid: rng.choice(_COLOR_MORPHS) for sid in specimen_ids}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    for image_idx in range(n_images):
        image_id = f"dummy_{image_idx:05d}"
        specimen_id = rng.choice(specimen_ids)
        color_morph = specimen_morph[specimen_id]
        source = rng.choice(_SOURCES)

        body_w = img_size * rng.uniform(0.32, 0.42)
        body_h = body_w * rng.uniform(0.38, 0.48)
        body_cx = img_size * rng.uniform(0.38, 0.48)
        body_cy = img_size * rng.uniform(0.45, 0.55)

        landmarks = _procedural_landmarks(body_cx, body_cy, body_w, body_h, rng)

        # Visibility: mostly clear (2), a slice ambiguous/occluded (1/0), and
        # a small slice out-of-frame (-1) — enough to exercise every branch
        # of the masking logic without dominating the "dataset".
        visibility = np_rng.choice(
            [Visibility.CLEAR, Visibility.AMBIGUOUS, Visibility.OCCLUDED, Visibility.OUT_OF_FRAME],
            size=NUM_KEYPOINTS,
            p=[0.80, 0.10, 0.06, 0.04],
        )
        # A landmark flagged out-of-frame is pushed outside the image bounds,
        # so the synthetic data is internally consistent with its own label
        # rather than lying about where the point "really" is.
        for kp_idx, vis in enumerate(visibility):
            if vis == Visibility.OUT_OF_FRAME:
                landmarks[kp_idx] = [img_size + 50.0, img_size + 50.0]

        img = _draw_fish(img_size, landmarks, body_cx, body_cy, body_w, body_h, _COLOR_RGB[color_morph])
        file_name = f"{image_id}.jpg"
        img.save(images_dir / file_name, quality=90)

        fish_box = _fish_box_from_landmarks(landmarks[visibility != Visibility.OUT_OF_FRAME], img_size)

        keypoints_flat: list[float] = []
        for (x, y), v in zip(landmarks, visibility):
            keypoints_flat.extend([float(x), float(y), int(v)])

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": img_size,
                "height": img_size,
            }
        )
        annotations.append(
            {
                "image_id": image_id,
                "specimen_id": specimen_id,
                "source": source,
                "color_morph": color_morph,
                "keypoints": keypoints_flat,
                "num_keypoints": NUM_KEYPOINTS,
                "fish_box": fish_box,
                "has_occlusion": bool(np.any((visibility == Visibility.OCCLUDED) | (visibility == Visibility.OUT_OF_FRAME))),
            }
        )

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "betta", "keypoints": KEYPOINT_SHORT_CODES}],
        "info": {"description": "Synthetic dummy dataset — see training/make_dummy_dataset.py", "n_images": n_images},
    }

    ann_path = output_dir / "annotations.json"
    ann_path.write_text(json.dumps(coco, indent=2))
    return ann_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/raw_dummy", help="Output directory.")
    parser.add_argument("--n", type=int, default=64, help="Number of images to generate.")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ann_path = generate_dummy_dataset(args.out, args.n, img_size=args.img_size, seed=args.seed)
    print(f"Wrote {args.n} synthetic images + annotations to {ann_path}")


if __name__ == "__main__":
    main()
