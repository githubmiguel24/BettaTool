"""COCO-Keypoints dataset loader for the Betta perceptual tier.

Build Prompt v2 §5.1-§5.3: loads a Roboflow/CVAT COCO Keypoints export
(`training/make_dummy_dataset.py` emits the same schema for the synthetic
dataset), validates every sample strictly at load time (§1 point 6 — fail
loudly, never silently skip malformed samples), crops to `fish_box` with
configurable padding, letterboxes to `image.input_size`, and returns
crop-space targets ready for the loss functions in `training/losses/`.

Real-data annotation loading is now implemented (superseding the
`NotImplementedError` stub this file used to raise) — but it has only been
exercised against the synthetic dummy dataset in this session, never
against an actual Roboflow export, because none exists yet. See
training/README.md, "Known limitations", before trusting this against real
data without re-checking it yourself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - exercised only where torch is absent
    # Falling back to `object` keeps this module importable (and its pure
    # numpy/PIL logic testable) even without torch installed. Instantiating
    # BettaKeypointDataset without torch will still fail, inside
    # __getitem__, with a clear ImportError rather than at import time.
    Dataset = object  # type: ignore[assignment, misc]

from app.perception.geometry import AffineTransform, bbox_crop_affine, crop_space_to_heatmap_space, letterbox_affine
from app.perception.keypoints import MASKED_FLAGS, NUM_KEYPOINTS, VISIBLE_FLAGS, Visibility
from training.targets import render_all_targets


class AnnotationValidationError(ValueError):
    """Raised when an annotation file contains malformed samples.

    `training/validate_annotations.py` catches this per-sample to build its
    report; `BettaKeypointDataset.__init__` lets it propagate (fail loudly —
    Build Prompt v2 §1 point 6), so a bad annotation file never silently
    yields a shorter-than-expected dataset.
    """


def _validate_sample(sample: dict[str, Any], images_by_id: dict[str, dict[str, Any]], images_dir: Path) -> list[str]:
    """Returns a list of human-readable problem descriptions for one COCO
    annotation record (empty list = valid). Never raises itself — the
    caller decides whether to collect-and-report (validate_annotations.py)
    or fail immediately (BettaKeypointDataset).
    """
    problems: list[str] = []
    image_id = str(sample.get("image_id", "<missing image_id>"))

    image_meta = images_by_id.get(image_id)
    if image_meta is None:
        problems.append(f"image_id {image_id!r} has no matching entry in images[]")
        return problems  # nothing further can be checked without image size

    image_path = images_dir / image_meta["file_name"]
    if not image_path.is_file():
        problems.append(f"image file missing on disk: {image_path}")

    keypoints = sample.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) != NUM_KEYPOINTS * 3:
        problems.append(
            f"keypoints must be a flat list of length {NUM_KEYPOINTS * 3} (x,y,v per landmark), "
            f"got length {len(keypoints) if isinstance(keypoints, list) else type(keypoints)}"
        )
        return problems

    width, height = image_meta.get("width"), image_meta.get("height")
    for i in range(NUM_KEYPOINTS):
        x, y, v = keypoints[3 * i], keypoints[3 * i + 1], keypoints[3 * i + 2]
        if v not in (Visibility.CLEAR, Visibility.AMBIGUOUS, Visibility.OCCLUDED, Visibility.OUT_OF_FRAME):
            problems.append(f"landmark {i}: visibility flag {v!r} is not one of 2/1/0/-1")
        if v != Visibility.OUT_OF_FRAME and width and height and not (0 <= x <= width and 0 <= y <= height):
            problems.append(f"landmark {i}: coordinate ({x}, {y}) outside image bounds ({width}x{height})")

    if "fish_box" not in sample:
        problems.append("missing fish_box")
    elif not (isinstance(sample["fish_box"], list) and len(sample["fish_box"]) == 4):
        problems.append(f"fish_box must be [x, y, w, h], got {sample['fish_box']!r}")

    if "specimen_id" not in sample:
        problems.append("missing specimen_id (required for grouped splitting — see training/splitting.py)")

    if "id" not in sample:
        problems.append(
            "missing annotation id (required to disambiguate multiple fish sharing one image_id — "
            "see BettaKeypointDataset._load_crop's cache key)"
        )

    return problems


def load_and_validate_annotations(annotations_path: str | Path, images_dir: str | Path) -> list[dict[str, Any]]:
    """Loads a COCO Keypoints JSON file, validating every record.

    Raises:
        AnnotationValidationError: on the FIRST malformed sample, naming
            every problem found with that sample (and, for a fatal
            structural fault, only that fault). Fails loudly rather than
            skipping bad samples (Build Prompt v2 §1 point 6) — use
            `training/validate_annotations.py` first to see a full report
            across ALL samples before fixing and re-running this loader.
        FileNotFoundError: if the file does not exist.
    """
    annotations_path = Path(annotations_path)
    images_dir = Path(images_dir)
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotations_path}")

    coco = json.loads(annotations_path.read_text())
    if "images" not in coco or "annotations" not in coco:
        raise AnnotationValidationError(f"{annotations_path} is not a valid COCO file: missing 'images' or 'annotations'.")

    images_by_id = {str(img["id"]): img for img in coco["images"]}

    # NOTE: multiple annotations CAN legitimately share one image_id -- an
    # image with more than one fish in it (a catalog/comparison photo) has
    # one annotation per fish, all pointing at the same image_id. What must
    # never repeat is the ANNOTATION's own id (each fish's own record).
    seen_annotation_ids: set[str] = set()
    for sample in coco["annotations"]:
        image_id = str(sample.get("image_id", ""))
        annotation_id = str(sample.get("id", f"<missing id, image_id={image_id}>"))
        if annotation_id in seen_annotation_ids:
            raise AnnotationValidationError(f"duplicate annotation id: {annotation_id!r}")
        seen_annotation_ids.add(annotation_id)

        problems = _validate_sample(sample, images_by_id, images_dir)
        if problems:
            raise AnnotationValidationError(
                f"annotation for image_id={image_id!r} failed validation:\n  - " + "\n  - ".join(problems)
            )

    return coco["annotations"]


def build_validation_report(annotations_path: str | Path, images_dir: str | Path) -> dict[str, Any]:
    """Collects validation problems across EVERY sample (never raises), for
    `training/validate_annotations.py`'s CLI report (Build Prompt v2 §5.4).

    Unlike `load_and_validate_annotations` (which fails loudly on the first
    bad sample, by design, for the training path), this function is meant
    to show a human everything that is wrong in one pass.

    Returns a dict with:
        n_images, n_annotations       : int
        problems_by_image             : {image_id: [problem, ...]}, only for
                                         images with at least one problem
        duplicate_image_ids           : list[str]
        visibility_histogram          : {flag: count}, across ALL keypoints
        per_keypoint_visibility       : {keypoint_index: {flag: count}}
        missing_fish_box_count        : int
    """
    annotations_path = Path(annotations_path)
    images_dir = Path(images_dir)
    coco = json.loads(annotations_path.read_text())
    images_by_id = {str(img["id"]): img for img in coco.get("images", [])}

    problems_by_image: dict[str, list[str]] = {}
    duplicate_image_ids: list[str] = []
    seen_ids: set[str] = set()
    visibility_histogram: dict[int, int] = {2: 0, 1: 0, 0: 0, -1: 0}
    per_keypoint_visibility: dict[int, dict[int, int]] = {i: {2: 0, 1: 0, 0: 0, -1: 0} for i in range(NUM_KEYPOINTS)}
    missing_fish_box_count = 0

    for sample in coco.get("annotations", []):
        image_id = str(sample.get("image_id", "<missing>"))
        if image_id in seen_ids:
            duplicate_image_ids.append(image_id)
        seen_ids.add(image_id)

        problems = _validate_sample(sample, images_by_id, images_dir)
        if problems:
            problems_by_image[image_id] = problems

        if "fish_box" not in sample:
            missing_fish_box_count += 1

        keypoints = sample.get("keypoints")
        if isinstance(keypoints, list) and len(keypoints) == NUM_KEYPOINTS * 3:
            for i in range(NUM_KEYPOINTS):
                v = keypoints[3 * i + 2]
                if v in visibility_histogram:
                    visibility_histogram[v] += 1
                    per_keypoint_visibility[i][v] += 1

    return {
        "n_images": len(images_by_id),
        "n_annotations": len(coco.get("annotations", [])),
        "problems_by_image": problems_by_image,
        "duplicate_image_ids": duplicate_image_ids,
        "visibility_histogram": visibility_histogram,
        "per_keypoint_visibility": per_keypoint_visibility,
        "missing_fish_box_count": missing_fish_box_count,
    }


class BettaKeypointDataset(Dataset):
    """COCO-Keypoints dataset producing crop-space training samples.

    Each `__getitem__` returns a dict:
        image            : (3, input_size, input_size) float32 tensor,
                            ImageNet-normalized (torch tensor; converted from
                            an (H, W, 3) uint8 array by `training/transforms.py`).
        keypoints_crop   : (K, 2) float32, ground-truth (x, y) in crop space.
        visibility       : (K,) int64, raw flag (2/1/0/-1).
        visibility_mask  : (K,) float32, 1.0 for flags 2/1, 0.0 for 0/-1.
        heatmap_target   : (K, heatmap_size, heatmap_size) float32.
        affine_crop_to_orig : the (A, b) mapping crop-space back to
                            original-image space, needed to report
                            predictions in original pixels
                            (`app/perception/geometry.py`).
        image_id         : str.

    Augmentation (train split only) is applied by `transform_fn`, an
    Albumentations-backed callable from `training/transforms.py` — this
    class does not import albumentations directly, so it stays importable
    (and independently testable) even where albumentations is not
    installed, as long as a caller does not actually request augmentation.
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations_path: str | Path,
        split_image_ids: list[str],
        input_size: int,
        heatmap_size: int,
        bbox_padding: float,
        heatmap_target_sigma_px: float,
        imagenet_mean: list[float],
        imagenet_std: list[float],
        transform_fn: Any = None,
        cache_crops_in_ram: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.stride = input_size / heatmap_size
        self.bbox_padding = bbox_padding
        self.heatmap_target_sigma_px = heatmap_target_sigma_px
        self.imagenet_mean = np.array(imagenet_mean, dtype=np.float32)
        self.imagenet_std = np.array(imagenet_std, dtype=np.float32)
        self.transform_fn = transform_fn
        self.cache_crops_in_ram = cache_crops_in_ram
        self._crop_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, AffineTransform]] = {}

        all_annotations = load_and_validate_annotations(annotations_path, self.images_dir)
        coco = json.loads(Path(annotations_path).read_text())
        self.images_by_id: dict[str, Any] = {str(img["id"]): img for img in coco["images"]}

        wanted = set(split_image_ids)
        self.samples = [a for a in all_annotations if str(a["image_id"]) in wanted]

        missing = wanted - {str(a["image_id"]) for a in self.samples}
        if missing:
            raise AnnotationValidationError(
                f"{len(missing)} image_ids listed in the split are not present in the "
                f"annotation file (first few: {sorted(missing)[:5]})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_crop(self, sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, AffineTransform]:
        """Returns (crop_image_uint8_HWC, keypoints_crop, visibility_flags, affine_orig_to_crop)."""
        # Cache key is the ANNOTATION's own id, not image_id: an image with
        # multiple fish (multiple annotations sharing one image_id) has a
        # different fish_box -- and therefore a different crop -- per
        # annotation. Keying the cache by image_id would return fish #1's
        # cached crop for fish #2 and #3 of the same photo, silently.
        cache_key = str(sample["id"])
        if self.cache_crops_in_ram and cache_key in self._crop_cache:
            return self._crop_cache[cache_key]

        keypoints = np.array(sample["keypoints"], dtype=np.float64).reshape(NUM_KEYPOINTS, 3)
        xy, visibility = keypoints[:, :2], keypoints[:, 2].astype(np.int64)

        bx, by, bw, bh = sample["fish_box"]
        crop_affine = bbox_crop_affine(bx, by, bw, bh, self.bbox_padding)

        image_path = self.images_dir / self._image_meta(sample)["file_name"]
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            cropped_w, cropped_h = bw * (1 + 2 * self.bbox_padding), bh * (1 + 2 * self.bbox_padding)
            crop_box = (
                max(0, int(bx - bw * self.bbox_padding)),
                max(0, int(by - bh * self.bbox_padding)),
                min(im.width, int(bx + bw * (1 + self.bbox_padding))),
                min(im.height, int(by + bh * (1 + self.bbox_padding))),
            )
            cropped_im = im.crop(crop_box)

        letterbox_t = letterbox_affine(cropped_im.width, cropped_im.height, self.input_size)
        # Recompute crop_affine's b to match the ACTUAL (clamped-to-image) crop box used,
        # rather than the unclamped padded box, so the affine and the pixels agree exactly.
        crop_affine = AffineTransform(A=np.eye(2), b=-np.array([crop_box[0], crop_box[1]], dtype=np.float64))
        full_affine = crop_affine.compose(letterbox_t)

        resized_w = max(1, round(cropped_im.width * letterbox_t.A[0, 0]))
        resized_h = max(1, round(cropped_im.height * letterbox_t.A[1, 1]))
        resized = cropped_im.resize((resized_w, resized_h), Image.BILINEAR)

        canvas = Image.new("RGB", (self.input_size, self.input_size), (0, 0, 0))
        paste_xy = (round(letterbox_t.b[0]), round(letterbox_t.b[1]))
        canvas.paste(resized, paste_xy)

        keypoints_crop = full_affine.apply_points(xy)
        result = (np.array(canvas), keypoints_crop, visibility, full_affine)

        if self.cache_crops_in_ram:
            self._crop_cache[cache_key] = result
        return result

    def _image_meta(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self.images_by_id[str(sample["image_id"])]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        crop_image, keypoints_crop, visibility, full_affine = self._load_crop(sample)

        if self.transform_fn is not None:
            crop_image, keypoints_crop, visibility = self.transform_fn(crop_image, keypoints_crop, visibility)

        image_norm = (crop_image.astype(np.float32) / 255.0 - self.imagenet_mean) / self.imagenet_std
        image_chw = np.transpose(image_norm, (2, 0, 1))

        visible = np.isin(visibility, [int(v) for v in VISIBLE_FLAGS])
        visibility_mask = visible.astype(np.float32)

        keypoints_heatmap_space = keypoints_crop / self.stride
        heatmap_target = render_all_targets(
            self.heatmap_size, keypoints_heatmap_space, self.heatmap_target_sigma_px, visible
        )

        import torch  # local import: keeps this module importable without torch for pure-logic tests

        return {
            "image": torch.from_numpy(image_chw).float(),
            "keypoints_crop": torch.from_numpy(keypoints_crop).float(),
            "visibility": torch.from_numpy(visibility).long(),
            "visibility_mask": torch.from_numpy(visibility_mask).float(),
            "heatmap_target": torch.from_numpy(heatmap_target).float(),
            "affine_A": torch.from_numpy(full_affine.A).float(),
            "affine_b": torch.from_numpy(full_affine.b).float(),
            "image_id": str(sample["image_id"]),
        }