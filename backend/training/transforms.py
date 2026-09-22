"""Albumentations augmentation pipeline (train split only) — Build Prompt v2 §5.3.

Requires `albumentations` (not a core `app/` dependency — see
`requirements-training.txt`). The import is deferred into the transform
objects' first call rather than module level, so `training/dataset.py` and
anything importing this module for its docstrings or constants stays
importable without albumentations installed; only actually calling a
transform requires it.

Val/test pipelines are always augmentation-free (Build Prompt v2 §5.3) —
`build_eval_transform` returns an identity callable, never partially-applied
augmentation, so there is no way to accidentally leak a "mild" augmentation
into evaluation.

PICKLABILITY (Windows / `num_workers > 0`): these transforms are
module-level CLASSES, not closures returned from a factory. A closure
(`def _transform(...)` inside `build_train_transform`) cannot be pickled,
and PyTorch's DataLoader pickles the dataset — including its `transform_fn`
— to hand it to each worker process under the `spawn` start method, which
is the default on Windows. A closure therefore fails there with
`AttributeError: Can't get local object 'build_train_transform.<locals>._transform'`,
while working fine on Linux (`fork`, no pickling). The albumentations
pipeline itself is built lazily, on first call inside whichever process
ends up running it, and is explicitly dropped from the pickled state
(`__getstate__`), so only the plain config dict ever crosses the process
boundary.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from app.perception.keypoints import FLIP_MAP, NUM_KEYPOINTS

TransformFn = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]

_ALBUMENTATIONS_IMPORT_ERROR = (
    "training/transforms.py requires `albumentations` "
    "(pip install -r requirements-training.txt). This was NOT "
    "installable in the sandbox this codebase was authored in "
    "(see training/README.md, 'Known limitations') — install it in "
    "your actual training environment (Kaggle / local GPU machine)."
)


class TrainTransform:
    """Training-time augmentation, built from `augmentation.train` in base.yaml.

    Callable as `(image_uint8_HWC, keypoints_xy, visibility_flags) ->
    (image_uint8_HWC, keypoints_xy, visibility_flags)`. A keypoint that
    augmentation pushes outside the image is NOT automatically relabeled -1
    here — Albumentations' keypoint params are configured with
    `remove_invisible=False` precisely so this stays a pure geometric
    transform; `training/dataset.py`'s caller is responsible for re-deriving
    `visibility_mask` from the (possibly now out-of-bounds) coordinates if
    that policy is wanted. As shipped, the stored `visibility` flag is left
    untouched by augmentation, which is the conservative choice: an
    augmentation that clips a landmark should not silently invent a new
    "out of frame" label that was never reviewed by a human annotator.
    """

    def __init__(self, aug_config: dict[str, Any], input_size: int) -> None:
        self.aug_config = aug_config
        self.input_size = input_size
        self._pipeline: Any | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Drops the built albumentations pipeline from the pickled state.

        Each worker process rebuilds it from `aug_config` on first call, so
        nothing but plain config data crosses the process boundary — this is
        what keeps `num_workers > 0` working under Windows' `spawn`.
        """
        state = self.__dict__.copy()
        state["_pipeline"] = None
        return state

    def _build_pipeline(self) -> Any:
        try:
            import albumentations as A
        except ImportError as exc:
            raise ImportError(_ALBUMENTATIONS_IMPORT_ERROR) from exc

        aug_config = self.aug_config
        color = aug_config["color_jitter"]
        return A.Compose(
            [
                A.HorizontalFlip(p=aug_config["hflip_p"]),
                A.Affine(
                    rotate=(-aug_config["rotate_deg"], aug_config["rotate_deg"]),
                    scale=tuple(aug_config["scale_range"]),
                    translate_percent=(-aug_config["translate_frac"], aug_config["translate_frac"]),
                    p=1.0,
                ),
                A.ColorJitter(
                    brightness=color["brightness"],
                    contrast=color["contrast"],
                    saturation=color["saturation"],
                    hue=color["hue"],
                    p=0.8,
                ),
                A.GaussianBlur(p=aug_config["blur_p"]),
                A.ImageCompression(quality_range=(60, 100), p=aug_config["jpeg_artifact_p"]),
                A.CoarseDropout(
                    num_holes_range=(1, aug_config["coarse_dropout_max_holes"]),
                    hole_height_range=(0.02, aug_config["coarse_dropout_hole_size_frac"]),
                    hole_width_range=(0.02, aug_config["coarse_dropout_hole_size_frac"]),
                    p=aug_config["coarse_dropout_p"],
                ),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def __call__(
        self, image: np.ndarray, keypoints_xy: np.ndarray, visibility: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Build Prompt v2 §3.1: the flip permutation is the IDENTITY — every
        # landmark is midline/dorsal-ventral, never a bilateral pair. Assert
        # this on every call so a future edit to FLIP_MAP cannot silently
        # introduce a swap map without this pipeline noticing.
        assert FLIP_MAP == list(range(NUM_KEYPOINTS)), "flip map must stay the identity — see keypoints.py"

        if self._pipeline is None:
            self._pipeline = self._build_pipeline()

        result = self._pipeline(image=image, keypoints=keypoints_xy.tolist())
        out_image = result["image"]
        out_keypoints = np.array(result["keypoints"], dtype=np.float64)
        if out_keypoints.shape != keypoints_xy.shape:
            # Should not happen with remove_invisible=False, but fail loudly
            # (Build Prompt v2 §1) rather than silently mis-aligning arrays.
            raise RuntimeError(
                f"Albumentations returned {out_keypoints.shape[0]} keypoints, expected {keypoints_xy.shape[0]}."
            )
        return out_image, out_keypoints, visibility


class EvalTransform:
    """Identity transform for val/test — augmentation-free by construction (Build Prompt v2 §5.3)."""

    def __call__(
        self, image: np.ndarray, keypoints_xy: np.ndarray, visibility: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return image, keypoints_xy, visibility


def build_train_transform(aug_config: dict[str, Any], input_size: int) -> TransformFn:
    """Builds the training-time augmentation callable from
    `training/configs/base.yaml`'s `augmentation.train` block.

    Returns a picklable `TrainTransform` instance (see this module's
    docstring on why it must not be a closure).

    Raises:
        ImportError: if `albumentations` is not installed — raised on first
            CALL, not here, so the object stays constructible (and
            picklable) in a process where albumentations is absent. Install
            it (see requirements-training.txt) before training — do not swap
            in a no-augmentation fallback silently, since Build Prompt v2
            §5.4 calls augmentation load-bearing at ~1,750 images.
    """
    return TrainTransform(aug_config, input_size)


def build_eval_transform() -> TransformFn:
    """Identity transform for val/test — augmentation-free by construction (Build Prompt v2 §5.3)."""
    return EvalTransform()