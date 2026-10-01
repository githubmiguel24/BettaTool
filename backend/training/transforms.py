# augmentations for training and eval using albumentations

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from app.perception.keypoints import FLIP_MAP, NUM_KEYPOINTS

TransformFn = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]

_ALBUMENTATIONS_IMPORT_ERROR = (
    "albumentations is required for training transforms. "
    "pip install -r requirements-training.txt"
)


class TrainTransform:
    # class wrapper for train augmentations so dataloader workers can pickle it on Windows

    def __init__(self, aug_config: dict[str, Any], input_size: int) -> None:
        # store config and lazy init the pipeline later
        self.aug_config = aug_config
        self.input_size = input_size
        self._pipeline: Any | None = None

    def __getstate__(self) -> dict[str, Any]:
        # drop the pipeline before Pickling so each worker rebuilds it from config
        state = self.__dict__.copy()
        state["_pipeline"] = None
        return state

    def _build_pipeline(self) -> Any:
        # lazy import albumentations and setup the augmentation transforms
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
        # make sure flip map stays identity since all landmarks are on the midline
        assert FLIP_MAP == list(range(NUM_KEYPOINTS)), "flip map must stay the identity — see keypoints.py"

        if self._pipeline is None:
            self._pipeline = self._build_pipeline()

        result = self._pipeline(image=image, keypoints=keypoints_xy.tolist())
        out_image = result["image"]
        out_keypoints = np.array(result["keypoints"], dtype=np.float64)
        if out_keypoints.shape != keypoints_xy.shape:
            # sanity check that no keypoins were dropped during transform
            raise RuntimeError(
                f"Albumentations returned {out_keypoints.shape[0]} keypoints, expected {keypoints_xy.shape[0]}."
            )
        return out_image, out_keypoints, visibility


class EvalTransform:
    # passthrough transform for val and test splits with no augmentation

    def __call__(
        self, image: np.ndarray, keypoints_xy: np.ndarray, visibility: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # return inputs untouched for evaluation
        return image, keypoints_xy, visibility


def build_train_transform(aug_config: dict[str, Any], input_size: int) -> TransformFn:
    # helper to return a picklable train transform instance
    return TrainTransform(aug_config, input_size)


def build_eval_transform() -> TransformFn:
    # helper to return the identity Transform for eval
    return EvalTransform()