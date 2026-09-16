"""Image bytes -> model-ready tensor, plus the affine needed to get back.

This is the inference-time counterpart to `training/dataset.py`'s
`_load_crop`. The two MUST agree on normalization and geometry or the model
sees a different distribution at serve time than it was trained on, which
shows up as quietly-degraded keypoints rather than a loud error.

Both paths letterbox to `input_size` with `letterbox_affine` from
`app/perception/geometry.py` (the single source of truth for coordinate
conversions, per the Build Prompt v2 §4.4 units contract) and normalize with
the same ImageNet statistics.

The returned `AffineTransform` maps ORIGINAL-image pixels -> crop space.
Invert it to map predicted keypoints back onto the user's uploaded image,
which is what the UI overlay needs.
"""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

from app.perception.geometry import AffineTransform, letterbox_affine

# Must match training/configs/base.yaml `image.imagenet_mean` / `imagenet_std`.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def decode_image(data: bytes) -> np.ndarray:
    """Decodes uploaded bytes into an (H, W, 3) uint8 RGB array.

    Raises:
        ValueError: if the bytes are not a decodable image. The caller
            (the /analyze route) turns this into an HTTP 400 rather than
            letting a PIL exception surface as a 500.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            return np.array(im.convert("RGB"))
    except Exception as exc:  # noqa: BLE001 - PIL raises a wide variety here
        raise ValueError(f"Could not decode uploaded file as an image: {exc}") from exc


def preprocess_image(
    image: np.ndarray,
    input_size: int = 384,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[torch.Tensor, AffineTransform]:
    """Letterboxes and normalizes one image for the keypoint model.

    Args:
        image: (H, W, 3) uint8 RGB.
        input_size: square side length the model expects (384 by default,
            matching `training/configs/base.yaml`).

    Returns:
        (tensor, affine) where `tensor` is (1, 3, input_size, input_size)
        float32 and `affine` maps original-image pixels -> crop space.
    """
    orig_h, orig_w = image.shape[:2]
    affine = letterbox_affine(orig_w=float(orig_w), orig_h=float(orig_h), target_size=input_size)

    # `letterbox_affine` is a uniform scale + translation, so we can read the
    # scale straight off the diagonal rather than re-deriving it here and
    # risking a divergence from the transform we hand back to the caller.
    scale = float(affine.A[0, 0])
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    with Image.fromarray(image) as pil_im:
        resized = np.array(pil_im.resize((new_w, new_h), Image.BILINEAR))

    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    off_x = int(round(float(affine.b[0])))
    off_y = int(round(float(affine.b[1])))
    # Clip defensively: rounding at the edges can push the paste 1px over.
    off_x = max(0, min(off_x, input_size - new_w))
    off_y = max(0, min(off_y, input_size - new_h))
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized

    arr = canvas.astype(np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, affine
