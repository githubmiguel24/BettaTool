# preps raw image bytes into model tensors for inference and returns the affine transform to map predictions back to original coords
# keep letterboxing and normalization synced with _load_crop in training/dataset.py or keypoints will quietly degrade

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

from app.perception.geometry import AffineTransform, letterbox_affine

# must match image.imagenet_mean and imagenet_std in training/configs/base.yaml
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# converts uploaded image bytes into an (H, W, 3) RGB array and raises ValueError so the route returns a 400 instead of 500
def decode_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as im:
            return np.array(im.convert("RGB"))
    except Exception as exc:  # noqa: BLE001 PIL throws all kinds of errors on bad files
        raise ValueError(f"Could not decode uploaded file as an image: {exc}") from exc


# letterboxes and normalizes the image for the keypoint model, returning the (1, 3, H, W) tensor and Affine transform
def preprocess_image(
    image: np.ndarray,
    input_size: int = 384,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[torch.Tensor, AffineTransform]:
    orig_h, orig_w = image.shape[:2]
    affine = letterbox_affine(orig_w=float(orig_w), orig_h=float(orig_h), target_size=input_size)

    # read scale straight off the diagonal so we dont re-derive it and risk mismatching the transform
    scale = float(affine.A[0, 0])
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    with Image.fromarray(image) as pil_im:
        resized = np.array(pil_im.resize((new_w, new_h), Image.BILINEAR))

    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    off_x = int(round(float(affine.b[0])))
    off_y = int(round(float(affine.b[1])))
    # clamp offsets just in case edge rounding pushes the paste 1px over
    off_x = max(0, min(off_x, input_size - new_w))
    off_y = max(0, min(off_y, input_size - new_h))
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized

    arr = canvas.astype(np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, affine