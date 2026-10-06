# MFLD-Net (Saleh et al. 2023) inference, used ONLY for the "compare with MFLD-Net" tab - never for the assessment itself
# the architecture is a compact copy of mfld-net/mfld/model.py so the backend does not depend on that sibling repository
# preprocessing follows mfld-net's own inference path: the WHOLE image squashed to 224x224 (cv2 INTER_LINEAR), ImageNet
# normalisation, soft-argmax over the 56x56 softmax heatmaps in (i + 0.5) / 56 coordinates, then scaled back by the image size

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import settings

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

_LOCK = threading.Lock()
_BUNDLE: "MfldBundle | None" = None
_MISSING = object()


class _ConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size // 2), nn.GELU(), nn.BatchNorm2d(dim))
        self.dropout = nn.Dropout2d(dropout)
        self.pointwise = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.GELU(), nn.BatchNorm2d(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.dropout(x + self.depthwise(x)))


class MfldNet(nn.Module):
    def __init__(self, img_size: int, patch_size: int, dim: int, depth: int, kernel_size: int, dropout: float, num_keypoints: int, in_chans: int) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_embed = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.Sequential(*[_ConvBlock(dim, kernel_size, dropout) for _ in range(depth)])
        self.heatmap_conv = nn.Conv2d(dim, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns (B, K, 2) soft-argmax coordinates, normalised to [0, 1]
        logits = self.heatmap_conv(self.blocks(self.patch_embed(x)))
        b, k, h, w = logits.shape
        probs = F.softmax(logits.reshape(b, k, h * w), dim=-1).reshape(b, k, h, w)
        xs = (torch.arange(w, dtype=probs.dtype, device=probs.device) + 0.5) / w
        ys = (torch.arange(h, dtype=probs.dtype, device=probs.device) + 0.5) / h
        return torch.stack([(probs.sum(dim=2) * xs).sum(-1), (probs.sum(dim=3) * ys).sum(-1)], dim=-1)


@dataclass
class MfldBundle:
    model: MfldNet
    device: str
    checkpoint: str
    params_millions: float
    train_fraction_note: str


def _describe_training(ckpt: dict) -> str:
    cfg = ckpt.get("train_cfg", {})
    frac, train_frac = cfg.get("labelled_frac"), cfg.get("train_frac")
    if frac is None or train_frac is None:
        return "training split not recorded in the checkpoint"
    return f"trained on {frac * train_frac * 100:.0f}% of the images ({cfg.get('epochs', '?')} epochs, whole-image input)"


def load_mfld_bundle(force_reload: bool = False) -> "MfldBundle | None":
    # loads the checkpoint once; returns None (and logs) when the file is missing so the app keeps working without it
    global _BUNDLE
    with _LOCK:
        if _BUNDLE is not None and not force_reload:
            return None if _BUNDLE is _MISSING else _BUNDLE  # type: ignore[return-value]
        path = Path(settings.mfld_checkpoint_path)
        if not path.is_file():
            logger.warning("MFLD-Net checkpoint not found at %s; the comparison tab is disabled.", path)
            _BUNDLE = _MISSING  # type: ignore[assignment]
            return None
        device = "cuda" if settings.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device)
        model = MfldNet(**ckpt["model_cfg"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        _BUNDLE = MfldBundle(model, device, str(path), sum(p.numel() for p in model.parameters()) / 1e6, _describe_training(ckpt))
        return _BUNDLE


def preprocess(image_rgb: np.ndarray, size: int) -> torch.Tensor:
    # (H, W, 3) uint8 RGB -> (1, 3, size, size) normalised tensor, exactly as mfld-net's predict_images does
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


@torch.no_grad()
def predict_keypoints(bundle: MfldBundle, image_rgb: np.ndarray) -> np.ndarray:
    # returns (K, 2) keypoints in the ORIGINAL image's pixels
    h, w = image_rgb.shape[:2]
    coords = bundle.model(preprocess(image_rgb, bundle.model.img_size).to(bundle.device))[0].cpu().numpy()
    return coords * np.array([w, h], dtype=np.float32)
