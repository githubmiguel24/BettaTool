"""Process-wide cached loader for the HRNet keypoint detector.

Two things this module deliberately does NOT do:

1. It does not silently pretend an untrained model is trained. If no
   checkpoint is found at `settings.model_checkpoint_path`, the model is
   still constructed and served — so the full pipeline can be demonstrated
   end to end before training finishes — but `ModelBundle.trained` is
   False and every downstream response is tagged accordingly. An untrained
   HRNet emits essentially arbitrary keypoints; a demo that does not say so
   on its face is a thesis-integrity problem, not a UI nicety.

2. It does not reload the checkpoint per request. Constructing HRNet-W32
   and reading its weights takes seconds; doing that inside a request
   handler would make every upload look pathologically slow and would
   thrash memory under concurrent requests.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from app.core.config import settings
from app.perception.hrnet import HRNetKeypointDetector

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_BUNDLE: "ModelBundle | None" = None


@dataclass
class ModelBundle:
    """A loaded detector plus provenance for the report's audit trail."""

    model: HRNetKeypointDetector
    device: str
    trained: bool
    checkpoint_path: str | None
    checkpoint_epoch: int | None = None
    checkpoint_val_metric: float | None = None

    @property
    def status_note(self) -> str:
        if self.trained:
            return f"Loaded trained checkpoint: {self.checkpoint_path}"
        return (
            "NO TRAINED CHECKPOINT LOADED - the perceptual tier is running "
            "randomly initialized weights. Keypoints and every measurement "
            "derived from them are meaningless; this mode exists only to "
            "exercise the end-to-end integration path."
        )


def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("device=%s requested but CUDA is unavailable; falling back to CPU.", requested)
        return "cpu"
    return requested


def load_model_bundle(force_reload: bool = False) -> ModelBundle:
    """Returns the process-wide `ModelBundle`, constructing it on first call."""
    global _BUNDLE
    if _BUNDLE is not None and not force_reload:
        return _BUNDLE

    with _LOCK:
        if _BUNDLE is not None and not force_reload:
            return _BUNDLE

        device = _resolve_device(settings.device)
        model = HRNetKeypointDetector()

        ckpt_path = Path(settings.model_checkpoint_path)
        trained = False
        epoch: int | None = None
        val_metric: float | None = None

        if ckpt_path.is_file():
            state = torch.load(ckpt_path, map_location=device)
            # training/train_hrnet.py writes {"model": ..., "epoch": ..., ...};
            # tolerate a bare state_dict too, since a hand-exported checkpoint
            # is a very likely thing to be handed this path.
            state_dict = state.get("model", state) if isinstance(state, dict) else state
            model.load_state_dict(state_dict)
            trained = True
            if isinstance(state, dict):
                epoch = state.get("epoch")
                val_metric = state.get("best_val_metric") or state.get("val_metric")
            logger.info("Loaded trained checkpoint from %s (epoch=%s)", ckpt_path, epoch)
        else:
            logger.warning(
                "No checkpoint at %s - serving RANDOMLY INITIALIZED weights. "
                "Predictions are not meaningful.",
                ckpt_path,
            )

        model.to(device).eval()
        _BUNDLE = ModelBundle(
            model=model,
            device=device,
            trained=trained,
            checkpoint_path=str(ckpt_path) if trained else None,
            checkpoint_epoch=epoch,
            checkpoint_val_metric=val_metric,
        )
        return _BUNDLE


def reset_model_bundle() -> None:
    """Drops the cached bundle. Used by tests and by a future /admin/reload."""
    global _BUNDLE
    with _LOCK:
        _BUNDLE = None
