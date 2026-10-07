# cached loader for the hrnet keypoint model so we dont reload weights on every request and tracks if its actually trained

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


# holds the loaded model and metadata for tracking
@dataclass
class ModelBundle:

    model: HRNetKeypointDetector
    device: str
    trained: bool
    checkpoint_path: str | None
    checkpoint_epoch: int | None = None
    checkpoint_val_metric: float | None = None

    # returns a quick string telling us if we are using real weights
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


# fallback to cpu if cuda is set in config but not actually availble
def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("device=%s requested but CUDA is unavailable; falling back to CPU.", requested)
        return "cpu"
    return requested


# loads the hrnet bundle once and caches it in memory
def load_model_bundle(force_reload: bool = False) -> ModelBundle:
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
            # handle both full training dicts and raw state dicts just in case
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


# raised when there is no trained checkpoint, so the system refuses to produce keypoints or measurements
class ModelNotTrainedError(RuntimeError):
    pass


# same as load_model_bundle but refuses to hand back randomly initialized weights
def require_trained_bundle() -> ModelBundle:
    bundle = load_model_bundle()
    if not bundle.trained:
        raise ModelNotTrainedError(
            f"No trained model found at '{settings.model_checkpoint_path}'. "
            "Analysis is disabled until a trained checkpoint is available."
        )
    return bundle


# clears the cached model mostly for unit tests or Manual reloads
def reset_model_bundle() -> None:
    global _BUNDLE
    with _LOCK:
        _BUNDLE = None