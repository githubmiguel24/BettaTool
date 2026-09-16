"""Response model for the full Reliability-Annotated Assessment Report."""

from datetime import datetime

from pydantic import BaseModel

from app.api.schemas.keypoint import KeypointPrediction
from app.api.schemas.measurement import MeasurementResult


class AssessmentReport(BaseModel):
    id: str
    fish_class: str = "Halfmoon Longfin"
    image_id: str
    analysis_date: datetime
    model_name: str = "HRNet-W32 (probabilistic heatmap)"
    measurements: list[MeasurementResult]

    # --- Perceptual-tier output, in ORIGINAL uploaded-image pixels ---
    keypoints: list[KeypointPrediction] = []
    image_width: int = 0
    image_height: int = 0

    # --- Provenance / integrity ---
    # `model_trained` is False when the perceptual tier is running randomly
    # initialized weights because no checkpoint was found. The UI surfaces
    # this as a blocking banner: every number below it is meaningless in
    # that mode, and a screenshot of this report must never be presented as
    # a result without the banner attached.
    model_trained: bool = False
    warnings: list[str] = []
