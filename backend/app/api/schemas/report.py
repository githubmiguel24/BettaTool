# response model for the assesment report

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

    # perceptual tier output in original image pixls
    keypoints: list[KeypointPrediction] = []
    image_width: int = 0
    image_height: int = 0

    # provenance and integrty stuff
    # if model_trained is false use random weights (no chkpoint). UI must show a blocking banner cause all numbers are basically garbage in that mode
    model_trained: bool = False
    warnings: list[str] = []