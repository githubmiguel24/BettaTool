"""POST /analyze - upload a lateral flare image, run the full pipeline, get a report.

Wires all three tiers behind one endpoint:

    upload bytes
      -> decode + letterbox (app/perception/preprocess.py)
      -> HRNet-W32 forward pass (app/perception/loader.py, cached)
      -> soft-argmax + covariance + visibility (app/pipeline.py)
      -> inverse letterbox back to original-image pixels
      -> GUM propagation + TSI + IBC rule engine + abstention gate
      -> AssessmentReport (persisted via app/core/db.py)

IMPORTANT (integrity): when no trained checkpoint is present, this endpoint
still answers 200 with a structurally complete report, but `model_trained`
is False and `warnings` carries an explicit notice. That is deliberate - it
lets the integration path be demonstrated and graded before training
finishes - but it means a consumer MUST check `model_trained` before
treating any number in the response as a measurement. The frontend renders a
blocking red banner in that case.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas.keypoint import KeypointPrediction
from app.api.schemas.report import AssessmentReport
from app.core.db import save_report
from app.perception.keypoints import KEYPOINT_SHORT_CODES
from app.perception.loader import load_model_bundle
from app.perception.preprocess import decode_image, preprocess_image
from app.pipeline import AssessmentPipeline
from app.reports.builder import build_report
from app.reports.labels import KEYPOINT_LABELS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("", response_model=AssessmentReport)
async def analyze_image(file: UploadFile = File(...)) -> AssessmentReport:
    """Runs the perception -> analytical -> decisional pipeline on one image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    orig_h, orig_w = image.shape[:2]
    tensor, to_crop = preprocess_image(image)
    to_original = to_crop.inverse()

    bundle = load_model_bundle()
    pipeline = AssessmentPipeline(bundle.model, device=bundle.device)

    try:
        output = pipeline.analyze_detailed(tensor, to_original=to_original)
    except Exception as exc:  # noqa: BLE001 - surface as 500 with context, don't leak a bare traceback
        logger.exception("Pipeline failed on upload %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    report = build_report(image_id=file.filename or "upload", criterion_results=output.criteria)

    low_vis = set(output.low_visibility_keypoints)
    report.keypoints = [
        KeypointPrediction(
            index=i,
            name=KEYPOINT_SHORT_CODES[i],
            label=KEYPOINT_LABELS[i],
            x=float(output.keypoints[i, 0]),
            y=float(output.keypoints[i, 1]),
            sigma_x=float(output.covariances[i, 0, 0] ** 0.5),
            sigma_y=float(output.covariances[i, 1, 1] ** 0.5),
            rho=_correlation(output.covariances[i]),
            visibility=float(output.visibility[i]),
            low_visibility=i in low_vis,
        )
        for i in range(len(KEYPOINT_SHORT_CODES))
    ]
    report.image_width = int(orig_w)
    report.image_height = int(orig_h)
    report.model_trained = bundle.trained
    report.warnings = [] if bundle.trained else [bundle.status_note]

    if low_vis:
        names = ", ".join(KEYPOINT_LABELS[i] for i in sorted(low_vis))
        report.warnings.append(
            f"Low predicted visibility for: {names}. Criteria depending on these "
            f"landmarks are less reliable than their stated uncertainty suggests."
        )

    save_report(report)
    return report


def _correlation(cov) -> float:
    """rho = Sigma_xy / (sigma_x * sigma_y), clamped to [-1, 1]."""
    denom = float(cov[0, 0] ** 0.5) * float(cov[1, 1] ** 0.5)
    if denom <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, float(cov[0, 1]) / denom))
