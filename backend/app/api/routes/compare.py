# endpoints behind the "Compare with MFLD-Net" tab: both models on one uploaded photo + the measured benchmark
# this is an inspection aid only: nothing here is saved to a report and the assessment pipeline is untouched

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.schemas.compare import ComparedKeypoint, ComparisonResponse, CompareSummary, ModelInfo
from app.core.config import settings
from app.perception.keypoints import KEYPOINT_GROUPS, KEYPOINT_SHORT_CODES, Keypoint
from app.perception.loader import require_trained_bundle
from app.perception.mfld import load_mfld_bundle, predict_keypoints
from app.perception.preprocess import decode_image, preprocess_image
from app.pipeline import AssessmentPipeline
from app.reports.labels import KEYPOINT_LABELS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze/compare", tags=["compare"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DISAGREE_PCT = 5.0  # two models count as disagreeing on a keypoint when they are >5% of body length apart (the PCK@5% scale)
BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "data" / "mfld_benchmark.json"
IMAGES_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"  # the labelled photos the benchmark examples come from


def compare_keypoints(ours: np.ndarray, mfld: np.ndarray) -> tuple[list[ComparedKeypoint], float, CompareSummary]:
    # per-keypoint distances between the two models; normalised by OUR snout-to-peduncle-midpoint length
    ped_mid = 0.5 * (ours[Keypoint.CAUDAL_PEDUNCLE_TOP] + ours[Keypoint.CAUDAL_PEDUNCLE_BOTTOM])
    body = float(np.linalg.norm(ours[Keypoint.SNOUT_TIP] - ped_mid))
    body = max(body, 1.0)
    dist = np.linalg.norm(ours - mfld, axis=1)
    pct = dist / body * 100.0
    items = [
        ComparedKeypoint(
            index=i, name=KEYPOINT_SHORT_CODES[i], label=KEYPOINT_LABELS[i], group=str(KEYPOINT_GROUPS[i].value if hasattr(KEYPOINT_GROUPS[i], "value") else KEYPOINT_GROUPS[i]),
            ours_x=float(ours[i, 0]), ours_y=float(ours[i, 1]), mfld_x=float(mfld[i, 0]), mfld_y=float(mfld[i, 1]),
            distance_px=float(dist[i]), distance_pct_body=float(pct[i]),
        )
        for i in range(len(KEYPOINT_SHORT_CODES))
    ]
    order = np.argsort(-pct)[:3]
    summary = CompareSummary(
        mean_distance_pct_body=float(pct.mean()), median_distance_pct_body=float(np.median(pct)),
        n_disagree=int((pct > DISAGREE_PCT).sum()), disagree_threshold_pct=DISAGREE_PCT,
        most_different=[KEYPOINT_LABELS[int(i)] for i in order],
    )
    return items, body, summary


@router.post("", response_model=ComparisonResponse)
async def compare_models(file: UploadFile = File(...)) -> ComparisonResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bundle = require_trained_bundle()  # no trained model -> 503, no results
    mfld = load_mfld_bundle()
    if mfld is None:
        raise HTTPException(status_code=503, detail=f"MFLD-Net checkpoint not found at '{settings.mfld_checkpoint_path}'. Set MFLD_CHECKPOINT_PATH to enable the comparison.")

    tensor, to_crop = preprocess_image(image)
    try:
        out = AssessmentPipeline(bundle.model, device=bundle.device).analyze_detailed(tensor, to_original=to_crop.inverse())
        mfld_pts = predict_keypoints(mfld, image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Comparison failed on upload %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}") from exc

    items, body, summary = compare_keypoints(np.asarray(out.keypoints, dtype=float), np.asarray(mfld_pts, dtype=float))
    h, w = image.shape[:2]
    ours_params = sum(p.numel() for p in bundle.model.parameters()) / 1e6
    return ComparisonResponse(
        image_width=int(w), image_height=int(h), body_length_px=body, keypoints=items, summary=summary,
        ours=ModelInfo(name="HRNet-W32 (this system)", params_millions=ours_params, note="whole photo letterboxed to 384 px; trained on fish-box crops"),
        mfld=ModelInfo(name="MFLD-Net", params_millions=mfld.params_millions, note=mfld.train_fraction_note),
        notes=[
            "No ground truth exists for an uploaded photo: these are two models' answers, so a difference shows disagreement, not which one is right.",
            "The measured accuracy comparison (held-out test set with labels) is in the benchmark panel below.",
        ],
    )


@router.get("/benchmark")
async def benchmark() -> dict:
    # the measured test-set numbers written by training/export_mfld_benchmark.py (nothing is computed here)
    if not BENCHMARK_PATH.is_file():
        raise HTTPException(status_code=404, detail="Benchmark file not generated yet: run `python -m training.export_mfld_benchmark`.")
    return json.loads(BENCHMARK_PATH.read_text())


@router.get("/examples/{image_id}/image")
async def example_image(image_id: str) -> FileResponse:
    # serves ONLY the photos listed as examples in the benchmark file (no arbitrary file access)
    if not BENCHMARK_PATH.is_file():
        raise HTTPException(status_code=404, detail="Benchmark file not generated yet.")
    examples = {e["image_id"]: e["file_name"] for e in json.loads(BENCHMARK_PATH.read_text()).get("examples", [])}
    name = examples.get(image_id)
    path = IMAGES_DIR / name if name else None
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Example photo not available on this machine.")
    return FileResponse(path)
