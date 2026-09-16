"""Assembles an `AssessmentReport` from pipeline output."""

from __future__ import annotations

import uuid
from datetime import datetime

from app.api.schemas.measurement import MeasurementResult
from app.api.schemas.report import AssessmentReport
from app.decisional.abstention_gate import CriterionResult
from app.reports.labels import CRITERION_LABELS


def build_report(image_id: str, criterion_results: list[CriterionResult]) -> AssessmentReport:
    measurements = [
        MeasurementResult(
            criterion_key=r.criterion_key,
            label=CRITERION_LABELS.get(r.criterion_key, r.criterion_key),
            value=r.measurement,
            uncertainty=r.uncertainty,
            tsi=r.tsi,
            rmse=r.actual_rmse,
            decision=r.decision,
        )
        for r in criterion_results
    ]

    return AssessmentReport(
        id=str(uuid.uuid4()),
        image_id=image_id,
        analysis_date=datetime.utcnow(),
        measurements=measurements,
    )
