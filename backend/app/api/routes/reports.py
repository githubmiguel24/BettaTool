"""GET /reports, /reports/{id} — assessment history."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas.report import AssessmentReport
from app.core.db import get_report, list_reports

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("", response_model=list[AssessmentReport])
async def get_reports() -> list[AssessmentReport]:
    return list_reports()


@router.get("/{report_id}", response_model=AssessmentReport)
async def get_report_by_id(report_id: str) -> AssessmentReport:
    report = get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report
