"""Minimal report persistence.

TODO: replace this in-memory dict with a real database (SQLite/Postgres)
before deployment — history is currently lost on every restart.
"""

from __future__ import annotations

from app.api.schemas.report import AssessmentReport

_REPORTS: dict[str, AssessmentReport] = {}


def save_report(report: AssessmentReport) -> None:
    _REPORTS[report.id] = report


def get_report(report_id: str) -> AssessmentReport | None:
    return _REPORTS.get(report_id)


def list_reports() -> list[AssessmentReport]:
    return sorted(_REPORTS.values(), key=lambda r: r.analysis_date, reverse=True)
