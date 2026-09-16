"""GET /reports/{id}/export — PDF/CSV export of a saved report."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.core.db import get_report
from app.reports.exporters import export_csv, export_pdf

router = APIRouter(prefix="/reports", tags=["export"])


@router.get("/{report_id}/export")
async def export_report(report_id: str, format: str = Query("csv", pattern="^(csv|pdf)$")) -> Response:
    report = get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")

    if format == "csv":
        return Response(
            content=export_csv(report),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="report-{report_id[:8]}.csv"'},
        )

    try:
        content = export_pdf(report)
    except NotImplementedError as exc:
        # Fail as a clean, documented 501 rather than a 500 traceback - the
        # frontend disables the PDF button on the strength of this contract.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report-{report_id[:8]}.pdf"'},
    )
