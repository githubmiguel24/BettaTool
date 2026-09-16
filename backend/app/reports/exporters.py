"""CSV/PDF export of a saved AssessmentReport."""

from __future__ import annotations

import csv
import io

from app.api.schemas.report import AssessmentReport


def export_csv(report: AssessmentReport) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Criterion", "Value", "Uncertainty", "TSI", "RMSE", "Decision"])
    for m in report.measurements:
        writer.writerow([m.label, m.value, m.uncertainty, m.tsi, m.rmse, m.decision])
    return buffer.getvalue()


def export_pdf(report: AssessmentReport) -> bytes:
    """TODO: render a proper PDF (e.g. via reportlab or weasyprint).

    Raises for now so the export endpoint fails loudly instead of silently
    returning an empty/broken file.
    """
    raise NotImplementedError("PDF export not yet implemented — see TODO above.")
