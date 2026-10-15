# =============================================================================
# Program Title:    Assessment Report Builder
# Programmers:      Daanay, Kurt Michael M.; Maraño, Alexander P.; Trinidad, Charles Miguel L.; Verceles, Rogel Cristian S.
# System Fit:       Final part of the decisional module. Which takes the final output from the Decisional Tier 
#                   and formats it for the frontend API response.
# Written:          September 14, 2026
# Revised:          October 15, 2026
# Purpose:          Assembles the final Assessment Report by aggregating individual criterion decisions, 
#                   geometric measurements, propagated uncertainties, the TSI, and the overall final 
#                   quality assessment into a single structured object.
# Data Structure:   Uses lists to hold multiple `MeasurementResult` objects and returns a comprehensive 
#                   `AssessmentReport` data class/schema. Uses dictionaries (`CRITERION_LABELS`, `CRITERION_LANDMARKS`) 
#                   for mapping metadata.
# Algorithm:        N/A
# Control:          Uses a list comprehension (for loop) to iterate through the list of `criterion_results` and map them into the required schema.
# =============================================================================

#Assembles an AssessmentReport

from __future__ import annotations

import uuid
from datetime import datetime

from app.api.schemas.measurement import MeasurementResult
from app.api.schemas.report import AssessmentReport
from app.decisional.rule_engine import CriterionResult
from app.reports.assessment import assess_quality
from app.reports.labels import CRITERION_LABELS, CRITERION_LANDMARKS


def build_report(image_id: str, criterion_results: list[CriterionResult]) -> AssessmentReport:
    measurements = [
        MeasurementResult(
            criterion_key=r.criterion_key,
            label=CRITERION_LABELS.get(r.criterion_key, r.criterion_key),
            value=r.measurement,
            uncertainty=r.uncertainty,
            tsi=r.tsi,
            rmse=r.sigma_hat,
            decision=r.decision,
            landmark_indices=CRITERION_LANDMARKS.get(r.criterion_key, []),
        )
        for r in criterion_results
    ]

    return AssessmentReport(
        id=str(uuid.uuid4()),
        image_id=image_id,
        analysis_date=datetime.utcnow(),
        measurements=measurements,
        final_assessment=assess_quality(measurements),
    )