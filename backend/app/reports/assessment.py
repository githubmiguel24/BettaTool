"""Derives an overall quality assessment from confident criterion decisions."""

from __future__ import annotations

from app.api.schemas.assessment import FinalAssessment
from app.api.schemas.measurement import MeasurementResult
from app.decisional.ibc_standards import CRITERION_THRESHOLDS

QUALITY_TIERS = (
    "Poor Quality",
    "Low Quality",
    "Fair Quality",
    "Good Quality",
    "Great Quality",
    "Perfect Quality",
)


def assess_quality(measurements: list[MeasurementResult]) -> FinalAssessment:
    expected_keys = set(CRITERION_THRESHOLDS)
    decisions = {
        measurement.criterion_key: measurement.decision
        for measurement in measurements
        if measurement.criterion_key in expected_keys
    }
    passed = sum(decision == "Confident Pass" for decision in decisions.values())
    faulted = sum(
        decision.startswith("Confident ") and decision != "Confident Pass"
        for decision in decisions.values()
    )
    total = len(expected_keys)
    deferred = total - passed - faulted
    confidence = (passed + faulted) / total if total else 0.0

    if deferred == total:
        quality = "Inconclusive"
        summary = (
            f"{deferred} of {total} criteria could not be confidently classified. "
            "Judge review is needed before show quality can be confirmed."
        )
    else:
        quality = QUALITY_TIERS[passed]
        summary = (
            f"{passed} of {total} criteria passed confidently, {faulted} had a "
            f"confident fault, and {deferred} need judge review. This project "
            "quality tier is based on the number of confidently passed criteria."
        )

    return FinalAssessment(
        quality=quality,
        confidence=confidence,
        criteria_total=total,
        criteria_passed=passed,
        criteria_faulted=faulted,
        criteria_deferred=deferred,
        summary=summary,
    )
