"""IBC Exhibition Standards thresholds for the six measurable criteria.

Numeric bands for the caudal spread angle are taken directly from the
thesis's evaluation scorecard (Appendix 1, Table 5). The five fin-ratio
thresholds are placeholders — replace with the exact figures from the IBC
Exhibition Standards Book (2025) once transcribed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaultBand:
    label: str
    low: float | None  # None = unbounded below
    high: float | None  # None = unbounded above


# Caudal spread angle (degrees) — ideal is exactly 180 degrees.
CAUDAL_SPREAD_ANGLE_BANDS = [
    FaultBand("Disqualify", 210.0, None),
    FaultBand("Major Fault", 195.0, 209.0),
    FaultBand("Slight Fault", 181.0, 194.0),
    FaultBand("Ideal", 180.0, 180.0),
    FaultBand("Major Fault", 166.0, 179.0),
    FaultBand("Disqualify", None, 165.0),
]
CAUDAL_SPREAD_ANGLE_THRESHOLD = 180.0  # tau used directly in the TSI formula

# TODO: replace with exact IBC Exhibition Standards Book (2025) values.
RATIO_THRESHOLDS = {
    "dorsal-body-ratio": 0.60,
    "anal-body-ratio": 0.60,
    "caudal-body-ratio": 0.75,
    "anal-caudal-ratio": 0.80,
    "dorsal-caudal-ratio": 0.80,
}

CRITERION_THRESHOLDS = {
    "caudal-spread-angle": CAUDAL_SPREAD_ANGLE_THRESHOLD,
    **RATIO_THRESHOLDS,
}
