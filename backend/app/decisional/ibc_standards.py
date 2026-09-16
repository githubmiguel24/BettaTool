# ibc exhibition standrds thresholds for the six measurable criteria
# todo: update and double chck the rules here
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaultBand:
    label: str
    low: float | None  # none means no lower boud
    high: float | None  # none means no upper bound


# caudal spread angle in degrees - idel is exactly 180
CAUDAL_SPREAD_ANGLE_BANDS = [
    FaultBand("Disqualify", 210.0, None),
    FaultBand("Major Fault", 195.0, 209.0),
    FaultBand("Slight Fault", 181.0, 194.0),
    FaultBand("Ideal", 180.0, 180.0),
    FaultBand("Major Fault", 166.0, 179.0),
    FaultBand("Disqualify", None, 165.0),
]
CAUDAL_SPREAD_ANGLE_THRESHOLD = 180.0  # tau used directl in the TSI formula

# todo: replace with actual ibc 2025 book values 
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