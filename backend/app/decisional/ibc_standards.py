# ibc exhibition standrds thresholds for the five measurable criteria
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaultBand:
    label: str
    low: float | None  # none means no lower boud
    high: float | None  # none means no upper bound


# caudal spread angle in degrees - ideal is exactly 180
CAUDAL_SPREAD_ANGLE_BANDS = [
    FaultBand("Disqualify", 210.0, None),
    FaultBand("Major Fault", 195.0, 209.0),
    FaultBand("Slight Fault", 181.0, 194.0),
    FaultBand("Ideal", 180.0, 180.0),
    FaultBand("Major Fault", 166.0, 179.0),
    FaultBand("Disqualify", None, 165.0),
]
CAUDAL_SPREAD_ANGLE_THRESHOLD = 180.0  # tau used directl in the TSI formula

# fin-to-body ratios must be at least 0.50, anal fin length-to-width must stay below 1.50
# still a simplification, asymmetric severity is a known open item
RATIO_THRESHOLDS = {
    "dorsal-body-ratio": 0.50,
    "anal-body-ratio": 0.50,
    "caudal-body-ratio": 0.50,
    "anal-length-width-ratio": 1.50,
}

# ratio criteria where the value must stay BELOW the threshold (1.5x or more = form fault)
UPPER_BOUND_RATIOS = frozenset({"anal-length-width-ratio"})

CRITERION_THRESHOLDS = {
    "caudal-spread-angle": CAUDAL_SPREAD_ANGLE_THRESHOLD,
    **RATIO_THRESHOLDS,
}