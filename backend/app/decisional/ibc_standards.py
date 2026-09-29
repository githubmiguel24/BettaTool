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

# IBC 2025 Exhibition Standards Book 1, Ch. 5 "DIMENSION", p. 47: all three
# fin-to-body ratios are "at least one-half the length of the body" -- a
# one-sided minimum of 0.50, not the placeholder two-sided targets this
# module carried before (see claude/thesis-revision-punchlist.md, C3).
#
# Fin-to-fin ratios: HALFMOON SPECIFIC FAULTS, p. 57, items 6-15. Ideal is
# 1.00 for both. NOTE this is still a simplification: Book 1 grades the
# fin-to-fin miss direction asymmetrically (anal shorter than caudal is an
# instant Severe fault, item 10; dorsal longer than caudal is an instant
# Severe fault, item 15) and severities in millimetres this pipeline has no
# scale reference to compute (C5). `classify_ratio`'s single >= threshold
# only recovers the PASS/FAULT direction, not that asymmetric severity --
# tracked as a separate, already-known open item, not fixed here.
RATIO_THRESHOLDS = {
    "dorsal-body-ratio": 0.50,
    "anal-body-ratio": 0.50,
    "caudal-body-ratio": 0.50,
    "anal-caudal-ratio": 1.00,
    "dorsal-caudal-ratio": 1.00,
}

CRITERION_THRESHOLDS = {
    "caudal-spread-angle": CAUDAL_SPREAD_ANGLE_THRESHOLD,
    **RATIO_THRESHOLDS,
}