# =============================================================================
# Program Title     : IBC Standards Criterias

# Programmers       : Daanay, Kurt Michael M.
#                     Trinidad, Charles Miguel L.
#                     Verceles, Rogel Cristian S.

# System Fit        : Part of the Decisional tier/module, 
#                     that holds the constants of the metrics
#                     used by rule_engine.py

# Written           : September 12, 2026
# Revised           : October 14, 2026 
# Purpose           : This stores the IBC standards' thresholds for 
#                     each of the five criteria. 
# Data Structure: DataClass only
# =============================================================================



from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class FaultBand:
    label: str
    low: float | None  
    high: float | None 


# Caudal spread angle in degrees. The ideal is exactly 180.
CAUDAL_SPREAD_ANGLE_BANDS = [
    FaultBand("Disqualify", 210.0, None),
    FaultBand("Major Fault", 195.0, 209.0),
    FaultBand("Slight Fault", 181.0, 194.0),
    FaultBand("Ideal", 180.0, 180.0),
    FaultBand("Major Fault", 166.0, 179.0),
    FaultBand("Disqualify", None, 165.0),
]
CAUDAL_SPREAD_ANGLE_THRESHOLD = 180.0  # tau used directl in the TSI formula

# Fin-to-body ratios must be at least 0.50, anal fin length-to-width must stay below 1.50
RATIO_THRESHOLDS = {
    "dorsal-body-ratio": 0.50,
    "anal-body-ratio": 0.50,
    "caudal-body-ratio": 0.50,
    "anal-length-width-ratio": 1.50,
}

# Ratio criteria where the value must stay BELOW the threshold
UPPER_BOUND_RATIOS = frozenset({"anal-length-width-ratio"})

CRITERION_THRESHOLDS = {
    "caudal-spread-angle": CAUDAL_SPREAD_ANGLE_THRESHOLD,
    **RATIO_THRESHOLDS,
}