# rule engine: mesurement -> pass/faut (no uncertainty checks here)

from __future__ import annotations

from app.decisional.ibc_standards import CAUDAL_SPREAD_ANGLE_BANDS, CRITERION_THRESHOLDS

def classify_caudal_spread(angle_degrees: float) -> str:
    # map caudal spred angle to the ibc fault band labl
    for band in CAUDAL_SPREAD_ANGLE_BANDS:
        low_ok = band.low is None or angle_degrees >= band.low
        high_ok = band.high is None or angle_degrees <= band.high
        if low_ok and high_ok:
            return band.label
    return "Unclassified"

def classify_ratio(criterion_key: str, value: float) -> str:
    # pass/fault for ratio based criterion against its single threshold
    threshold = CRITERION_THRESHOLDS[criterion_key]
    return "Pass" if value >= threshold else "Fault"

def classify(criterion_key: str, value: float) -> str:
    # route to correct classifier
    if criterion_key == "caudal-spread-angle":
        return classify_caudal_spread(value)
    return classify_ratio(criterion_key, value)