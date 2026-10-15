# =============================================================================
# Program Title:    Main System Rule Engine
# Programmers:      Daanay, Maraño, Trinidad, Verceles
# System Fit:       Part of the Decisional tier/module, its called once per criterion, 
#                   the output is used in the Assessment Report. Also handles the 
#                   caudal angle criterion classifications.
# Written:          September 12, 2026
# Revised:          October 14, 2026 
# Purpose:          Main gate that decides if the output can be trusted. It defers 
#                   when it cannot trust its output. It also converts measurements 
#                   into the equivalent IBC classification.
# Data Structure:   DataClass named criterionresult; uses a linear search to look
# Algorithm:        N/A
# Control:          If/else conditionals
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass

from app.analytical.tsi import is_confident
from app.decisional.ibc_standards import CAUDAL_SPREAD_ANGLE_BANDS, CRITERION_THRESHOLDS, UPPER_BOUND_RATIOS


@dataclass
class CriterionResult:
    criterion_key: str
    measurement: float
    threshold: float  
    uncertainty: float 
    tsi: float 
    sigma_hat: float 
    decision: str  
    label: str 


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
    if criterion_key in UPPER_BOUND_RATIOS:
        return "Pass" if value < threshold else "Fault"
    return "Pass" if value >= threshold else "Fault"


def classify(criterion_key: str, value: float) -> str:
    # route to correct classifier
    if criterion_key == "caudal-spread-angle":
        return classify_caudal_spread(value)
    return classify_ratio(criterion_key, value)


def evaluate_criterion(
    criterion_key: str,
    measurement: float,
    threshold: float,
    uncertainty: float,
    tsi: float,
    sigma_hat: float,
) -> CriterionResult:

    # evaluates criterion and defers when the predicted keypoint uncertainty reaches the TSI
    label = classify(criterion_key, measurement)
    
    if not is_confident(sigma_hat, tsi):
        decision = "Defer to Judge"
    else:
        # mark as pass or fault
        if label in ("Pass", "Ideal"):
            decision = "Confident Pass"
        elif criterion_key == "caudal-spread-angle":
            # shows if the fault is Slight Fault / Major Fault / Disqualify
            decision = f"Confident {label}"
        else:
            decision = "Confident Fault"

    return CriterionResult(
        criterion_key=criterion_key,
        measurement=measurement,
        threshold=threshold,
        uncertainty=uncertainty,
        tsi=tsi,
        sigma_hat=sigma_hat,
        decision=decision,
        label=label,
    )