# =============================================================================
# Program Title: Abstention Gate for Deferral
# Programmers: Daanay, Maraño, Trinidad, Verceles
# System Fit:  Part of the Decisional tier/module, 
#               its called once per criterion, the output
#               is used in the Assessment Report.
# Written: September 12, 2026
# Revised: October 14, 2026 
# Purpose:  Main gate that decides if the output can be trusted.
#           It defers when it cannot trust its output. 
# Data Structure: DataClass named criterionresult
# Algorithm: N/A
# Control: If/else conditionals
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass

from app.analytical.tsi import is_confident
from app.decisional.rule_engine import classify

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