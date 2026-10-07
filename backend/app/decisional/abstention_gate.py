# handles selective abstention - only classify if confident, otherwise defer to human
from __future__ import annotations
from dataclasses import dataclass

from app.analytical.tsi import is_confident
from app.decisional.rule_engine import classify

@dataclass
class CriterionResult:
    criterion_key: str
    measurement: float
    threshold: float  # tau compared against for audit
    uncertainty: float  # reported margin of error U(y) = s_c * k * sqrt(J Sigma J^T); not used for the decision
    tsi: float  # threshold sensitivity index in px
    sigma_hat: float  # calibrated predicted keypoint uncertainty in px, compared against tsi
    decision: str  # pass, fault, or defer
    label: str  # what the rule engine actually says


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
            # still a fault, but say how severe (Slight Fault / Major Fault / Disqualify)
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