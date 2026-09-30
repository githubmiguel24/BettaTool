# handles selective abstention - only classify if confident, otherwise defer to human
from __future__ import annotations
from dataclasses import dataclass, field

from app.analytical.tsi import is_confident_gum
from app.decisional.rule_engine import classify

@dataclass
class CriterionResult:
    criterion_key: str
    measurement: float
    threshold: float  # tau compared against for audit
    uncertainty: float  # k*sqrt(J Sigma J^T) used for decision making
    tsi: float  # legacy tsi for audit only
    actual_rmse: float  # legacy rmse for audit only
    decision: str  # pass, fault, or defer
    label: str  # what the rule engine actually says
    low_visibility_keypoints: list[int] = field(default_factory=list) # todo: wire this up later so low visbility auto defers


def evaluate_criterion(
    criterion_key: str,
    measurement: float,
    threshold: float,
    uncertainty: float,
    tsi: float,
    actual_rmse: float,
) -> CriterionResult:
    # evaluates criterion and checks uncertainty to see if we are confident
    label = classify(criterion_key, measurement)
    # check if confident based on uncertainty
    if not is_confident_gum(measurement, threshold, uncertainty):
        decision = "Defer to Judge"
    else:
        # mark as pass or fault
        decision = "Confident Fault" if label not in ("Pass", "Ideal") else "Confident Pass"

    return CriterionResult(
        criterion_key=criterion_key,
        measurement=measurement,
        threshold=threshold,
        uncertainty=uncertainty,
        tsi=tsi,
        actual_rmse=actual_rmse,
        decision=decision,
        label=label,
    )