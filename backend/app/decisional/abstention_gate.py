# Selective abstention rule - only classify if confident, otherwis defer to human

from __future__ import annotations

from dataclasses import dataclass, field

from app.analytical.tsi import is_confident
from app.decisional.rule_engine import classify


@dataclass
class CriterionResult:
    criterion_key: str
    measurement: float
    uncertainty: float  # same units as the measurment
    tsi: float  # In pixels
    actual_rmse: float  # pixels
    decision: str  # pass, fault, or defer
    label: str  # what the rule engine says
    # TODO: wire this up later so low visbility auto defers
    low_visibility_keypoints: list[int] = field(default_factory=list)


def evaluate_criterion(
    criterion_key: str,
    measurement: float,
    uncertainty: float,
    tsi: float,
    actual_rmse: float,
) -> CriterionResult:
    label = classify(criterion_key, measurement)

    # check if we can Trust the keypoint
    if not is_confident(actual_rmse, tsi):
        decision = "Defer to Judge"
    else:
        # mark as pass or fault
        decision = "Confident Fault" if label not in ("Pass", "Ideal") else "Confident Pass"

    return CriterionResult(
        criterion_key=criterion_key,
        measurement=measurement,
        uncertainty=uncertainty,
        tsi=tsi,
        actual_rmse=actual_rmse,
        decision=decision,
        label=label,
    )