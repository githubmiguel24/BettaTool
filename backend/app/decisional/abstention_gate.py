# Selective abstention rule - only classify if confident, otherwise defer to human

from __future__ import annotations

from dataclasses import dataclass, field

from app.analytical.tsi import is_confident_gum
from app.decisional.rule_engine import classify


@dataclass
class CriterionResult:
    criterion_key: str
    measurement: float
    threshold: float  # tau this measurement was actually compared against (audit trail)
    uncertainty: float  # k*sqrt(J Sigma J^T), same units as measurement -- what the decision now uses
    tsi: float  # legacy isotropic TSI, pixels -- audit/ablation only, NOT used to decide
    actual_rmse: float  # legacy blended RMSE, pixels -- audit/ablation only, NOT used to decide
    decision: str  # pass, fault, or defer
    label: str  # what the rule engine says
    # TODO: wire this up later so low visbility auto defers
    low_visibility_keypoints: list[int] = field(default_factory=list)


def evaluate_criterion(
    criterion_key: str,
    measurement: float,
    threshold: float,
    uncertainty: float,
    tsi: float,
    actual_rmse: float,
) -> CriterionResult:
    label = classify(criterion_key, measurement)

    # Confident iff the SAME uncertainty already reported to the user
    # doesn't reach the threshold -- see tsi.py's is_confident_gum docstring
    # for why this replaced the old is_confident_isotropic(actual_rmse, tsi)
    # check (that rule ignored the real per-keypoint covariance entirely).
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