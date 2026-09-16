"""Selective Abstention Rule (Decisional Tier, Fig. 6).

Gates every rule-engine decision behind the Threshold Sensitivity Index:
if the system's actual keypoint RMSE exceeds the TSI for a criterion, the
propagated uncertainty is wide enough to straddle the IBC boundary and the
specimen is deferred to a human judge instead of classified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.analytical.tsi import is_confident
from app.decisional.rule_engine import classify


@dataclass
class CriterionResult:
    criterion_key: str
    measurement: float
    uncertainty: float  # expanded uncertainty U(y), same units as measurement
    tsi: float  # pixels
    actual_rmse: float  # pixels
    decision: str  # "Confident Pass" | "Confident Fault" | "Defer to Judge"
    label: str  # underlying rule-engine label (e.g. "Slight Fault")
    # Indices (into app.perception.keypoints.Keypoint) of landmarks the
    # perceptual tier's visibility head reports below
    # `AssessmentPipeline.visibility_threshold` for this image. Populated by
    # `app/pipeline.py` when the model provides a visibility head (Build
    # Prompt v2 §4.3); NOT yet consulted by `evaluate_criterion` below — see
    # app/pipeline.py's module docstring, "Known limitations" — so a
    # criterion computed from a low-visibility landmark is not automatically
    # deferred yet. Wiring that in is flagged follow-up work, not silently
    # skipped.
    low_visibility_keypoints: list[int] = field(default_factory=list)


def evaluate_criterion(
    criterion_key: str,
    measurement: float,
    uncertainty: float,
    tsi: float,
    actual_rmse: float,
) -> CriterionResult:
    label = classify(criterion_key, measurement)

    if not is_confident(actual_rmse, tsi):
        decision = "Defer to Judge"
    else:
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
