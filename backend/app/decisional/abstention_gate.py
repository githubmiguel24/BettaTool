# =============================================================================
# Program Title : abstention_gate.py - Selective Abstention Gate
# Programmers   : Trinidad, C. M. L.; Daanay, K. M. M.; Verceles, R. C. S.
# System Fit    : Betta Fish Quality Assessment Model > Decisional Tier.
#                 Called by app/pipeline.py once per criterion with the outputs
#                 of the Analytical Tier; uses rule_engine.py and tsi.py. Its
#                 CriterionResult objects go to the report and Final Assessment.
# Written       : September 12, 2026
# Revised       : October 6, 2026 (last saved);
#                 October 14, 2026 (header block and comments for documentation)
# Purpose       : Decides whether each IBC label can be trusted. A label is
#                 reported as Confident Pass / Confident Fault only when the
#                 predicted keypoint error is below the TSI; otherwise the
#                 criterion is deferred to a human judge.
# Data / Algo   : Data: dataclass CriterionResult (one per criterion).
#                 Algorithm: Selective Abstention Rule, confident if
#                 sigma_hat < TSI.
#                 Control: selection (if/elif/else).
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

from app.analytical.tsi import is_confident
from app.decisional.rule_engine import classify

# Decision flag used when the gate abstains.
DEFER_DECISION = "Defer to Judge"


@dataclass
class CriterionResult:
    """Outcome of one IBC criterion for one image."""

    criterion_key: str
    measurement: float
    threshold: float  # tau the measurement is compared against (kept for audit)
    uncertainty: float  # reported margin of error U(y); not used for the decision
    tsi: float  # Threshold Sensitivity Index in pixels
    sigma_hat: float  # calibrated predicted keypoint error in pixels
    decision: str  # Confident Pass, Confident Fault, or Defer to Judge
    label: str  # raw label from the rule engine


def evaluate_criterion(
    criterion_key: str,
    measurement: float,
    threshold: float,
    uncertainty: float,
    tsi: float,
    sigma_hat: float,
) -> CriterionResult:
    pass