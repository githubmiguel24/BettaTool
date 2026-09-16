"""RQ2 — Accuracy/Precision/Recall/F1 comparison across three models:
MFLD-Net (deterministic baseline), the proposed pipeline with forced binary
classification, and the proposed pipeline with the TSI deferral gate.

Populates Table 3 (Appendix 1.2). "Fault" is treated as the positive class.
"""

from __future__ import annotations

from sklearn.metrics import precision_recall_fscore_support


def score_model(predictions: list[str], ground_truth: list[str]) -> dict:
    """predictions/ground_truth are lists of "Pass"/"Fault" labels, one per
    (image, criterion) pair the model did NOT defer on."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, pos_label="Fault", average="binary"
    )
    accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / len(ground_truth)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}


def coverage(total: int, deferred: int) -> float:
    """Fraction of specimens the model was willing to classify (1.0 for
    MFLD-Net and forced-binary; < 1.0 for the TSI-gated pipeline)."""
    return (total - deferred) / total


if __name__ == "__main__":
    raise SystemExit("Load real model outputs + expert ground truth before running.")
