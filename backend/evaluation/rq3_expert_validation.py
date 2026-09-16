"""RQ3 — Cohen's Kappa and McNemar's test between the system's
Confident-Zone decisions and human expert scorecards. Populates Table 4.
"""

from __future__ import annotations

from sklearn.metrics import cohen_kappa_score


def mcnemar_chi_square(b: int, c: int) -> float:
    """chi^2 = (b - c)^2 / (b + c), per the thesis's Statistical Treatment section.

    Args:
        b: cases the system flagged Fault but the human Passed.
        c: cases the system Passed but the human flagged Fault.
    """
    if b + c == 0:
        return 0.0
    return ((b - c) ** 2) / (b + c)


def run(system_labels: list[str], human_labels: list[str]) -> dict:
    kappa = cohen_kappa_score(system_labels, human_labels)

    b = sum(s == "Fault" and h == "Pass" for s, h in zip(system_labels, human_labels))
    c = sum(s == "Pass" and h == "Fault" for s, h in zip(system_labels, human_labels))
    chi_square = mcnemar_chi_square(b, c)

    return {"cohens_kappa": kappa, "mcnemar_chi_square": chi_square, "b": b, "c": c}


if __name__ == "__main__":
    raise SystemExit("Load real Confident-Zone system labels + expert scorecards before running.")
