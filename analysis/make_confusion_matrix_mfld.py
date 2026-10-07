"""Confusion matrices for MFLD-Net's Pass/Fault decisions on the TEST split, for direct comparison with
make_confusion_matrix.py's output for our system. Read-only: loads MFLD-Net's own prediction dump
(mfld-net/predictions_test.npz) and our ground truth, retrains and changes nothing.

    cd backend
    PYTHONPATH=.. python -m analysis.make_confusion_matrix_mfld

MFLD-Net has no covariance/uncertainty head, so it has no abstention gate: every criterion assessment is a
FORCED decision (Pass or Fault, never Defer). That is not directly comparable to our deployed-gate numbers
(which defer about 28% of cases), so this script reports three rows per criterion, all on the identical
per-criterion image set used by make_confusion_matrix.py (every test image where the criterion's own keypoints are labelled):

    mfld_forced   MFLD-Net,  no gate (it has none)              <- MFLD-Net's real-world behaviour
    ours_forced   our model, gate disabled (apples-to-apples with MFLD-Net's lack of a gate)
    ours_recal    our model, the deployed TSI gate, tsi_scale * sigma_hat < TSI (defers when uncertain)   <- our real-world behaviour

Ground truth and the keypoint order come from training/mfld_full_comparison.py's verified mapping (MFLD-Net's
predictions_*.npz columns are already in OUR keypoint order; keypoint_names has two columns mislabelled but the
data columns are not swapped — see that module's load_mfld() docstring for the verification).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.analytical.morphometrics import MORPHOMETRIC_FUNCTIONS
from app.decisional.ibc_standards import CRITERION_THRESHOLDS
from analysis.make_confusion_matrix import (
    Predictions, binary_label, build_2x2, build_2x3, compute_metrics,
    criterion_mask, fmt, load_predictions, plot_confusion, sklearn_cross_check, subset_for_criterion,
)

ROOT = Path(__file__).resolve().parents[1]
OURS_PREDICTIONS = "training/runs/v3_run/eval_test/predictions.npz"
MFLD_PREDICTIONS = ROOT / "mfld-net" / "predictions_test.npz"


def load_mfld_aligned(ours: Predictions) -> np.ndarray:
    """MFLD-Net's predicted keypoints (N, 13, 2), re-ordered to match `ours.image_ids` exactly."""
    d = np.load(MFLD_PREDICTIONS, allow_pickle=True)
    mfld_ids = [str(x) for x in d["image_ids"]]
    mfld_mu = d["pred_mu"].astype(float)
    index = {iid: i for i, iid in enumerate(mfld_ids)}
    missing = [iid for iid in ours.image_ids if str(iid) not in index]
    if missing:
        raise SystemExit(f"{len(missing)} test image(s) in our predictions have no MFLD-Net prediction, e.g. {missing[:5]}")
    return np.stack([mfld_mu[index[str(iid)]] for iid in ours.image_ids])


def forced_decisions(criterion_key: str, y_hat: np.ndarray, y_true: np.ndarray, caudal_rule: str = "ibc") -> tuple[np.ndarray, np.ndarray]:
    # no uncertainty -> no deferral; every case is decided
    actual = np.array([binary_label(criterion_key, v, caudal_rule) for v in y_true])
    predicted = np.array([binary_label(criterion_key, v, caudal_rule) for v in y_hat])
    return actual, predicted


def run_row(model: str, criterion_key: str, actual: np.ndarray, predicted: np.ndarray, out_dir: Path) -> dict:
    cm = build_2x2(actual, predicted)
    cm3 = build_2x3(actual, predicted)
    metrics = compute_metrics(cm)
    sk_note = sklearn_cross_check(actual, predicted, cm)
    n_total = len(actual)
    n_fault, n_pass = int(np.sum(actual == "Fault")), int(np.sum(actual == "Pass"))
    low_support = n_fault < 5 or n_pass < 5
    row = {"model": model, "criterion": criterion_key, "n_total": n_total, "n_decided": cm["n_decided"],
           "n_deferred": n_total - cm["n_decided"], "coverage": cm["n_decided"] / n_total,
           "n_actual_fault": n_fault, "n_actual_pass": n_pass, "low_support_flag": low_support,
           **cm, **metrics, **cm3, "sklearn_check": sk_note}
    warn = "  <-- LOW SUPPORT, metrics not meaningful" if low_support else ""
    print(f"  [{model:<11}] {criterion_key:<24} N={n_total:>3}  decided={cm['n_decided']:>3} ({row['coverage']*100:5.1f}%)  "
          f"TP={cm['TP']:>3} FP={cm['FP']:>3} FN={cm['FN']:>3} TN={cm['TN']:>3}  "
          f"acc={fmt(metrics['accuracy'])}  prec={fmt(metrics['precision'])}  "
          f"rec={fmt(metrics['recall'])}  f1={fmt(metrics['f1'])}  "
          f"actualFault={n_fault} actualPass={n_pass}{warn}")
    plot_confusion(cm, f"{criterion_key}\n({model}, N={n_total}, decided={cm['n_decided']})",
                    out_dir / f"confusion_{criterion_key}_{model}.png")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None, help="default: confusion_matrices_out (or _caudal_ge180 with --caudal-rule ge180)")
    ap.add_argument("--caudal-rule", choices=["ibc", "ge180"], default="ibc",
                     help="ibc (default) = real deployed rule_engine.classify bands. "
                          "ge180 = TEMPORARY analysis-only override: caudal-spread-angle Pass iff angle>=180, else Fault.")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path("confusion_matrices_out" if args.caudal_rule == "ibc" else "confusion_matrices_out_caudal_ge180")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.caudal_rule == "ge180":
        print("=" * 78)
        print("TEMPORARY OVERRIDE ACTIVE: caudal-spread-angle Pass/Fault redefined as")
        print("  Pass  if angle >= 180 degrees   /   Fault otherwise")
        print("This is NOT the deployed IBC rule and is for this analysis run only.")
        print(f"Outputs are written to {out_dir}/ to keep them separate from the official confusion_matrices_out/.")
        print("=" * 78 + "\n")

    print(f"Loading our predictions from {OURS_PREDICTIONS}")
    ours_all = load_predictions(OURS_PREDICTIONS)
    ours = ours_all
    print(f"  {len(ours.image_ids)} test images; each criterion uses those where its own keypoints are labelled")

    print(f"Loading MFLD-Net predictions from {MFLD_PREDICTIONS}")
    mfld_mu = load_mfld_aligned(ours)
    print(f"  aligned {len(mfld_mu)} MFLD-Net predictions to our image_ids\n")

    rows: list[dict] = []
    pooled = {"mfld_forced": ([], []), "ours_forced": ([], []), "ours_recal": ([], [])}

    # ours_recal: reuse the deployed gate exactly as make_confusion_matrix.py does (measurement-factor scaled mode)
    from analysis.make_confusion_matrix import compute_criterion_quantities, gate_decisions, load_calibration_scales
    scales = load_calibration_scales("training/configs/hrnet_w32.yaml")

    for key, fn in MORPHOMETRIC_FUNCTIONS.items():
        tau = CRITERION_THRESHOLDS[key]
        keep = criterion_mask(ours, fn)
        y_hat_mfld = np.array([fn(mu.reshape(-1)) for mu in mfld_mu[keep]])
        y_hat_ours = np.array([fn(mu.reshape(-1)) for mu in ours.pred_mu[keep]])
        y_true = np.array([fn(gt.reshape(-1)) for gt in ours.gt[keep]])

        a, p = forced_decisions(key, y_hat_mfld, y_true, args.caudal_rule)
        rows.append(run_row("mfld_forced", key, a, p, out_dir)); pooled["mfld_forced"][0].append(a); pooled["mfld_forced"][1].append(p)

        a, p = forced_decisions(key, y_hat_ours, y_true, args.caudal_rule)
        rows.append(run_row("ours_forced", key, a, p, out_dir)); pooled["ours_forced"][0].append(a); pooled["ours_forced"][1].append(p)

        q = compute_criterion_quantities(subset_for_criterion(ours, fn), fn, tau)
        a, p = gate_decisions(key, q, scales.get(key, 1.0), args.caudal_rule)
        rows.append(run_row("ours_recal", key, a, p, out_dir)); pooled["ours_recal"][0].append(a); pooled["ours_recal"][1].append(p)
        print()

    for model, (a_list, p_list) in pooled.items():
        a, p = np.concatenate(a_list), np.concatenate(p_list)
        rows.append(run_row(model, "overall", a, p, out_dir))

    df = pd.DataFrame(rows)
    csv_path = out_dir / "confusion_matrices_mfld_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path} ({len(df)} rows) and {len(rows)} PNGs to {out_dir}/\n")

    print("=== console summary: overall (pooled across the 5 criteria) ===")
    overall = df[df.criterion == "overall"][["model", "n_decided", "coverage", "accuracy", "precision", "recall", "f1"]]
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
