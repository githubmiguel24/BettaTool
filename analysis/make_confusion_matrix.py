from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.analytical.jacobian import numerical_jacobian
from app.analytical.morphometrics import MORPHOMETRIC_FUNCTIONS
from app.analytical.tsi import compute_tsi, predicted_keypoint_sigma
from app.decisional.ibc_standards import CRITERION_THRESHOLDS
from app.decisional.rule_engine import classify
from training.mfld_full_comparison import criterion_keypoints

# "not Fault" per classify()'s possible labels: Pass (ratio criteria) or Ideal (caudal-spread-angle at exactly 180).
# Every other label (Slight/Major Fault, Disqualify, and the rare "Unclassified" gap between bands) counts as Fault.
# This mirrors the NOT_FAULT convention of training/mfld_full_comparison.py, i.e. exactly what the deployed system does.
NOT_FAULT_LABELS = ("Pass", "Ideal")


# --------------------------------------------------------------------------------------------------------- loading

@dataclass
class Predictions:
    image_ids: np.ndarray
    pred_mu: np.ndarray   # (N, 13, 2)
    pred_cov: np.ndarray  # (N, 13, 2, 2)
    gt: np.ndarray        # (N, 13, 2)
    vis_mask: np.ndarray  # (N, 13) bool


def load_predictions(path: str | Path) -> Predictions:
    d = np.load(path, allow_pickle=True)
    return Predictions(
        image_ids=np.asarray(d["image_ids"]),
        pred_mu=d["pred_mu"].astype(float),
        pred_cov=d["pred_cov"].astype(float),
        gt=d["gt"].astype(float),
        vis_mask=d["vis_mask"].astype(bool),
    )


def criterion_mask(p: Predictions, fn) -> np.ndarray:
    # images on which this criterion can be measured: every keypoint it depends on is labelled
    return p.vis_mask[:, criterion_keypoints(fn)].all(axis=1)


def subset_for_criterion(p: Predictions, fn) -> Predictions:
    keep = criterion_mask(p, fn)
    return Predictions(p.image_ids[keep], p.pred_mu[keep], p.pred_cov[keep], p.gt[keep], p.vis_mask[keep])


def load_calibration_scales(path: str | Path) -> dict[str, float]:
    from app.analytical.measurement_factor import load_tsi_scales
    return load_tsi_scales(path)


def check_split_list(image_ids: np.ndarray, split_json: str | Path) -> None:
    # optional cross-check against data/splits/test.json; never required, never fails the run
    p = Path(split_json)
    if not p.is_file():
        print(f"  (no split list at {p}; skipping the cross-check)")
        return
    listed = {str(x) for x in json.loads(p.read_text())}
    have = {str(x) for x in image_ids}
    if listed != have:
        print(f"  WARNING: predictions.npz image_ids differ from {p}: "
              f"{len(have - listed)} extra, {len(listed - have)} missing")
    else:
        print(f"  image_ids match {p} exactly ({len(listed)} images)")


# ---------------------------------------------------------------------------------------------- per-criterion gate

def compute_criterion_quantities(p: Predictions, fn, tau: float) -> dict[str, np.ndarray]:
    """y_hat, y_true, the TSI and the UNSCALED predicted keypoint uncertainty sigma_hat (before any calibration
    scale), one value per image, using the live app.analytical functions."""
    y_hat, y_true, tsi, sigma_hat = [], [], [], []
    for mu, cov, gt in zip(p.pred_mu, p.pred_cov, p.gt):
        x_hat, x_true = mu.reshape(-1), gt.reshape(-1)
        jac = numerical_jacobian(fn, x_hat)
        y_hat.append(fn(x_hat))
        y_true.append(fn(x_true))
        tsi.append(compute_tsi(y_hat[-1], tau, jac))
        sigma_hat.append(predicted_keypoint_sigma(jac, cov))
    return {"y_hat": np.array(y_hat), "y_true": np.array(y_true), "tsi": np.array(tsi), "sigma_hat": np.array(sigma_hat), "tau": tau}


def binary_label(criterion_key: str, value: float, caudal_rule: str = "ibc") -> str:
    """Pass/Fault. Default ("ibc") is exactly app.decisional.rule_engine.classify binarised for the gate's own
    bookkeeping — the real, deployed decisional logic in app/decisional/, read here, never modified.

    caudal_rule="ge180" is a TEMPORARY, analysis-only override for caudal-spread-angle, requested to get a
    non-degenerate Pass/Fault split for that criterion (the real IBC bands make every test fish some Fault band,
    so its matrix has 0 actual Pass). It redefines caudal-spread-angle ONLY as Pass if angle >= 180 else Fault.
    It does NOT touch app/decisional/ibc_standards.py or rule_engine.py, and every other criterion is unaffected."""
    if caudal_rule == "ge180" and criterion_key == "caudal-spread-angle":
        return "Pass" if value >= 180.0 else "Fault"
    return "Pass" if classify(criterion_key, value) in NOT_FAULT_LABELS else "Fault"


def gate_decisions(criterion_key: str, q: dict[str, np.ndarray], scale: float, caudal_rule: str = "ibc") -> tuple[np.ndarray, np.ndarray]:
    """Returns (actual, predicted) string arrays, predicted in {"Pass","Fault","Defer"}.
    Confident iff scale * sigma_hat < TSI (exactly app.analytical.tsi.is_confident)."""
    confident = scale * q["sigma_hat"] < q["tsi"]
    actual = np.array([binary_label(criterion_key, v, caudal_rule) for v in q["y_true"]])
    predicted = np.array([binary_label(criterion_key, v, caudal_rule) if c else "Defer" for v, c in zip(q["y_hat"], confident)])
    return actual, predicted


# ----------------------------------------------------------------------------------------------- matrices & metrics

def build_2x2(actual: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    # over DECIDED cases only (predicted != "Defer"); Fault = positive
    decided = predicted != "Defer"
    a, p = actual[decided], predicted[decided]
    tp = int(np.sum((a == "Fault") & (p == "Fault")))
    tn = int(np.sum((a == "Pass") & (p == "Pass")))
    fp = int(np.sum((a == "Pass") & (p == "Fault")))
    fn = int(np.sum((a == "Fault") & (p == "Pass")))
    n_decided = int(decided.sum())
    assert tp + fp + fn + tn == n_decided, f"{tp}+{fp}+{fn}+{tn} != {n_decided}"
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "n_decided": n_decided}


def build_2x3(actual: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    # rows actual(Pass,Fault) x columns predicted(Pass,Fault,Defer)
    out = {}
    for a_label in ("Pass", "Fault"):
        for p_label in ("Pass", "Fault", "Defer"):
            out[f"actual_{a_label}_pred_{p_label}"] = int(np.sum((actual == a_label) & (predicted == p_label)))
    return out


def sklearn_cross_check(actual: np.ndarray, predicted: np.ndarray, cm: dict[str, int]) -> str:
    # best-effort agreement check against sklearn; never required (spec: "sklearn is OK for checking")
    try:
        from sklearn.metrics import confusion_matrix as sk_cm
    except ImportError:
        return "sklearn not installed, skipped"
    decided = predicted != "Defer"
    sk = sk_cm(actual[decided], predicted[decided], labels=["Pass", "Fault"])
    tn, fp, fn, tp = int(sk[0, 0]), int(sk[0, 1]), int(sk[1, 0]), int(sk[1, 1])
    ok = (tn, fp, fn, tp) == (cm["TN"], cm["FP"], cm["FN"], cm["TP"])
    assert ok, f"sklearn disagrees with the manual count: sklearn={(tn, fp, fn, tp)} manual={(cm['TN'], cm['FP'], cm['FN'], cm['TP'])}"
    return "matches manual count"


def safe_div(num: float, den: float) -> float | str:
    return "n/a" if den == 0 else num / den


def fmt(v: float | str) -> str:
    return v if v == "n/a" else f"{v:.3f}"


def compute_metrics(cm: dict[str, int]) -> dict[str, float | str]:
    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    n = cm["n_decided"]
    accuracy = safe_div(tp + tn, n)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = "n/a" if precision == "n/a" or recall == "n/a" or (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------------------------------------- plotting

def plot_confusion(cm: dict[str, int], title: str, out_path: Path) -> None:
    matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])  # rows actual Pass/Fault, cols pred Pass/Fault
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Pass", "Pred Fault"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Pass", "Actual Fault"])
    thresh = matrix.max() / 2 if matrix.max() > 0 else 0
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i, j]}", ha="center", va="center",
                     color="white" if matrix[i, j] > thresh else "black", fontsize=12)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- driver

def run_mode(p: Predictions, mode: str, scales: dict[str, float], out_dir: Path, caudal_rule: str = "ibc") -> list[dict]:
    rows = []
    pooled_actual, pooled_predicted = [], []
    for key, fn in MORPHOMETRIC_FUNCTIONS.items():
        q = compute_criterion_quantities(subset_for_criterion(p, fn), fn, CRITERION_THRESHOLDS[key])
        scale = scales.get(key, 1.0) if mode == "recal" else 1.0
        actual, predicted = gate_decisions(key, q, scale, caudal_rule)
        pooled_actual.append(actual); pooled_predicted.append(predicted)

        cm = build_2x2(actual, predicted)
        cm3 = build_2x3(actual, predicted)
        metrics = compute_metrics(cm)
        sk_note = sklearn_cross_check(actual, predicted, cm)
        n_total = len(actual)
        n_fault, n_pass = int(np.sum(actual == "Fault")), int(np.sum(actual == "Pass"))
        low_support = n_fault < 5 or n_pass < 5

        row = {"mode": mode, "criterion": key, "n_total": n_total, "n_decided": cm["n_decided"],
               "n_deferred": n_total - cm["n_decided"], "coverage": cm["n_decided"] / n_total,
               "n_actual_fault": n_fault, "n_actual_pass": n_pass, "low_support_flag": low_support,
               **cm, **metrics, **cm3, "sklearn_check": sk_note}
        rows.append(row)
        plot_confusion(cm, f"{key}\n({mode}, N={n_total}, decided={cm['n_decided']})", out_dir / f"confusion_{key}_{mode}.png")

        warn = "  <-- LOW SUPPORT, metrics not meaningful" if low_support else ""
        print(f"  [{mode}] {key:<24} N={n_total:>3}  decided={cm['n_decided']:>3} ({row['coverage']*100:4.1f}%)  "
              f"TP={cm['TP']:>3} FP={cm['FP']:>3} FN={cm['FN']:>3} TN={cm['TN']:>3}  "
              f"acc={fmt(metrics['accuracy'])}  prec={fmt(metrics['precision'])}  "
              f"rec={fmt(metrics['recall'])}  f1={fmt(metrics['f1'])}  "
              f"actualFault={n_fault} actualPass={n_pass}{warn}")

    # pooled "overall": every (criterion, image) decision stacked as one independent sample
    actual_all = np.concatenate(pooled_actual)
    predicted_all = np.concatenate(pooled_predicted)
    cm = build_2x2(actual_all, predicted_all)
    cm3 = build_2x3(actual_all, predicted_all)
    metrics = compute_metrics(cm)
    sk_note = sklearn_cross_check(actual_all, predicted_all, cm)
    n_total = len(actual_all)
    n_fault, n_pass = int(np.sum(actual_all == "Fault")), int(np.sum(actual_all == "Pass"))
    row = {"mode": mode, "criterion": "overall", "n_total": n_total, "n_decided": cm["n_decided"],
           "n_deferred": n_total - cm["n_decided"], "coverage": cm["n_decided"] / n_total,
           "n_actual_fault": n_fault, "n_actual_pass": n_pass, "low_support_flag": n_fault < 5 or n_pass < 5,
           **cm, **metrics, **cm3, "sklearn_check": sk_note}
    rows.append(row)
    plot_confusion(cm, f"overall (pooled, {mode}, N={n_total})", out_dir / f"confusion_overall_{mode}.png")
    print(f"  [{mode}] {'overall (pooled)':<24} N={n_total:>3}  decided={cm['n_decided']:>3} ({row['coverage']*100:4.1f}%)  "
          f"TP={cm['TP']:>3} FP={cm['FP']:>3} FN={cm['FN']:>3} TN={cm['TN']:>3}  acc={metrics['accuracy']:.3f}  "
          f"prec={metrics['precision']:.3f}  rec={metrics['recall']:.3f}  f1={metrics['f1']:.3f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="training/runs/v3_run/eval_test/predictions.npz",
                     help="predictions.npz for the TEST split (pred_mu, pred_cov, gt, vis_mask, image_ids)")
    ap.add_argument("--calibration", default="training/configs/hrnet_w32.yaml",
                     help="config holding the measurement_factor block (per-criterion tsi_scale), fitted on VAL only (used in --recal mode)")
    ap.add_argument("--split-json", default="data/splits/test.json", help="optional cross-check; not required")
    ap.add_argument("--out-dir", default=None, help="default: confusion_matrices_out (or _caudal_ge180 with --caudal-rule ge180)")
    ap.add_argument("--caudal-rule", choices=["ibc", "ge180"], default="ibc",
                     help="ibc (default) = real deployed rule_engine.classify bands. "
                          "ge180 = TEMPORARY analysis-only override: caudal-spread-angle Pass iff angle>=180, else Fault. "
                          "Does not touch app/decisional/; every other criterion is unaffected.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--raw", action="store_const", dest="mode", const="raw")
    group.add_argument("--recal", action="store_const", dest="mode", const="recal")
    ap.set_defaults(mode="both")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("confusion_matrices_out" if args.caudal_rule == "ibc" else "confusion_matrices_out_caudal_ge180")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.caudal_rule == "ge180":
        print("=" * 78)
        print("TEMPORARY OVERRIDE ACTIVE: caudal-spread-angle Pass/Fault redefined as")
        print("  Pass  if angle >= 180 degrees")
        print("  Fault otherwise")
        print("This is NOT the deployed IBC rule (app/decisional/) and is for this analysis run only.")
        print(f"Outputs are written to {out_dir}/ to keep them separate from the official confusion_matrices_out/.")
        print("=" * 78 + "\n")

    print(f"Loading predictions from {args.predictions}")
    p_all = load_predictions(args.predictions)
    print(f"  {len(p_all.image_ids)} images in the file")
    check_split_list(p_all.image_ids, args.split_json)
    p = p_all
    for k, f in MORPHOMETRIC_FUNCTIONS.items():
        print(f"  {k:<24} scored on {int(criterion_mask(p, f).sum())} images (all keypoints it uses are labelled)")
    print()

    scales = load_calibration_scales(args.calibration) if Path(args.calibration).is_file() else {}
    modes = ["raw", "recal"] if args.mode == "both" else [args.mode]

    all_rows: list[dict] = []
    for mode in modes:
        print(f"=== mode: {mode} {'(sigma_hat < TSI, scale = 1.0)' if mode == 'raw' else '(tsi_scale_c * sigma_hat < TSI, VAL-fitted scales)'} ===")
        all_rows.extend(run_mode(p, mode, scales, out_dir, args.caudal_rule))
        print()

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "confusion_matrices.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows) and {len(modes) * (len(MORPHOMETRIC_FUNCTIONS) + 1)} PNGs to {out_dir}/")

    print("\n=== console summary ===")
    show_cols = ["mode", "criterion", "n_total", "n_decided", "coverage", "accuracy", "precision", "recall", "f1", "low_support_flag"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
