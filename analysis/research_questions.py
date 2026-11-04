"""Measures the three research questions of the study on the TEST split.

    RQ1  Keypoint localization error: RMSE in pixels between predicted and ground-truth keypoints.
    RQ2  Fault classification performance of the proposed system: accuracy, precision, recall, F1 (and coverage).
    RQ3  Is the proposed system significantly different from the deterministic MFLD-Net baseline?  McNemar's test.

Run it from anywhere:

    python analysis/research_questions.py                 # all three questions
    python analysis/research_questions.py --rq 1 3        # only RQ1 and RQ3
    python analysis/research_questions.py --self-test     # check the McNemar code against hand-computed values

It reuses the project's own code (app.analytical, app.decisional, analysis.make_confusion_matrix, ...) so the numbers
are identical to the confusion matrices already produced by the other analysis scripts.  It reads:
    backend/training/runs/v3_run/eval_test/predictions.npz   our model on the test split
    backend/data/mfld_predictions/predictions_test.npz       MFLD-Net on the same test split
    backend/training/configs/hrnet_w32.yaml                  validation-fitted TSI scales (used in --mode recal)
and writes CSV/JSON files to research_question_results/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for _p in (str(ROOT), str(BACKEND)):  # lets `app.*`, `training.*` and `analysis.*` resolve from any working directory
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

ALPHA = 0.05            # significance level for RQ3
EXACT_BELOW = 25        # McNemar: exact binomial p-value when b + c < 25, chi-square (continuity corrected) otherwise
LOW_SUPPORT = 5         # fewer than this many actual Fault or Pass cases -> metrics are not meaningful
POOLED_RELIABLE = "POOLED excl. low-support"


# ==================================================================================================== RQ1: RMSE
def rq1_rmse(ours, mfld_mu: np.ndarray, n_boot: int, seed: int) -> list[dict]:
    """RMSE (pixels) of the predicted keypoints against the ground truth, per keypoint and overall.

    RMSE = sqrt(mean(d^2)) where d is the Euclidean distance between a predicted and a ground-truth keypoint,
    over the keypoints that are labelled (visible).  The 95% interval is a percentile bootstrap over images.

    Extra columns put the RMSE in context.  The predictions are in ORIGINAL-photo pixels, so a pixel error grows with the
    photo's resolution; `rmse_pct_body` (RMSE as a percentage of each fish's body length) removes that.  RMSE also
    weights large misses heavily, so the mean and median error are shown next to it.
    """
    from app.perception.keypoints import KEYPOINT_NAMES, Keypoint
    from training.metrics.localization import body_length, overall_rmse, per_keypoint_rmse, radial_errors

    blen = body_length(ours.gt, Keypoint.SNOUT_TIP, Keypoint.CAUDAL_PEDUNCLE_TOP, Keypoint.CAUDAL_PEDUNCLE_BOTTOM)
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for model, pred in (("proposed", ours.pred_mu), ("mfld_net", mfld_mu)):
        err = radial_errors(pred, ours.gt, ours.vis_mask)                    # (N, K), nan where not labelled
        rel = err / blen[:, None]
        per_kp = per_keypoint_rmse(pred, ours.gt, ours.vis_mask)
        for k, name in enumerate(KEYPOINT_NAMES):
            rows.append({"model": model, "keypoint": name, "n_labelled": int(ours.vis_mask[:, k].sum()),
                         "rmse_px": float(per_kp[k]), "mean_px": float(np.nanmean(err[:, k])),
                         "median_px": float(np.nanmedian(err[:, k])),
                         "rmse_pct_body": float(100 * np.sqrt(np.nanmean(rel[:, k] ** 2))), "ci_low": "", "ci_high": ""})
        sq = err ** 2                                                         # image-level bootstrap for the overall RMSE
        boots = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, len(sq), len(sq))
            boots[i] = np.sqrt(np.nanmean(sq[idx]))
        rows.append({"model": model, "keypoint": "ALL KEYPOINTS", "n_labelled": int(ours.vis_mask.sum()),
                     "rmse_px": overall_rmse(pred, ours.gt, ours.vis_mask), "mean_px": float(np.nanmean(err)),
                     "median_px": float(np.nanmedian(err)), "rmse_pct_body": float(100 * np.sqrt(np.nanmean(rel ** 2))),
                     "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5))})
    return rows


# ====================================================================== shared: per-criterion decisions
def criterion_decisions(ours, mfld_mu, mode: str, scales: dict, caudal_rule: str) -> dict[str, dict]:
    """For every IBC criterion, the Pass/Fault decision of each system on the images where the criterion can be
    scored, all in the same image order (so the systems are PAIRED).

        actual        ground truth label, from the validated keypoints through the IBC rule engine
        proposed      Pass / Fault / Defer   proposed system with the TSI deferral gate
        proposed_forced   Pass / Fault       the same keypoints with no deferral
        mfld_forced   Pass / Fault           MFLD-Net keypoints, no uncertainty, no deferral
    """
    from analysis.make_confusion_matrix import compute_criterion_quantities, criterion_mask, gate_decisions, subset_for_criterion
    from analysis.make_confusion_matrix_mfld import forced_decisions
    from app.analytical.morphometrics import MORPHOMETRIC_FUNCTIONS
    from app.decisional.ibc_standards import CRITERION_THRESHOLDS

    out: dict[str, dict] = {}
    for key, fn in MORPHOMETRIC_FUNCTIONS.items():
        keep = criterion_mask(ours, fn)
        y_true = np.array([fn(g.reshape(-1)) for g in ours.gt[keep]])
        y_ours = np.array([fn(m.reshape(-1)) for m in ours.pred_mu[keep]])
        y_mfld = np.array([fn(m.reshape(-1)) for m in mfld_mu[keep]])

        q = compute_criterion_quantities(subset_for_criterion(ours, fn), fn, CRITERION_THRESHOLDS[key])
        scale = scales.get(key, 1.0) if mode == "recal" else 1.0
        actual, proposed = gate_decisions(key, q, scale, caudal_rule)
        actual_f, proposed_forced = forced_decisions(key, y_ours, y_true, caudal_rule)
        actual_m, mfld_forced = forced_decisions(key, y_mfld, y_true, caudal_rule)
        assert (actual == actual_f).all() and (actual == actual_m).all(), "ground truth labels disagree between systems"
        out[key] = {"actual": actual, "proposed": proposed, "proposed_forced": proposed_forced, "mfld_forced": mfld_forced}
    return out


# ================================================================================ RQ2: accuracy/precision/recall/F1
def rq2_classification(decisions: dict[str, dict]) -> list[dict]:
    """Accuracy, precision, recall and F1 of the Pass/Fault decisions (Fault = positive class).

    The proposed system is scored on the cases it DECIDED; `coverage` is the share it decided (the rest are deferred to a
    judge).  The no-deferral and MFLD-Net rows are given for context: they decide every case.
    """
    from analysis.make_confusion_matrix import build_2x2, compute_metrics

    rows: list[dict] = []
    systems = (("proposed_with_deferral", "proposed"), ("proposed_forced", "proposed_forced"), ("mfld_net_forced", "mfld_forced"))
    pooled = {s: ([], []) for _, s in systems}
    for key, d in decisions.items():
        for label, s in systems:
            rows.append(_metric_row(label, key, d["actual"], d[s], build_2x2, compute_metrics))
            pooled[s][0].append(d["actual"]); pooled[s][1].append(d[s])
    for label, s in systems:
        rows.append(_metric_row(label, "ALL CRITERIA (pooled)", np.concatenate(pooled[s][0]), np.concatenate(pooled[s][1]),
                                build_2x2, compute_metrics))
    # A criterion with (almost) no Pass or no Fault cases is trivially "right" and inflates the pooled score, so the
    # pooled score is also given without those criteria.
    reliable = [k for k, d in decisions.items()
                if (d["actual"] == "Fault").sum() >= LOW_SUPPORT and (d["actual"] == "Pass").sum() >= LOW_SUPPORT]
    if reliable and len(reliable) < len(decisions):
        for label, s in systems:
            rows.append(_metric_row(label, POOLED_RELIABLE, np.concatenate([decisions[k]["actual"] for k in reliable]),
                                    np.concatenate([decisions[k][s] for k in reliable]), build_2x2, compute_metrics))
    return rows


def _metric_row(label, criterion, actual, predicted, build_2x2, compute_metrics) -> dict:
    cm = build_2x2(actual, predicted)
    m = compute_metrics(cm)
    n_total = len(actual)
    n_fault, n_pass = int(np.sum(actual == "Fault")), int(np.sum(actual == "Pass"))
    return {"system": label, "criterion": criterion, "n_total": n_total, "n_decided": cm["n_decided"],
            "coverage": cm["n_decided"] / n_total if n_total else float("nan"),
            "TP": cm["TP"], "FP": cm["FP"], "FN": cm["FN"], "TN": cm["TN"],
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "n_actual_fault": n_fault, "n_actual_pass": n_pass,
            "low_support": n_fault < LOW_SUPPORT or n_pass < LOW_SUPPORT}


# ======================================================================================== RQ3: McNemar's test
def mcnemar(b: int, c: int) -> dict:
    """McNemar's test on paired correct/incorrect outcomes.

    b = cases the baseline (MFLD-Net) got right and the proposed system got wrong
    c = cases the proposed system got right and the baseline got wrong
    H0: both systems have the same error rate, i.e. b and c are equally likely (each 50% of the discordant pairs).

    Returns the continuity-corrected chi-square statistic, both p-values, and the one used: the exact binomial test
    when b + c < 25 (small samples) and the chi-square approximation otherwise.  Only the standard library is used.
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_exact": 1.0, "p_chi2": 1.0, "p": 1.0, "method": "no discordant pairs"}
    chi2 = max(abs(b - c) - 1, 0) ** 2 / n
    p_chi2 = math.erfc(math.sqrt(chi2 / 2.0))                       # survival function of chi-square, 1 degree of freedom
    k = min(b, c)
    p_exact = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    use_exact = n < EXACT_BELOW
    return {"b": b, "c": c, "chi2": chi2, "p_exact": p_exact, "p_chi2": p_chi2,
            "p": p_exact if use_exact else p_chi2, "method": "exact binomial" if use_exact else "chi-square (continuity corrected)"}


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate across the criteria)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def rq3_mcnemar(decisions: dict[str, dict]) -> list[dict]:
    """Proposed system vs MFLD-Net with McNemar's test, two comparisons:

    A  "deferral"  proposed system (with the TSI deferral gate) vs MFLD-Net, on the cases the proposed system decided.
                   This is the comparison stated in the methodology.
    B  "forced"    proposed keypoints vs MFLD-Net keypoints with NO deferral, on all cases: isolates keypoint quality.

    One test per criterion (Holm-adjusted across the criteria) and one pooled test over all (criterion, image) decisions.
    The pooled test stacks decisions that are not fully independent (the same fish appears in every criterion), so read
    it as a summary and base conclusions on the per-criterion tests.
    """
    rows: list[dict] = []
    for comparison, ours_key in (("deferral", "proposed"), ("forced", "proposed_forced")):
        per_criterion = []
        pooled_b = pooled_c = pooled_n = 0
        for key, d in decisions.items():
            decided = d[ours_key] != "Defer"                                  # all True for the forced comparison
            ours_ok = d[ours_key][decided] == d["actual"][decided]
            mfld_ok = d["mfld_forced"][decided] == d["actual"][decided]
            b = int(np.sum(mfld_ok & ~ours_ok))                                # baseline right, proposed wrong
            c = int(np.sum(ours_ok & ~mfld_ok))                                # proposed right, baseline wrong
            per_criterion.append({"comparison": comparison, "criterion": key, "n_pairs": int(decided.sum()),
                                  "ours_correct": int(ours_ok.sum()), "mfld_correct": int(mfld_ok.sum()),
                                  **mcnemar(b, c)})
            pooled_b += b; pooled_c += c; pooled_n += int(decided.sum())
        for r, adj in zip(per_criterion, holm_adjust([r["p"] for r in per_criterion])):
            r["p_holm"] = adj
        rows.extend(per_criterion)
        pooled_ours = sum(r["ours_correct"] for r in per_criterion)
        pooled_mfld = sum(r["mfld_correct"] for r in per_criterion)
        rows.append({"comparison": comparison, "criterion": "ALL CRITERIA (pooled)", "n_pairs": pooled_n,
                     "ours_correct": pooled_ours, "mfld_correct": pooled_mfld, **mcnemar(pooled_b, pooled_c), "p_holm": ""})
    for r in rows:
        p = r["p_holm"] if r["p_holm"] != "" else r["p"]
        if r["b"] + r["c"] == 0:
            r["conclusion"] = "no difference (identical decisions)"
        elif p < ALPHA:
            r["conclusion"] = "significant: proposed better" if r["c"] > r["b"] else "significant: MFLD-Net better"
        else:
            r["conclusion"] = "not significant"
    return rows


def self_test() -> None:
    """Check mcnemar() and holm_adjust() against values worked out by hand (and scipy, when it is installed)."""
    # b=5, c=15: n=20, P(X<=5 | n=20, p=.5) = 21700/1048576 = 0.020695, two-sided = 0.041390
    r = mcnemar(5, 15)
    assert abs(r["p_exact"] - 0.04139) < 1e-4, r
    assert abs(r["chi2"] - (9 ** 2) / 20) < 1e-12, r
    assert mcnemar(7, 7)["p_exact"] == 1.0 and mcnemar(7, 7)["chi2"] == 0.0
    assert mcnemar(0, 0)["p"] == 1.0
    assert mcnemar(0, 10)["p_exact"] < 0.005                                   # 2 / 2**10 = 0.00195
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    try:
        from scipy import stats
        for b, c in ((5, 15), (3, 9), (40, 22), (100, 130), (1, 0)):
            n = b + c
            assert abs(mcnemar(b, c)["p_exact"] - stats.binomtest(min(b, c), n, 0.5).pvalue) < 1e-9, (b, c)
            chi2 = max(abs(b - c) - 1, 0) ** 2 / n
            assert abs(mcnemar(b, c)["p_chi2"] - stats.chi2.sf(chi2, 1)) < 1e-9, (b, c)
        print("self-test passed (hand-computed values and scipy agree)")
    except ImportError:
        print("self-test passed (hand-computed values; scipy not installed, so no cross-check)")


# ============================================================================================== output helpers
def fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else ""
    if isinstance(v, (float, np.floating)):
        return "n/a" if math.isnan(v) else (f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}")
    return str(v)


def fmt_p(v) -> str:
    return "" if v == "" else ("<0.0001" if v < 0.0001 else f"{v:.4f}")


def print_table(rows: list[dict], columns: list[str]) -> None:
    cell = lambda r, c: fmt_p(r[c]) if c in ("p", "p_holm") else fmt(r[c])  # noqa: E731
    w = {c: max(len(c), *(len(cell(r, c)) for r in rows)) for c in columns}
    print("  " + "  ".join(c.ljust(w[c]) for c in columns))
    print("  " + "  ".join("-" * w[c] for c in columns))
    for r in rows:
        print("  " + "  ".join(cell(r, c).ljust(w[c]) for c in columns))


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) if isinstance(v, float) else v for k, v in r.items()})


# ===================================================================================================== driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rq", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3], help="which questions to measure")
    ap.add_argument("--predictions", default=str(BACKEND / "training/runs/v3_run/eval_test/predictions.npz"),
                    help="predictions.npz of the proposed model on the TEST split")
    ap.add_argument("--calibration", default=str(BACKEND / "training/configs/hrnet_w32.yaml"),
                    help="config with the measurement_factor block (validation-fitted tsi_scale per criterion)")
    ap.add_argument("--mode", choices=["recal", "raw"], default="recal",
                    help="recal (default) = the deployed gate, tsi_scale * sigma_hat < TSI;  raw = sigma_hat < TSI (scale 1.0)")
    ap.add_argument("--caudal-rule", choices=["ibc", "ge180"], default="ibc",
                    help="ibc (default) = the deployed IBC bands; ge180 = analysis-only override, caudal angle Pass iff >= 180")
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples for the RMSE interval (RQ1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "research_question_results"))
    ap.add_argument("--self-test", action="store_true", help="only verify the McNemar implementation, then exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    from analysis.make_confusion_matrix import load_calibration_scales, load_predictions
    from analysis.make_confusion_matrix_mfld import load_mfld_aligned

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ours = load_predictions(args.predictions)
    mfld_mu = load_mfld_aligned(ours)
    scales = load_calibration_scales(args.calibration) if Path(args.calibration).is_file() else {}
    print(f"Test split: {len(ours.image_ids)} images; MFLD-Net predictions aligned to the same image ids.")
    print(f"Gate mode: {args.mode}" + (" (validation-fitted TSI scales)" if args.mode == "recal" else " (scale = 1.0)")
          + (f" | caudal rule: {args.caudal_rule}" if args.caudal_rule != "ibc" else ""))
    summary: dict = {"n_test_images": int(len(ours.image_ids)), "mode": args.mode, "caudal_rule": args.caudal_rule}

    if 1 in args.rq:
        print("\n" + "=" * 100 + "\nRQ1  Keypoint localization error (RMSE, pixels)\n" + "=" * 100)
        rows = rq1_rmse(ours, mfld_mu, args.bootstrap, args.seed)
        proposed = [r for r in rows if r["model"] == "proposed"]
        print_table([{**r, "ci95": (f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]" if r["ci_low"] != "" else "")} for r in proposed],
                    ["keypoint", "n_labelled", "rmse_px", "mean_px", "median_px", "rmse_pct_body", "ci95"])
        overall = next(r for r in proposed if r["keypoint"] == "ALL KEYPOINTS")
        base = next(r for r in rows if r["model"] == "mfld_net" and r["keypoint"] == "ALL KEYPOINTS")
        print(f"\n  ANSWER RQ1: the proposed model's overall RMSE is {overall['rmse_px']:.2f} px "
              f"(95% CI {overall['ci_low']:.2f}-{overall['ci_high']:.2f}); MFLD-Net: {base['rmse_px']:.2f} px.")
        print(f"  Context: mean error {overall['mean_px']:.2f} px, median {overall['median_px']:.2f} px, "
              f"RMSE = {overall['rmse_pct_body']:.1f}% of body length (MFLD-Net {base['rmse_pct_body']:.1f}%).")
        print(f"  Pixels are in the original photos (ground-truth coordinates reach x={ours.gt[..., 0].max():.0f}, "
              f"y={ours.gt[..., 1].max():.0f}), so pixel errors depend on photo resolution; the %-of-body-length figure does not.")
        write_csv(out_dir / "rq1_rmse.csv", rows)
        summary["rq1"] = {"proposed_rmse_px": overall["rmse_px"], "ci95": [overall["ci_low"], overall["ci_high"]],
                          "proposed_mean_px": overall["mean_px"], "proposed_median_px": overall["median_px"],
                          "proposed_rmse_pct_body": overall["rmse_pct_body"], "mfld_net_rmse_px": base["rmse_px"]}

    decisions = None
    if 2 in args.rq or 3 in args.rq:
        decisions = criterion_decisions(ours, mfld_mu, args.mode, scales, args.caudal_rule)

    if 2 in args.rq:
        print("\n" + "=" * 100 + "\nRQ2  Fault classification performance (Fault = positive class)\n" + "=" * 100)
        rows = rq2_classification(decisions)
        print_table(rows, ["system", "criterion", "n_total", "n_decided", "coverage", "TP", "FP", "FN", "TN",
                           "accuracy", "precision", "recall", "f1", "low_support"])
        print("  low_support = fewer than 5 actual Fault or Pass cases: that row's metrics are not meaningful.")
        low = [r["criterion"] for r in rows if r["system"] == "proposed_with_deferral" and r["low_support"]
               and not r["criterion"].startswith(("ALL", "POOLED"))]
        if low:
            print(f"  Low-support criteria here: {', '.join(low)}.  '{POOLED_RELIABLE}' leaves them out, because a criterion with")
            print("  (almost) no Pass or no Fault cases is trivially 'right' and inflates the pooled score.")
        write_csv(out_dir / "rq2_classification.csv", rows)
        summary["rq2"] = {}
        for crit, name in (("ALL CRITERIA (pooled)", "pooled"), (POOLED_RELIABLE, "pooled_excl_low_support")):
            p = next((r for r in rows if r["system"] == "proposed_with_deferral" and r["criterion"] == crit), None)
            if p is None:
                continue
            print(f"\n  ANSWER RQ2 ({name.replace('_', ' ')}, decided cases, coverage {p['coverage'] * 100:.1f}%): "
                  f"accuracy {fmt(p['accuracy'])}, precision {fmt(p['precision'])}, recall {fmt(p['recall'])}, F1 {fmt(p['f1'])}.")
            summary["rq2"][name] = {k: (v if not isinstance(v, float) else round(v, 6)) for k, v in p.items()
                                    if k in ("n_total", "n_decided", "coverage", "accuracy", "precision", "recall", "f1")}

    if 3 in args.rq:
        print("\n" + "=" * 100 + f"\nRQ3  Proposed system vs MFLD-Net: McNemar's test (alpha = {ALPHA})\n" + "=" * 100)
        rows = rq3_mcnemar(decisions)
        for comparison, title in (("deferral", "A. With the deferral gate, on the cases the proposed system decided"),
                                  ("forced", "B. No deferral (keypoint quality only), on all cases")):
            print(f"\n  {title}")
            print_table([r for r in rows if r["comparison"] == comparison],
                        ["criterion", "n_pairs", "ours_correct", "mfld_correct", "b", "c", "chi2", "p", "method", "p_holm", "conclusion"])
        print("\n  b = MFLD-Net right and proposed wrong;  c = proposed right and MFLD-Net wrong.")
        print("  Read A with care: the proposed system chooses which cases to decide (those where its keypoints are reliable),")
        print("  so A favours it by construction.  B compares the two sets of keypoints on exactly the same cases: the like-for-like test.")
        write_csv(out_dir / "rq3_mcnemar.csv", rows)
        pooled = next(r for r in rows if r["comparison"] == "deferral" and r["criterion"].startswith("ALL"))
        print(f"\n  ANSWER RQ3 (pooled, deferral comparison): b={pooled['b']}, c={pooled['c']}, p={pooled['p']:.4g} -> "
              f"{pooled['conclusion']}.")
        summary["rq3"] = {r["comparison"] + ":" + r["criterion"]: {k: r[k] for k in ("b", "c", "p", "p_holm", "conclusion")}
                          for r in rows}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote the tables to {out_dir}/")


if __name__ == "__main__":
    main()
