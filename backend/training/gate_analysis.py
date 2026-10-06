"""Gate analysis on a split with a FROZEN calibration.json (nothing is refit here, except the
constant-margin gate, which is fitted on VAL only and reused unchanged on any other split).

    python -m training.gate_analysis --val  training/runs/v3_run/eval_val/predictions.npz \\
                                     --test training/runs/v3_run/eval_test/predictions.npz

Steps per split: (1) U-vs-error information + learned vs constant-margin gate at equal deferral,
(2) base rates / bands, (3) forced vs gated decisions, plus PICP before/after scaling.
Fault is the positive class. Ground truth = rule engine on the true measurement.
Precision/recall of the gated decisions are computed over DECIDED cases only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from app.analytical.gum_propagation import assemble_block_covariance, combined_uncertainty, expanded_uncertainty
from app.analytical.jacobian import numerical_jacobian
from app.analytical.morphometrics import MORPHOMETRIC_FUNCTIONS
from app.decisional.ibc_standards import CAUDAL_SPREAD_ANGLE_BANDS, CRITERION_THRESHOLDS
from app.decisional.rule_engine import classify

NOT_FAULT = ("Pass", "Ideal")


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    # 95% Wilson score interval for a proportion
    if n == 0:
        return float("nan"), float("nan")
    p = hits / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def compute(path: str) -> dict[str, dict[str, np.ndarray]]:
    d = np.load(path)
    keep = d["vis_mask"].astype(bool).all(axis=1)
    out = {}
    for key, fn in MORPHOMETRIC_FUNCTIONS.items():
        yh, yt, u = [], [], []
        for m, c, g in zip(d["pred_mu"][keep], d["pred_cov"][keep], d["gt"][keep]):
            x = m.reshape(-1)
            u.append(expanded_uncertainty(combined_uncertainty(numerical_jacobian(fn, x), assemble_block_covariance(c))))
            yh.append(fn(x))
            yt.append(fn(g.reshape(-1)))
        out[key] = {"y_hat": np.array(yh), "y_true": np.array(yt), "u": np.array(u)}
    return out


def is_fault(key: str, values: np.ndarray) -> np.ndarray:
    return np.array([classify(key, float(v)) not in NOT_FAULT for v in values])


def prf(pred_fault: np.ndarray, true_fault: np.ndarray) -> dict:
    n = len(pred_fault)
    tp = int((pred_fault & true_fault).sum())
    fp = int((pred_fault & ~true_fault).sum())
    fn = int((~pred_fault & true_fault).sum())
    return {
        "n": n,
        "acc": float((pred_fault == true_fault).mean()) if n else float("nan"),
        "prec": tp / (tp + fp) if tp + fp else float("nan"),
        "rec": tp / (tp + fn) if tp + fn else float("nan"),
    }


def gated(key: str, d: dict, margin: np.ndarray | float) -> dict:
    # decide only if |y_hat - tau| > margin (same rule as is_confident_gum); ground truth = rule engine on y_true
    tau = CRITERION_THRESHOLDS[key]
    decided = np.abs(d["y_hat"] - tau) > margin
    res = prf(is_fault(key, d["y_hat"])[decided], is_fault(key, d["y_true"])[decided])
    res["defer"] = 1 - decided.mean()
    return res


def forced(key: str, d: dict) -> dict:
    return prf(is_fault(key, d["y_hat"]), is_fault(key, d["y_true"]))


def fit_margin(key: str, d: dict, defer_rate: float) -> float:
    # constant margin deferring the same share of val cases as the calibrated gate
    dist = np.abs(d["y_hat"] - CRITERION_THRESHOLDS[key])
    return float(np.quantile(dist, defer_rate, method="higher")) if defer_rate > 0 else -1.0


def band_name(key: str, v: float) -> str:
    if key != "caudal-spread-angle":
        return classify(key, v)
    for band in CAUDAL_SPREAD_ANGLE_BANDS:
        if (band.low is None or v >= band.low) and (band.high is None or v <= band.high):
            side = "" if band.label == "Ideal" else (" (>180)" if (band.low or 0) >= 180 else " (<180)")
            return band.label + side
    return "Unclassified (gap)"


def f(x: float, p: int = 3) -> str:
    return "  n/a" if x != x else f"{x:.{p}f}"


def report(split: str, data: dict, scales: dict, margins: dict | None) -> dict:
    fit = margins is None
    margins = {} if fit else margins
    print(f"\n{'=' * 30} SPLIT: {split.upper()} {'=' * 30}")

    print(f"\n[PICP before/after scaling, mean U, deferral]  ({split})")
    print(f"{'criterion':<21}{'n':>4}{'PICP_b':>8}{'CI_b':>16}{'PICP_a':>8}{'CI_a':>16}{'U_b':>9}{'U_a':>9}{'defer_a':>9}")
    gates = {}
    for key, d in data.items():
        s, err, n = scales[key], np.abs(d["y_hat"] - d["y_true"]), len(d["y_hat"])
        hb, ha = int((err <= d["u"]).sum()), int((err <= s * d["u"]).sum())
        (lb, ub), (la, ua) = wilson_interval(hb, n), wilson_interval(ha, n)
        g = gated(key, d, s * d["u"])
        gates[key] = g
        print(f"{key:<21}{n:>4}{hb / n:>8.3f}{f'[{lb:.2f},{ub:.2f}]':>16}{ha / n:>8.3f}{f'[{la:.2f},{ua:.2f}]':>16}"
              f"{d['u'].mean():>9.4f}{(s * d['u']).mean():>9.4f}{g['defer']:>9.3f}")

    print(f"\n[STEP 1: information check, learned U gate vs constant-margin gate at equal deferral]  ({split})")
    print(f"{'criterion':<21}{'spearman(U,err)':>16}{'defer':>7}{'margin':>9} |{'acc L':>7}{'prec L':>8}{'rec L':>7} |{'acc C':>7}{'prec C':>8}{'rec C':>7}  (L=learned, C=constant)")
    s1 = {}
    for key, d in data.items():
        rho = spearmanr(d["u"], np.abs(d["y_hat"] - d["y_true"]))[0]
        L = gates[key]
        if fit:
            margins[key] = fit_margin(key, d, L["defer"])
        C = gated(key, d, margins[key])
        s1[key] = (rho, L, C)
        print(f"{key:<21}{rho:>16.3f}{L['defer']:>7.3f}{margins[key]:>9.4f} |{f(L['acc']):>7}{f(L['prec']):>8}{f(L['rec']):>7} |"
              f"{f(C['acc']):>7}{f(C['prec']):>8}{f(C['rec']):>7}   (C defer {C['defer']:.3f})")

    print(f"\n[STEP 2: base rates by rule-engine label of the TRUE measurement]  ({split})")
    for key, d in data.items():
        bands = Counter(band_name(key, v) for v in d["y_true"])
        tf = is_fault(key, d["y_true"])
        print(f"{key:<21} Pass/Ideal={int((~tf).sum()):>3}  Fault={int(tf.sum()):>3}  bands: " +
              ", ".join(f"{k}={v}" for k, v in sorted(bands.items())))
    ang = data["caudal-spread-angle"]["y_true"]
    print(f"caudal-spread-angle labelled Ideal (exactly 180): {sum(classify('caudal-spread-angle', float(v)) == 'Ideal' for v in ang)} of {len(ang)}"
          f"; true angle range [{ang.min():.1f}, {ang.max():.1f}], median {np.median(ang):.1f}")

    print(f"\n[STEP 3: forced (no deferral) vs gated decisions, Fault = positive]  ({split})")
    print(f"{'criterion':<21}{'n':>4}{'FaultRate':>10} |{'acc F':>7}{'prec F':>8}{'rec F':>7} |{'defer':>7}{'acc G':>7}{'prec G':>8}{'rec G':>7}  (F=forced, G=gated)")
    for key, d in data.items():
        F, G = forced(key, d), gates[key]
        fr = is_fault(key, d["y_true"]).mean()
        print(f"{key:<21}{F['n']:>4}{fr:>10.3f} |{f(F['acc']):>7}{f(F['prec']):>8}{f(F['rec']):>7} |"
              f"{G['defer']:>7.3f}{f(G['acc']):>7}{f(G['prec']):>8}{f(G['rec']):>7}")
    return margins


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--calibration", default="training/runs/v3_run/calibration.json")
    ap.add_argument("--margins-out", default="training/runs/v3_run/constant_margins.json")
    args = ap.parse_args()

    scales = {k: v["scale"] for k, v in json.loads(Path(args.calibration).read_text())["criteria"].items()}
    margins = report("val", compute(args.val), scales, None)
    Path(args.margins_out).write_text(json.dumps(margins, indent=2))
    if args.test:
        report("test", compute(args.test), scales, margins)


if __name__ == "__main__":
    main()
