"""Post-hoc per-criterion recalibration of the GUM expanded uncertainty on VAL.

    python -m training.recalibrate training/runs/v3_run/eval_val/predictions.npz \\
        --out training/runs/v3_run/calibration.json

s_c = conformal quantile of |y_hat - y_true| / U, so s_c * U covers ~95% of val cases.
Also reports the abstention-gate deferral rate and decided-case accuracy before/after.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from app.analytical.calibration import conformal_scale
from app.analytical.gum_propagation import assemble_block_covariance, combined_uncertainty, expanded_uncertainty
from app.analytical.jacobian import numerical_jacobian
from app.analytical.morphometrics import MORPHOMETRIC_FUNCTIONS
from app.decisional.abstention_gate import evaluate_criterion
from app.decisional.ibc_standards import CRITERION_THRESHOLDS
from app.decisional.rule_engine import classify

NOT_FAULT = ("Pass", "Ideal")


def gate_stats(key: str, y_hat: np.ndarray, y_true: np.ndarray, u: np.ndarray) -> dict:
    tau = CRITERION_THRESHOLDS[key]
    decided = correct = 0
    for yh, yt, ui in zip(y_hat, y_true, u):
        res = evaluate_criterion(key, float(yh), tau, float(ui), 0.0, 0.0)
        if res.decision == "Defer to Judge":
            continue
        decided += 1
        predicted_ok = res.decision == "Confident Pass"
        correct += predicted_ok == (classify(key, float(yt)) in NOT_FAULT)
    n = len(y_hat)
    return {
        "deferral_rate": (n - decided) / n,
        "n_decided": decided,
        "decided_accuracy": correct / decided if decided else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=str)
    parser.add_argument("--out", type=str, default="training/runs/v3_run/calibration.json")
    args = parser.parse_args()

    d = np.load(args.predictions)
    keep = d["vis_mask"].astype(bool).all(axis=1)
    mu, cov, gt = d["pred_mu"][keep], d["pred_cov"][keep], d["gt"][keep]

    out = {"date": date.today().isoformat(), "source": args.predictions, "coverage": 0.95, "criteria": {}}
    rows = []
    for key, fn in MORPHOMETRIC_FUNCTIONS.items():
        y_hat, y_true, u = [], [], []
        for m, c, g in zip(mu, cov, gt):
            x = m.reshape(-1)
            jac = numerical_jacobian(fn, x)
            u.append(expanded_uncertainty(combined_uncertainty(jac, assemble_block_covariance(c))))
            y_hat.append(fn(x))
            y_true.append(fn(g.reshape(-1)))
        y_hat, y_true, u = np.array(y_hat), np.array(y_true), np.array(u)
        err = np.abs(y_hat - y_true)
        s = conformal_scale(err, u)
        before, after = gate_stats(key, y_hat, y_true, u), gate_stats(key, y_hat, y_true, s * u)
        rows.append((key, len(err), (err <= u).mean(), (err <= s * u).mean(), s, u.mean(), (s * u).mean(), err.mean(), before, after))
        out["criteria"][key] = {"scale": s, "n": int(len(err))}

    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"{'criterion':<21}{'n':>4}{'PICP_b':>8}{'PICP_a':>8}{'s_c':>7}{'U_b':>9}{'U_a':>9}{'MAE':>9}{'defer_b':>9}{'defer_a':>9}{'acc_b':>7}{'acc_a':>7}")
    for key, n, pb, pa, s, ub, ua, mae, gb, ga in rows:
        print(f"{key:<21}{n:>4}{pb:>8.3f}{pa:>8.3f}{s:>7.2f}{ub:>9.4f}{ua:>9.4f}{mae:>9.4f}"
              f"{gb['deferral_rate']:>9.3f}{ga['deferral_rate']:>9.3f}{gb['decided_accuracy']:>7.3f}{ga['decided_accuracy']:>7.3f}")
    print("\ncriteria deferring >70% after calibration:",
          [r[0] for r in rows if r[9]["deferral_rate"] > 0.70] or "none")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
