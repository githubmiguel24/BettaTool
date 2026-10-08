from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from app.analytical.measurement_factor import load_tsi_scales
from app.analytical.jacobian import numerical_jacobian
from app.analytical.morphometrics import MORPHOMETRIC_FUNCTIONS
from app.analytical.tsi import compute_tsi, is_confident, predicted_keypoint_sigma
from app.decisional.ibc_standards import CRITERION_THRESHOLDS
from app.perception.keypoints import KEYPOINT_SHORT_CODES as K
from app.reports.labels import CRITERION_LABELS, KEYPOINT_LABELS
from training.metrics.localization import pck_at_alpha
from training.mfld_full_comparison import BODY, IX, TIPS, criterion_keypoints, is_fault, load_mfld, load_ours

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "backend" / "app" / "data" / "mfld_benchmark.json"
MFLD_REPORTED_OKS = {"mean_OKS": 0.7086, "AP": 0.4311, "AP.75": 0.3162}   # from mfld-net's own evaluate.py (betta_full run)


def row(key, label, unit, ours, mfld, higher_is_better, note=None):
    better = "ours" if (ours > mfld if higher_is_better else ours < mfld) else "mfld"
    ratio = (ours / mfld if higher_is_better else mfld / ours) if min(ours, mfld) > 0 else None   # >1 means ours is better
    times = None if ratio is None else float(max(ratio, 1.0 / ratio))                              # how many times better the winner is
    return {"key": key, "label": label, "unit": unit, "ours": float(ours), "mfld": float(mfld), "better": better,
            "times_better": times, "higher_is_better": higher_is_better, "note": note, "ours_deferral": None}


def main() -> None:
    mids, mpred, _ = load_mfld("test")
    oids, ov = load_ours("test")
    assert mids == oids
    gt, vis, opred = ov["gt"], ov["vis_mask"].astype(bool), ov["pred_mu"]
    em, eo = np.linalg.norm(mpred - gt, axis=-1), np.linalg.norm(opred - gt, axis=-1)
    rows = []
    for key, label, cols in [("mean_all", "Mean keypoint error", list(range(13))), ("mean_tips", "Mean fin-tip error", TIPS), ("mean_body", "Mean body-keypoint error", BODY)]:
        f = lambda e: float(np.mean([e[i, cols][vis[i, cols]].mean() for i in range(len(e)) if vis[i, cols].any()]))
        rows.append(row(key, label, "px", f(eo), f(em), False))
    rows.append(row("median_all", "Median keypoint error", "px", float(np.median(eo[vis])), float(np.median(em[vis])), False))
    rows.append(row("rmse_all", "RMSE (all keypoints)", "px", float(np.sqrt((eo[vis] ** 2).mean())), float(np.sqrt((em[vis] ** 2).mean())), False))
    per_o = np.array([eo[i][vis[i]].mean() for i in range(len(gt))])
    per_m = np.array([em[i][vis[i]].mean() for i in range(len(gt))])
    win = row("win_rate", "Test images where the model has the lower mean keypoint error", "% of images",
              float((per_o < per_m).mean() * 100), float((per_m < per_o).mean() * 100), True)
    win["times_better"] = None                                  # a ratio of win shares is not meaningful
    rows.append(win)
    for al in (0.05, 0.10):
        f = lambda p: pck_at_alpha(p, gt, vis, al, IX["snout_tip"], IX["peduncle_top"], IX["peduncle_bottom"])["pck"] * 100
        rows.append(row(f"pck_{int(al * 100)}", f"Keypoints within {int(al * 100)}% of body length", "%", f(opred), f(mpred), True))

    maes_o, maes_m, acc_o, acc_m, majority, dec_o, dec_share = [], [], [], [], [], [], []
    tsi_scales, n_cases, n_decided, n_decided_ok = load_tsi_scales(), 0, 0, 0
    n_per_criterion = []
    for key, fn in MORPHOMETRIC_FUNCTIONS.items():
        keep = vis[:, criterion_keypoints(fn)].all(axis=1)      # every image where THIS criterion's keypoints are labelled
        n_per_criterion.append(int(keep.sum()))
        yt = np.array([fn(g.reshape(-1)) for g in gt[keep]])
        ym = np.array([fn(p.reshape(-1)) for p in mpred[keep]])
        yo = np.array([fn(p.reshape(-1)) for p in opred[keep]])
        ft = is_fault(key, yt)
        maes_o.append(np.abs(yo - yt).mean() / np.abs(yt).mean() * 100)
        maes_m.append(np.abs(ym - yt).mean() / np.abs(yt).mean() * 100)
        acc_o.append((is_fault(key, yo) == ft).mean() * 100)
        acc_m.append((is_fault(key, ym) == ft).mean() * 100)
        majority.append(max(ft.mean(), 1 - ft.mean()) * 100)
        # our deployed gate on the same cases: decide only if tsi_scale * sigma_hat < TSI
        decided_mask = []
        for p, c, y, f_true in zip(opred[keep], ov["pred_cov"][keep], yo, ft):
            jac = numerical_jacobian(fn, p.reshape(-1))
            sigma_hat = tsi_scales.get(key, 1.0) * predicted_keypoint_sigma(jac, c)
            n_cases += 1
            confident = is_confident(sigma_hat, compute_tsi(y, CRITERION_THRESHOLDS[key], jac))
            decided_mask.append(confident)
            if confident:
                n_decided += 1
                n_decided_ok += int(is_fault(key, np.array([y]))[0] == f_true)
        decided_mask = np.array(decided_mask)
        share = decided_mask.mean()
        dec_o.append((is_fault(key, yo) == ft)[decided_mask].mean() * 100)
        dec_share.append(share * 100)
    n_note = ("Each criterion is scored on every test image where its own keypoints are labelled: "
              + ", ".join(f"{k} {n}" for k, n in zip(MORPHOMETRIC_FUNCTIONS, n_per_criterion)) + " images.")
    rows.append(row("criterion_error", "Measurement error, 5 IBC criteria (relative)", "%", np.mean(maes_o), np.mean(maes_m), False, n_note))
    acc = row("criterion_accuracy", "Pass/Fault accuracy, 5 criteria", "%", np.mean(acc_o), np.mean(acc_m), True,
              "BettaTool without deferral and MFLD-Net both answer every case. BettaTool with deferral answers only the cases its uncertainty "
              "says it can decide, so its accuracy is over those cases only. Reference: always answering each criterion's most common outcome, "
              f"with no model at all, scores {np.mean(majority):.1f}% (the test set is mostly Pass / mostly Fault per criterion).")
    acc["ours_deferral"] = float(np.mean(dec_o))
    rows.append(acc)
    answered = row("answered", "Share of cases the system gives an answer on", "%", 100.0, 100.0, True,
                   "The rest are deferred to a human judge by BettaTool's gate. MFLD-Net has no deferral, so it answers every case.")
    answered["ours_deferral"] = float(np.mean(dec_share))
    answered["better"], answered["times_better"] = None, None
    rows.append(answered)


    per_kp = [{"name": k, "label": KEYPOINT_LABELS[i], "ours_px": float(eo[vis[:, i], i].mean()), "mfld_px": float(em[vis[:, i], i].mean())} for i, k in enumerate(K)]
    coco = json.loads((ROOT / "backend" / "data" / "annotations" / "annotations.json").read_text())
    meta = {str(i["id"]): i for i in coco["images"]}
    body = np.linalg.norm(gt[:, IX["snout_tip"]] - 0.5 * (gt[:, IX["peduncle_top"]] + gt[:, IX["peduncle_bottom"]]), axis=1)
    # photos 5-8 are the original random sample (seed 0) and stay; photos 1-4 are a fresh random draw (seed 1) from the rest
    keep_ids = ["507", "447", "79", "996"]
    keep_idx = [oids.index(i) for i in keep_ids]
    old_idx = set(np.random.default_rng(0).choice(len(gt), 8, replace=False).tolist())
    pool = [i for i in range(len(gt)) if i not in old_idx]
    fresh = np.random.default_rng(1).choice(pool, 4, replace=False).tolist()
    pick = fresh + keep_idx
    examples = []
    for i in pick:
        iid = oids[int(i)]
        examples.append({"image_id": iid, "file_name": meta[iid]["file_name"], "width": meta[iid]["width"], "height": meta[iid]["height"],
                         "body_length_px": float(body[i]), "ours_mean_px": float(per_o[i]), "mfld_mean_px": float(per_m[i]),
                         "keypoints": [{"name": k, "label": KEYPOINT_LABELS[j], "visible": bool(vis[i, j]),
                                        "gt": [float(gt[i, j, 0]), float(gt[i, j, 1])], "ours": [float(opred[i, j, 0]), float(opred[i, j, 1])],
                                        "mfld": [float(mpred[i, j, 0]), float(mpred[i, j, 1])]} for j, k in enumerate(K)]})
    out = {
        "generated": date.today().isoformat(), "split": "test", "n_images": int(len(gt)),
        "ours": {"name": "BettaTool (HRNet-W32 + probabilistic heads, v3)", "training": "1,161 training images (70%), ImageNet-pretrained backbone, fish-box crop input"},
        "mfld": {"name": "MFLD-Net (retrained)", "training": "1,127 training images (68%) from scratch, fish-box crop input, same split",
                 "reported_oks": MFLD_REPORTED_OKS},
        "protocol": "Both models scored against our annotations on the same 247 held-out test images, in original-photo pixels, with the labelled "
                    "fish box as the crop. Error figures are lower-is-better; PCK and accuracy are higher-is-better.",
        "rows": rows, "per_keypoint": per_kp, "examples": examples,
        "examples_note": "8 test images drawn at random (fixed seeds), not hand-picked. Green = human label, blue = BettaTool, orange = MFLD-Net.",
        "caveats": ["MFLD-Net is far smaller and faster; the accuracy gap partly reflects model size and ImageNet pretraining.",
                    "The live overlay in the app feeds both models the whole photo (the app has no fish detector), which is harder than the benchmark input.",
                    "A single uploaded photo has no ground truth, so differences on one image show disagreement, not which model is right."],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)
    for r in rows:
        print(f"  {r['label']:<56}{r['ours']:>9.2f}{r['mfld']:>9.2f}  {r['better']} better x{(r['times_better'] or 0):.1f}  ({r['unit']})")


if __name__ == "__main__":
    main()
