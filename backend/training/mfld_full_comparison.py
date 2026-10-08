from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from app.analytical.morphometrics import MORPHOMETRIC_FUNCTIONS
from app.analytical.jacobian import numerical_jacobian
from app.perception.keypoints import KEYPOINT_SHORT_CODES as K
from app.decisional.rule_engine import classify
from training.metrics.localization import pck_at_alpha

ROOT = Path(__file__).resolve().parents[2]
NAME_MAP = {"dorsal_fin_base_anterior": "dorsal_base_ant", "dorsal_fin_base_posterior": "dorsal_base_post", "dorsal_fin_tip": "dorsal_tip",
            "caudal_peduncle_top": "peduncle_top", "caudal_peduncle_bottom": "peduncle_bottom", "caudal_fin_tip_upper": "caudal_tip_upper",
            "caudal_fin_tip_lower": "caudal_tip_lower", "caudal_fin_center": "caudal_center", "anal_fin_base_anterior": "anal_base_ant",
            "anal_fin_base_posterior": "anal_base_post", "anal_fin_tip": "anal_tip"}
NOT_FAULT = ("Pass", "Ideal")


def is_fault(key: str, values: np.ndarray) -> np.ndarray:
    return np.array([classify(key, float(v)) not in NOT_FAULT for v in values])


def criterion_keypoints(fn) -> list[int]:
    """Indices of the keypoints a criterion's measurement actually depends on (non-zero Jacobian columns). A test image can be
    scored on a criterion whenever THESE are labelled, even if another keypoint (e.g. the caudal centre) is missing."""
    x = np.random.default_rng(0).uniform(10.0, 500.0, 2 * len(K))
    jac = numerical_jacobian(fn, x).reshape(len(K), 2)
    return [j for j in range(len(K)) if np.abs(jac[j]).sum() > 1e-9]


IX = {k: i for i, k in enumerate(K)}
TIPS = [IX[t] for t in ["dorsal_tip", "caudal_tip_upper", "caudal_tip_lower", "caudal_center", "anal_tip"]]
BODY = [i for i in range(13) if i not in TIPS]


def load_mfld(split):
    """The dump's columns are in OUR keypoint order. Its `keypoint_names` list has caudal_peduncle_top / dorsal_fin_tip
    (columns 4 and 5) labelled the other way round, but matching every column to the ground truth shows the identity
    order (column 4 = dorsal tip, error ~65 px; column 5 = peduncle top, ~18 px), so no re-ordering is applied."""
    d = np.load(ROOT / "mfld-net" / f"predictions_{split}.npz", allow_pickle=True)
    return [str(i) for i in d["image_ids"]], d["pred_mu"].astype(float), d["pred_conf"]


def load_ours(split):
    v = np.load(ROOT / "backend" / "training" / "runs" / "v3_run" / f"eval_{split}" / "predictions.npz")
    return [str(i) for i in v["image_ids"]], v


def per_image(e, vis, cols):
    return np.array([e[i, cols][vis[i, cols]].mean() if vis[i, cols].any() else np.nan for i in range(len(e))])


def main() -> None:
    rng = np.random.default_rng(0)
    for split in ("test", "val"):
        mids, mpred, _ = load_mfld(split)
        oids, ov = load_ours(split)
        assert mids == oids, "image order differs"
        gt, vis, opred = ov["gt"], ov["vis_mask"].astype(bool), ov["pred_mu"]
        em, eo = np.linalg.norm(mpred - gt, axis=-1), np.linalg.norm(opred - gt, axis=-1)
        n = len(gt)
        print(f"\n{'=' * 18} {split.upper()}  ({n} images, identical for both models, original px) {'=' * 18}")
        print(f"{'':<32}{'MFLD retrained':>16}{'ours v3':>10}{'MFLD/ours':>11}   95% paired bootstrap CI of (ours - MFLD)")
        for name, cols in [("mean error, all keypoints", list(range(13))), ("mean error, fin tips + centre", TIPS), ("mean error, body keypoints", BODY)]:
            a, b = per_image(em, vis, cols), per_image(eo, vis, cols)
            ok = ~np.isnan(a) & ~np.isnan(b)
            d = lambda idx: np.mean(b[idx][ok[idx]] - a[idx][ok[idx]])
            ci = np.percentile([d(rng.integers(0, n, n)) for _ in range(2000)], [2.5, 97.5])
            print(f"{name:<32}{np.nanmean(a):>16.1f}{np.nanmean(b):>10.1f}{np.nanmean(a) / np.nanmean(b):>10.1f}x   {d(np.arange(n)):+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]")
        x_m, x_o = em[vis], eo[vis]
        print(f"{'median error (all keypoints)':<32}{np.median(x_m):>16.1f}{np.median(x_o):>10.1f}{np.median(x_m) / np.median(x_o):>10.1f}x")
        print(f"{'RMSE (all keypoints)':<32}{np.sqrt((x_m ** 2).mean()):>16.1f}{np.sqrt((x_o ** 2).mean()):>10.1f}{np.sqrt((x_m ** 2).mean()) / np.sqrt((x_o ** 2).mean()):>10.1f}x")
        for al in (0.02, 0.05, 0.10):
            f = lambda p: pck_at_alpha(p, gt, vis, al, IX["snout_tip"], IX["peduncle_top"], IX["peduncle_bottom"])["pck"] * 100
            print(f"{'PCK@' + str(al):<32}{f(mpred):>15.0f}%{f(opred):>9.0f}%")

        if split == "test":
            print("\nper-keypoint mean error px (MFLD / ours / MFLD÷ours):")
            for k, j in IX.items():
                m = vis[:, j]
                print(f"  {k:<17}{em[m, j].mean():>7.1f}{eo[m, j].mean():>7.1f}{em[m, j].mean() / eo[m, j].mean():>7.1f}x")

        keep = vis.all(axis=1)
        print(f"\nIBC criteria on {keep.sum()} images (forced Pass/Fault, no deferral): MAE of the measurement and accuracy vs the rule engine")
        print(f"{'criterion':<26}{'MAE MFLD':>10}{'MAE ours':>10}{'ratio':>7} |{'acc MFLD':>9}{'acc ours':>9}   paired CI (ours - MFLD) accuracy")
        for key, fn in MORPHOMETRIC_FUNCTIONS.items():
            yt = np.array([fn(g.reshape(-1)) for g in gt[keep]])
            ym = np.array([fn(p.reshape(-1)) for p in mpred[keep]])
            yo = np.array([fn(p.reshape(-1)) for p in opred[keep]])
            ft = is_fault(key, yt)
            am, ao = (is_fault(key, ym) == ft).astype(float), (is_fault(key, yo) == ft).astype(float)
            k = len(yt)
            dd = lambda idx: (ao[idx] - am[idx]).mean()
            ci = np.percentile([dd(rng.integers(0, k, k)) for _ in range(2000)], [2.5, 97.5])
            print(f"{key:<26}{np.abs(ym - yt).mean():>10.4f}{np.abs(yo - yt).mean():>10.4f}{np.abs(ym - yt).mean() / np.abs(yo - yt).mean():>6.1f}x |{am.mean():>9.3f}{ao.mean():>9.3f}   {dd(np.arange(k)):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")


if __name__ == "__main__":
    main()
