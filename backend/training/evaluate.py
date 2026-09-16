"""Full evaluation: localization + calibration + visibility metrics, flip
TTA, and qualitative overlay export (Build Prompt v2 §9.1, §10).

    python -m training.evaluate --config training/configs/hrnet_w32.yaml \\
        --checkpoint training/runs/<run>/checkpoints/best.pt --split test

Reports land in `<checkpoint's run dir>/eval_<split>/`: a JSON metrics
summary, the reliability plot PNG, and `overlay/best/` + `overlay/worst/`
(20 images each, Build Prompt v2 §10.5).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.perception.geometry import AffineTransform
from app.perception.heatmap import soft_argmax
from app.perception.hrnet import HRNetKeypointDetector
from app.perception.keypoints import KEYPOINT_SHORT_CODES
from training.dataset import BettaKeypointDataset
from training.metrics.calibration import (
    fixed_sigma_baseline_nll,
    gaussian_nll_numpy,
    mahalanobis_coverage,
    reliability_bins,
    sigma_error_spearman_correlation,
)
from training.metrics.localization import (
    mean_and_median_radial_error,
    overall_rmse,
    pck_at_alpha,
    per_keypoint_rmse,
    radial_errors,
)
from training.splitting import load_splits
from training.transforms import build_eval_transform
from training.utils.config import load_config
from training.viz.overlay import export_best_worst

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_with_flip_tta(model: torch.nn.Module, image: torch.Tensor, enabled: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Prompt v2 §9.1: average heatmaps (not coordinates) from the
    original and horizontally-flipped image, THEN soft-argmax. The flip
    keypoint permutation is the identity (§3.1), so no channel reordering
    is needed on the flipped heatmaps — only the spatial flip-back.

    Covariance and visibility are taken from the ORIGINAL (unflipped) pass
    only — the spec defines TTA for the heatmap/mean pathway; averaging two
    independently-parameterized covariance predictions is not addressed by
    the spec and is left as a documented open item (see training/README.md).
    """
    heatmaps, covariances, visibility_logits = model(image)
    if not enabled:
        mu_crop = HRNetKeypointDetector.mu_to_crop_space(soft_argmax(heatmaps))
        return mu_crop, covariances, visibility_logits

    flipped_image = torch.flip(image, dims=[-1])
    flipped_heatmaps, _, _ = model(flipped_image)

    # Flip the heatmap back spatially, with the standard half-pixel shift
    # correction (Build Prompt v2 §9.1) to avoid a systematic sub-pixel bias.
    flipped_back = torch.flip(flipped_heatmaps, dims=[-1])
    flipped_back = torch.roll(flipped_back, shifts=1, dims=-1)

    averaged = 0.5 * (heatmaps + flipped_back)
    averaged = averaged / averaged.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    mu_crop = HRNetKeypointDetector.mu_to_crop_space(soft_argmax(averaged))
    return mu_crop, covariances, visibility_logits


def evaluate_split(cfg: dict[str, Any], checkpoint_path: str, split: str, device: str) -> dict[str, Any]:
    splits = load_splits(cfg["paths"]["splits_dir"], list(cfg["dataset"]["splits"].keys()))
    img_cfg = cfg["image"]
    ds = BettaKeypointDataset(
        images_dir=cfg["paths"]["images_dir"],
        annotations_path=Path(cfg["paths"]["annotations_dir"]) / "annotations.json",
        split_image_ids=splits[split],
        input_size=img_cfg["input_size"],
        heatmap_size=img_cfg["heatmap_size"],
        bbox_padding=img_cfg["bbox_padding"],
        heatmap_target_sigma_px=cfg["loss"]["heatmap_target_sigma_px"],
        imagenet_mean=img_cfg["imagenet_mean"],
        imagenet_std=img_cfg["imagenet_std"],
        transform_fn=build_eval_transform(),
    )
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    model = HRNetKeypointDetector(backbone_source=cfg["model"]["backbone_source"])
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    tta_enabled = cfg["tta"]["enabled_eval"]

    all_mu_orig, all_cov_orig, all_gt_orig, all_vis_mask, all_vis_prob, all_image_ids = [], [], [], [], [], []

    for batch in loader:
        image = batch["image"].to(device)
        mu_crop, cov_crop, vis_logits = predict_with_flip_tta(model, image, tta_enabled)
        vis_prob = torch.sigmoid(vis_logits)

        for i in range(image.shape[0]):
            affine = AffineTransform(A=batch["affine_A"][i].numpy(), b=batch["affine_b"][i].numpy()).inverse()
            all_mu_orig.append(affine.apply_points(mu_crop[i].cpu().numpy()))
            all_cov_orig.append(affine.apply_covariances(cov_crop[i].cpu().numpy()))
            all_gt_orig.append(affine.apply_points(batch["keypoints_crop"][i].numpy()))
            all_vis_mask.append(batch["visibility_mask"][i].numpy())
            all_vis_prob.append(vis_prob[i].cpu().numpy())
            all_image_ids.append(batch["image_id"][i])

    pred_mu = np.stack(all_mu_orig)
    pred_cov = np.stack(all_cov_orig)
    gt = np.stack(all_gt_orig)
    vis_mask = np.stack(all_vis_mask)
    vis_prob = np.stack(all_vis_prob)

    kp_short = KEYPOINT_SHORT_CODES
    snout_idx, ptop_idx, pbot_idx = kp_short.index("snout_tip"), kp_short.index("peduncle_top"), kp_short.index("peduncle_bottom")

    err = radial_errors(pred_mu, gt, vis_mask)
    mean_err, median_err = mean_and_median_radial_error(pred_mu, gt, vis_mask)

    results: dict[str, Any] = {
        "split": split,
        "n_images": len(all_image_ids),
        "tta_enabled": tta_enabled,
        "overall_rmse_px": overall_rmse(pred_mu, gt, vis_mask),
        "per_keypoint_rmse_px": dict(zip(kp_short, per_keypoint_rmse(pred_mu, gt, vis_mask).tolist())),
        "mean_radial_error_px": mean_err,
        "median_radial_error_px": median_err,
        "pck": {
            str(alpha): pck_at_alpha(pred_mu, gt, vis_mask, alpha, snout_idx, ptop_idx, pbot_idx)
            for alpha in cfg["evaluation"]["pck_alphas"]
        },
        "calibration": {
            "nll": gaussian_nll_numpy(pred_mu, pred_cov, gt, vis_mask),
            "fixed_sigma_baseline_nll": fixed_sigma_baseline_nll(gt, pred_mu, vis_mask, fixed_sigma_px=8.0),
            "coverage": {
                str(level): mahalanobis_coverage(pred_mu, pred_cov, gt, vis_mask, level)
                for level in cfg["evaluation"]["coverage_levels"]
            },
            "sigma_error_spearman": sigma_error_spearman_correlation(pred_cov, err, vis_mask),
        },
    }

    run_dir = Path(checkpoint_path).parent.parent
    out_dir = run_dir / f"eval_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    _save_reliability_plot(pred_cov, err, vis_mask, out_dir / "reliability_plot.png", cfg["evaluation"]["coverage_levels"])

    per_image_error = np.nanmean(err, axis=1)
    images_for_overlay = [_load_original_image(ds, image_id) for image_id in all_image_ids]
    export_best_worst(
        all_image_ids, images_for_overlay, pred_mu, pred_cov, vis_prob, gt, per_image_error,
        out_dir / "overlay", k=cfg["evaluation"]["export_best_worst_k"],
    )

    logger.info("Wrote evaluation report to %s", out_dir)
    return results


def _load_original_image(ds: BettaKeypointDataset, image_id: str) -> np.ndarray:
    from PIL import Image

    sample = next(s for s in ds.samples if str(s["image_id"]) == image_id)
    with Image.open(ds.images_dir / ds.images_by_id[image_id]["file_name"]) as im:
        return np.array(im.convert("RGB"))


def _save_reliability_plot(pred_cov: np.ndarray, err: np.ndarray, vis_mask: np.ndarray, out_path: Path, coverage_levels: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = reliability_bins(pred_cov, err, vis_mask)
    fig, ax = plt.subplots(figsize=(5, 5))
    if bins:
        xs = [b["mean_predicted_sigma"] for b in bins]
        ys = [b["mean_observed_error"] for b in bins]
        ax.scatter(xs, ys, label="observed")
        lim = max(max(xs), max(ys)) * 1.1
        ax.plot([0, lim], [0, lim], "--", color="gray", label="y = x (perfect calibration)")
    ax.set_xlabel("Mean predicted sigma (px)")
    ax.set_ylabel("Mean observed radial error (px)")
    ax.set_title("Reliability plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    results = evaluate_split(cfg, args.checkpoint, args.split, device)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
