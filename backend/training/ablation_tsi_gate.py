"""Ablation: legacy isotropic TSI gate vs. the corrected anisotropic
(GUM-consistent) gate, on a held-out split.

WHY THIS EXISTS: app/decisional/abstention_gate.py used to decide
Pass/Fault/Defer using `is_confident_isotropic` -- a Jacobian-only TSI
(implicitly Sigma = I) compared against a single RMSE blended across all
13 keypoints' variances, regardless of which keypoints a given criterion's
Jacobian actually depends on. That was NEVER the same number reported to
the user as the measurement's own +/- margin of error, which was already
computed correctly from the real per-keypoint anisotropic covariance (see
app/analytical/gum_propagation.py's `combined_uncertainty`). The pipeline
now uses the corrected rule, `is_confident_gum`, by default.

This script answers the obvious follow-up question: did that inconsistency
actually change anything? For every image x criterion in the split, it
computes BOTH gates' decision from the SAME model prediction, and scores
each against an "oracle" label computed directly from the ground-truth
keypoints (no model uncertainty involved at all). Reports, per criterion:

  - each gate's defer rate
  - each gate's accuracy on the specimens it did NOT defer (i.e. how often
    a "confident" Pass/Fault call actually agreed with ground truth)
  - how often the two gates disagreed, with up to 30 concrete examples

Usage:
    python -m training.ablation_tsi_gate --config training/configs/hrnet_w32.yaml \\
        --checkpoint training/runs/<run>/checkpoints/best.pt --split test

NOTE: this reuses `predict_with_flip_tta` from training/evaluate.py
verbatim (same model-loading and prediction path already exercised by your
normal evaluation run) rather than reimplementing it, so the only new
logic here is the scoring loop itself, which mirrors app/pipeline.py's own
per-criterion loop line for line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.analytical.gum_propagation import (
    assemble_block_covariance,
    combined_uncertainty,
    expanded_uncertainty,
)
from app.analytical.jacobian import numerical_jacobian
from app.analytical.morphometrics import MORPHOMETRIC_FUNCTIONS
from app.analytical.tsi import compute_tsi, is_confident_gum, is_confident_isotropic
from app.decisional.ibc_standards import CRITERION_THRESHOLDS
from app.decisional.rule_engine import classify
from app.perception.geometry import AffineTransform
from app.perception.hrnet import HRNetKeypointDetector
from training.dataset import BettaKeypointDataset
from training.evaluate import predict_with_flip_tta
from training.splitting import load_splits
from training.transforms import build_eval_transform
from training.utils.config import load_config


def _pass_or_fault(label: str) -> str:
    """Collapses a rule-engine label (Pass/Ideal/Slight Fault/Major
    Fault/Disqualify/...) to the binary side, matching the same collapse
    app/decisional/abstention_gate.py already uses for Confident Pass vs
    Confident Fault."""
    return "Pass" if label in ("Pass", "Ideal") else "Fault"


def run_ablation(cfg: dict[str, Any], checkpoint_path: str, split: str, device: str) -> dict[str, Any]:
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

    tally = {
        key: {
            "n": 0,
            "iso_defer": 0,
            "gum_defer": 0,
            "iso_confident_correct": 0,
            "iso_confident_wrong": 0,
            "gum_confident_correct": 0,
            "gum_confident_wrong": 0,
            "disagreements": 0,
        }
        for key in MORPHOMETRIC_FUNCTIONS
    }
    disagreement_examples: list[dict[str, Any]] = []

    for batch in loader:
        image = batch["image"].to(device)
        mu_crop, cov_crop, _ = predict_with_flip_tta(model, image, tta_enabled)

        for i in range(image.shape[0]):
            affine = AffineTransform(A=batch["affine_A"][i].numpy(), b=batch["affine_b"][i].numpy()).inverse()
            pred_mu = affine.apply_points(mu_crop[i].cpu().numpy())
            pred_cov = affine.apply_covariances(cov_crop[i].cpu().numpy())
            gt_mu = affine.apply_points(batch["keypoints_crop"][i].numpy())
            image_id = batch["image_id"][i]

            flat_pred = pred_mu.reshape(-1)
            flat_gt = gt_mu.reshape(-1)
            block_covariance = assemble_block_covariance(pred_cov)

            for criterion_key, measurement_fn in MORPHOMETRIC_FUNCTIONS.items():
                jacobian = numerical_jacobian(measurement_fn, flat_pred)
                measurement = measurement_fn(flat_pred)
                gt_measurement = measurement_fn(flat_gt)
                threshold = CRITERION_THRESHOLDS[criterion_key]

                u_c = combined_uncertainty(jacobian, block_covariance)
                uncertainty = expanded_uncertainty(u_c)
                tsi = compute_tsi(measurement, threshold, jacobian)
                actual_rmse = float(np.sqrt(np.mean(np.diag(block_covariance))))

                iso_confident = is_confident_isotropic(actual_rmse, tsi)
                gum_confident = is_confident_gum(measurement, threshold, uncertainty)

                oracle = _pass_or_fault(classify(criterion_key, gt_measurement))
                predicted_side = _pass_or_fault(classify(criterion_key, measurement))

                t = tally[criterion_key]
                t["n"] += 1

                if not iso_confident:
                    t["iso_defer"] += 1
                elif predicted_side == oracle:
                    t["iso_confident_correct"] += 1
                else:
                    t["iso_confident_wrong"] += 1

                if not gum_confident:
                    t["gum_defer"] += 1
                elif predicted_side == oracle:
                    t["gum_confident_correct"] += 1
                else:
                    t["gum_confident_wrong"] += 1

                if iso_confident != gum_confident:
                    t["disagreements"] += 1
                    if len(disagreement_examples) < 30:
                        disagreement_examples.append(
                            {
                                "image_id": image_id,
                                "criterion": criterion_key,
                                "measurement": measurement,
                                "gt_measurement": gt_measurement,
                                "threshold": threshold,
                                "oracle": oracle,
                                "predicted_side": predicted_side,
                                "iso_confident": iso_confident,
                                "gum_confident": gum_confident,
                                "isotropic_rmse_px": actual_rmse,
                                "isotropic_tsi_px": tsi,
                                "gum_expanded_uncertainty": uncertainty,
                            }
                        )

    summary: dict[str, Any] = {}
    for key, t in tally.items():
        n = t["n"]
        iso_conf_n = t["iso_confident_correct"] + t["iso_confident_wrong"]
        gum_conf_n = t["gum_confident_correct"] + t["gum_confident_wrong"]
        summary[key] = {
            "n_images": n,
            "isotropic_legacy": {
                "defer_rate": t["iso_defer"] / n,
                "n_confident": iso_conf_n,
                "accuracy_when_confident": (t["iso_confident_correct"] / iso_conf_n) if iso_conf_n else None,
            },
            "gum_anisotropic_corrected": {
                "defer_rate": t["gum_defer"] / n,
                "n_confident": gum_conf_n,
                "accuracy_when_confident": (t["gum_confident_correct"] / gum_conf_n) if gum_conf_n else None,
            },
            "disagreement_rate": t["disagreements"] / n,
        }

    return {"split": split, "per_criterion": summary, "disagreement_examples": disagreement_examples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Where to write the JSON report (default: <checkpoint's run dir>/ablation_tsi_gate_<split>.json).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    results = run_ablation(cfg, args.checkpoint, args.split, device)

    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent.parent / f"ablation_tsi_gate_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results["per_criterion"], indent=2))
    print(f"\n{len(results['disagreement_examples'])} example disagreements included. Full report: {out_path}")


if __name__ == "__main__":
    main()
