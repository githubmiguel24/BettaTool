# Two-stage training 
# Stage 1 warms up heatmap and L1 loss 
#  Stage 2 ramps in Gaussian NLL with a fresh LR schedule
# Pass --resume <run_dir>/checkpoints/last.pt to continue an interrupted run (best.pt only holds model weights)

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.perception.heatmap import soft_argmax
from app.perception.hrnet import HRNetKeypointDetector
from training.dataset import BettaKeypointDataset
from app.perception.keypoints import KEYPOINT_SHORT_CODES
from training.losses import gaussian_nll_loss, heatmap_kl_loss
from training.metrics.uncertainty_metrics import gaussian_nll_numpy, mahalanobis_coverage
from training.splitting import load_splits
from training.transforms import build_eval_transform, build_train_transform
from training.utils.config import load_config
from training.utils.ema import ModelEMA
from training.utils.logging import MetricCSVLogger, create_run_dir
from training.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    # set up augmented train loader and clean val loader from config
    splits = load_splits(cfg["paths"]["splits_dir"], list(cfg["dataset"]["splits"].keys()))
    img_cfg = cfg["image"]
    ann_path = Path(cfg["paths"]["annotations_dir"]) / "annotations.json"

    common_kwargs = dict(
        images_dir=cfg["paths"]["images_dir"],
        annotations_path=ann_path,
        input_size=img_cfg["input_size"],
        heatmap_size=img_cfg["heatmap_size"],
        bbox_padding=img_cfg["bbox_padding"],
        heatmap_target_sigma_px=cfg["loss"]["heatmap_target_sigma_px"],
        imagenet_mean=img_cfg["imagenet_mean"],
        imagenet_std=img_cfg["imagenet_std"],
        cache_crops_in_ram=cfg["caching"]["cache_crops_in_ram"],
    )

    aniso_cfg = cfg["loss"].get("heatmap_target_anisotropy", {})
    target_anisotropy = None
    if aniso_cfg.get("enabled", False):
        target_anisotropy = {
            "across_scale": aniso_cfg["across_scale"],
            "keypoint_indices": [KEYPOINT_SHORT_CODES.index(name) for name in aniso_cfg["keypoints"]],
        }
    train_ds = BettaKeypointDataset(
        split_image_ids=splits["train"],
        transform_fn=build_train_transform(cfg["augmentation"]["train"], img_cfg["input_size"]),
        target_anisotropy=target_anisotropy,
        **common_kwargs,
    )
    val_ds = BettaKeypointDataset(
        split_image_ids=splits["val"],
        transform_fn=build_eval_transform(),
        **common_kwargs,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    return train_loader, val_loader


def build_model_and_optimizer(cfg: dict[str, Any], stage: int) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    # init model and AdamW with separate learning rates for backbone and heads
    model_cfg = cfg["model"]
    detach = model_cfg["detach_mu_for_covariance_stage1"] if stage == 1 else model_cfg["detach_mu_for_covariance_stage2"]
    model = HRNetKeypointDetector(backbone_source=model_cfg["backbone_source"], detach_mu_for_covariance=detach)

    opt_cfg = cfg["training"]["optimizer"]
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.heatmap_head.parameters()) + list(model.covariance_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": opt_cfg["lr_backbone"]},
            {"params": head_params, "lr": opt_cfg["lr_heads"]},
        ],
        weight_decay=opt_cfg["weight_decay"],
    )
    return model, optimizer


def run_stage(
    stage: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict[str, Any],
    run_dir: Path,
    csv_logger: MetricCSVLogger,
    device: str,
    start_epoch: int = 0,
    resume_optimizer_state: dict[str, Any] | None = None,
    resume_scaler_state: dict[str, Any] | None = None,
    ema: ModelEMA | None = None,
) -> None:
    # run a single training stage and handle checkpointing and early stopping
    loss_cfg = cfg["loss"]
    nl_cfg = cfg.get("noisy_labels", {})
    tip_names = nl_cfg.get("fin_keypoints", ["dorsal_tip", "caudal_tip_upper", "caudal_tip_lower", "caudal_center", "anal_tip"])
    tip_mask = torch.zeros(len(KEYPOINT_SHORT_CODES), dtype=torch.bool, device=device)
    for name in tip_names:
        tip_mask[KEYPOINT_SHORT_CODES.index(name)] = True
    n_epochs = cfg["training"]["stage1_epochs"] if stage == 1 else cfg["training"]["stage2_epochs"]
    warmup_epochs = cfg["training"]["schedule"]["warmup_epochs"]
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["amp"])

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup_epochs)),
        ],
        milestones=[warmup_epochs],
    )

    best_val_metric = float("inf")
    best_val_tip = float("inf")  # lowest val fin-tip + caudal-centre error, saved as best_tip.pt
    best_val_nll = float("inf")  # stage 2 only: lowest val NLL seen, saved as best_nll.pt
    epochs_without_improvement = 0
    checkpoints_dir = run_dir / "checkpoints"

    if start_epoch >= n_epochs:
        logger.info("Stage %d already completed before resume point; skipping.", stage)
        return

    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
    if resume_scaler_state is not None:
        scaler.load_state_dict(resume_scaler_state)
    if start_epoch > 0:
        # fast-forward scheduler steps to match the Resumed epoch LR
        for _ in range(start_epoch):
            scheduler.step()
        logger.info("Resuming stage %d at epoch %d/%d.", stage, start_epoch, n_epochs)

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_start = time.time()
        lambda_nll = 0.0
        if stage == 2:
            ramp = min(1.0, epoch / max(1, loss_cfg["nll_ramp_epochs"]))
            lambda_nll = loss_cfg["lambda_nll_target"] * ramp

        running = {"heatmap": 0.0, "mu_l1": 0.0, "nll": 0.0, "total": 0.0, "n": 0}
        for step, batch in enumerate(train_loader):
            image = batch["image"].to(device)
            gt_crop = batch["keypoints_crop"].to(device)
            visibility_mask = batch["visibility_mask"].to(device)
            heatmap_target = batch["heatmap_target"].to(device)

            flags = batch["visibility"].to(device)  # annotator flag per keypoint: 2 clear, 1 ambiguous, 0 occluded, -1 out of frame
            ambiguous = (flags == 1).float()
            # per-keypoint weight: ambiguous labels count less in the localisation losses (not in the NLL)
            label_weight = 1.0 - (1.0 - nl_cfg.get("ambiguous_weight", 1.0)) * ambiguous
            loc_weight = visibility_mask * label_weight

            teacher_mu = None
            distill_on = (
                stage == 2 and ema is not None and epoch >= nl_cfg.get("distill_start_epoch", 10**9)
                and (nl_cfg.get("distill_alpha_ambiguous", 0.0) > 0 or nl_cfg.get("distill_alpha_fin", 0.0) > 0)
            )
            if distill_on:
                with torch.no_grad(), torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=cfg["training"]["amp"]):
                    teacher_mu = HRNetKeypointDetector.mu_to_crop_space(soft_argmax(ema.module(image)[0])).float()

            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=cfg["training"]["amp"]):
                heatmaps, covariances = model(image)
                mu_heatmap = soft_argmax(heatmaps)
                mu_crop = HRNetKeypointDetector.mu_to_crop_space(mu_heatmap)

                loss_heatmap = heatmap_kl_loss(heatmaps, heatmap_target, loc_weight)

                # localisation target: the label, blended with the EMA teacher's prediction where the label is untrustworthy
                target_mu = gt_crop
                if teacher_mu is not None:
                    alpha = nl_cfg.get("distill_alpha_ambiguous", 0.0) * ambiguous + nl_cfg.get("distill_alpha_fin", 0.0) * tip_mask.float()
                    alpha = alpha.clamp(0.0, 1.0).unsqueeze(-1)
                    target_mu = (1.0 - alpha) * gt_crop + alpha * teacher_mu
                l1 = torch.abs(mu_crop.float() - target_mu).sum(-1)  # (B, K)

                # trimmed loss: ignore the worst fraction of fin-tip residuals in the batch (likely label noise)
                weights = loc_weight
                trim_q = nl_cfg.get("trim_fraction", 0.0)
                if trim_q > 0 and stage == 2 and epoch >= nl_cfg.get("trim_start_epoch", 0):
                    with torch.no_grad():
                        candidates = (weights > 0) & tip_mask.unsqueeze(0)
                        if candidates.sum() >= 4:
                            threshold = torch.quantile(l1.detach()[candidates], 1.0 - trim_q)
                            weights = weights * (~(candidates & (l1.detach() > threshold))).float()
                loss_mu = ((l1 * weights).sum(-1) / weights.sum(-1).clamp_min(1e-8)).mean()

                loss = loss_cfg["lambda_heatmap"] * loss_heatmap + loss_cfg["lambda_mu_l1"] * loss_mu

                loss_nll = torch.tensor(0.0, device=device)
                if stage == 2 and lambda_nll > 0:
                    # with nll_detach_mu the NLL only trains the covariance (sigma), not the position
                    mu_for_nll = mu_crop.detach() if loss_cfg.get("nll_detach_mu", False) else mu_crop
                    loss_nll = gaussian_nll_loss(mu_for_nll, covariances, gt_crop, visibility_mask, beta=loss_cfg["beta_nll"])
                    loss = loss + lambda_nll * loss_nll

                loss = loss / cfg["training"]["grad_accum_steps"]

            scaler.scale(loss).backward()
            if (step + 1) % cfg["training"]["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), loss_cfg["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

            bs = image.shape[0]
            running["heatmap"] += loss_heatmap.item() * bs
            running["mu_l1"] += loss_mu.item() * bs
            running["nll"] += float(loss_nll.item()) * bs
            running["total"] += loss.item() * cfg["training"]["grad_accum_steps"] * bs
            running["n"] += bs

            if step % cfg["logging"]["log_every_n_steps"] == 0:
                logger.info(
                    "stage=%d epoch=%d step=%d loss=%.4f (heatmap=%.4f mu_l1=%.4f nll=%.4f, lambda_nll=%.3f)",
                    stage, epoch, step, loss.item(), loss_heatmap.item(), loss_mu.item(), float(loss_nll.item()), lambda_nll,
                )

        scheduler.step()

        eval_model = ema.module if ema is not None else model
        val_metrics = evaluate_epoch(eval_model, val_loader, device, cfg)
        raw_val_error = evaluate_epoch(model, val_loader, device, cfg)["mean_radial_error_px"] if ema is not None else ""
        n = max(1, running["n"])
        row = {
            "stage": stage,
            "epoch": epoch,
            "lambda_nll": lambda_nll,
            "train_loss_heatmap": running["heatmap"] / n,
            "train_loss_mu_l1": running["mu_l1"] / n,
            "train_loss_nll": running["nll"] / n,
            "train_loss_total": running["total"] / n,
            "val_radial_error_px": val_metrics["mean_radial_error_px"],
            "val_mean_sigma_px": val_metrics["mean_sigma_px"],
            "val_tip_error_px": val_metrics.get("tip_error_px", ""),
            "val_radial_error_px_raw_weights": raw_val_error,
            "val_nll": val_metrics.get("nll", ""),
            "val_coverage_68": val_metrics.get("coverage_68", ""),
            "val_coverage_95": val_metrics.get("coverage_95", ""),
            "epoch_seconds": time.time() - epoch_start,
        }
        csv_logger.log(row)
        logger.info("=== stage=%d epoch=%d val_radial_error_px=%.3f mean_sigma_px=%.3f ===", stage, epoch, row["val_radial_error_px"], row["val_mean_sigma_px"])
        if "nll" in val_metrics:
            logger.info("    val_nll=%.3f coverage68=%.3f coverage95=%.3f (crop space)", row["val_nll"], row["val_coverage_68"], row["val_coverage_95"])

        _check_covariance_collapse(row["val_mean_sigma_px"], row["val_radial_error_px"], cfg)

        eval_state = eval_model.state_dict()  # EMA weights when EMA is on, so evaluate.py / the app load the better weights
        last = {"model": eval_state, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "stage": stage}
        if ema is not None:
            last["raw_model"] = model.state_dict()
            last["ema_updates"] = ema.updates
        torch.save(last, checkpoints_dir / "last.pt")

        # best_nll.pt: best val NLL (stage 2 only, where the covariance head is trained); opt-in via checkpoint.keep
        if stage >= 2 and "nll" in val_metrics and "best_nll" in cfg["training"]["checkpoint"]["keep"]:
            if val_metrics["nll"] < best_val_nll:
                best_val_nll = val_metrics["nll"]
                torch.save({"model": eval_state, "epoch": epoch, "stage": stage, "val_nll": best_val_nll}, checkpoints_dir / "best_nll.pt")
                logger.info("New best val NLL %.3f at stage=%d epoch=%d -> best_nll.pt", best_val_nll, stage, epoch)

        # best_tip.pt: lowest val fin-tip + caudal-centre error (stage 2), opt-in via checkpoint.keep
        if stage >= 2 and "tip_error_px" in val_metrics and "best_tip" in cfg["training"]["checkpoint"]["keep"]:
            if val_metrics["tip_error_px"] < best_val_tip:
                best_val_tip = val_metrics["tip_error_px"]
                torch.save({"model": eval_state, "epoch": epoch, "stage": stage, "val_tip_error_px": best_val_tip}, checkpoints_dir / "best_tip.pt")
                logger.info("New best val fin-tip error %.3f at stage=%d epoch=%d -> best_tip.pt", best_val_tip, stage, epoch)

        early_cfg = cfg["training"]["early_stopping"]
        metric_key = {"val_radial_error_px": "val_radial_error_px", "val_nll": "val_nll", "val_tip_error_px": "val_tip_error_px"}[early_cfg.get("metric", "val_radial_error_px")]
        metric_value = row[metric_key]
        if stage >= early_cfg["active_from_stage"] and metric_value != "":
            if metric_value < best_val_metric:
                best_val_metric = metric_value
                epochs_without_improvement = 0
                torch.save({"model": eval_state, "epoch": epoch, "stage": stage}, checkpoints_dir / "best.pt")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_cfg["patience"]:
                    logger.warning("Early stopping at stage=%d epoch=%d (patience=%d exhausted).", stage, epoch, early_cfg["patience"])
                    break


def _check_covariance_collapse(mean_sigma_px: float, mean_error_px: float, cfg: dict[str, Any]) -> None:
    # warn if sigma collapses toward sigma_min while radial error stays high
    sigma_min = cfg["units"]["sigma_min_px"]
    if mean_sigma_px <= sigma_min * 1.5 and mean_error_px > sigma_min * 3:
        logger.warning(
            "WARNING: covariance collapse detected (mean predicted sigma=%.3fpx is near "
            "sigma_min=%.3fpx while mean radial error=%.3fpx stays high). This run's "
            "uncertainty estimates are not trustworthy regardless of localization RMSE — "
            "see Build Prompt v2 §7.",
            mean_sigma_px, sigma_min, mean_error_px,
        )


@torch.no_grad()
def evaluate_epoch(model: torch.nn.Module, loader: DataLoader, device: str, cfg: dict[str, Any]) -> dict[str, float]:
    # fast validation pass per epoch
    model.eval()
    total_error, total_sigma, n = 0.0, 0.0, 0
    tip_error_sum, tip_count = 0.0, 0.0
    tip_idx = [KEYPOINT_SHORT_CODES.index(k) for k in cfg.get("noisy_labels", {}).get("fin_keypoints", ["dorsal_tip", "caudal_tip_upper", "caudal_tip_lower", "caudal_center", "anal_tip"])]
    log_calibration = cfg["logging"].get("log_val_calibration", False)
    all_mu, all_cov, all_gt, all_vis = [], [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        gt_crop = batch["keypoints_crop"].to(device)
        visibility_mask = batch["visibility_mask"].to(device)

        heatmaps, covariances = model(image)
        mu_crop = HRNetKeypointDetector.mu_to_crop_space(soft_argmax(heatmaps))

        error = torch.linalg.norm(mu_crop - gt_crop, dim=-1)  # shape (B, K)
        masked_error = (error * visibility_mask).sum() / visibility_mask.sum().clamp_min(1e-8)

        mean_eig = 0.5 * (covariances[..., 0, 0] + covariances[..., 1, 1])
        sigma = torch.sqrt(mean_eig.clamp_min(0))
        masked_sigma = (sigma * visibility_mask).sum() / visibility_mask.sum().clamp_min(1e-8)

        bs = image.shape[0]
        total_error += masked_error.item() * bs
        total_sigma += masked_sigma.item() * bs
        n += bs
        tip_error_sum += (error[:, tip_idx] * visibility_mask[:, tip_idx]).sum().item()
        tip_count += visibility_mask[:, tip_idx].sum().item()

        if log_calibration:
            all_mu.append(mu_crop.float().cpu().numpy())
            all_cov.append(covariances.float().cpu().numpy())
            all_gt.append(gt_crop.float().cpu().numpy())
            all_vis.append(visibility_mask.cpu().numpy())

    n = max(1, n)
    metrics = {"mean_radial_error_px": total_error / n, "mean_sigma_px": total_sigma / n, "tip_error_px": tip_error_sum / max(tip_count, 1.0)}
    if log_calibration:
        # crop-space (384 px) Gaussian NLL and keypoint coverage, same functions as training/evaluate.py
        mu, cov, gt, vis = (np.concatenate(a) for a in (all_mu, all_cov, all_gt, all_vis))
        metrics["nll"] = gaussian_nll_numpy(mu, cov, gt, vis)
        for level in cfg["evaluation"]["coverage_levels"]:
            metrics[f"coverage_{round(level * 100)}"] = mahalanobis_coverage(mu, cov, gt, vis, level)["empirical_coverage"]
    return metrics


def main() -> None:
    # parse CLI args and run both training stages
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None, help="Overrides config's training.device.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a 'last.pt' checkpoint (NOT 'best.pt' — it lacks optimizer/scaler "
        "state) to continue a cut-off run from its next epoch, in its original run directory.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    device = args.device or cfg["training"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU. Training will be very slow.")
        device = "cpu"

    seed_everything(cfg["seed"], cfg["deterministic_cudnn"])

    resume_ckpt: dict[str, Any] | None = None
    resume_stage: int | None = None
    resume_epoch: int | None = None
    if args.resume:
        resume_path = Path(args.resume)
        resume_ckpt = torch.load(resume_path, map_location="cpu")
        if "optimizer" not in resume_ckpt:
            raise SystemExit(
                f"{resume_path} has no optimizer state — only 'last.pt' checkpoints can resume "
                "training ('best.pt' is weights-only, for deployment). Point --resume at last.pt."
            )
        resume_stage = resume_ckpt["stage"]
        resume_epoch = resume_ckpt["epoch"]
        # keep logging into the original run directory so metrics.csv stays continuous
        run_dir = resume_path.parent.parent
        logger.info("Resuming from %s (stage=%d epoch=%d), run_dir=%s", resume_path, resume_stage, resume_epoch, run_dir)
    else:
        run_dir = create_run_dir(cfg["paths"]["runs_dir"], cfg["experiment_name"], cfg)
        logger.info("Run directory: %s", run_dir)

    csv_logger = MetricCSVLogger(run_dir, cfg["logging"]["metric_csv_filename"])

    train_loader, val_loader = build_dataloaders(cfg)

    model, optimizer = build_model_and_optimizer(cfg, stage=1)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt.get("raw_model", resume_ckpt["model"]))
    model.to(device)

    ema: ModelEMA | None = None
    if cfg["training"].get("ema", {}).get("enabled", False):
        ema = ModelEMA(model, decay=cfg["training"]["ema"]["decay"])
        if resume_ckpt is not None and "raw_model" in resume_ckpt:
            ema.module.load_state_dict(resume_ckpt["model"])
            ema.updates = resume_ckpt.get("ema_updates", 0)
        logger.info("EMA enabled (decay %.4f)", ema.decay)

    stage1_start = resume_epoch + 1 if resume_stage == 1 else (cfg["training"]["stage1_epochs"] if resume_stage == 2 else 0)
    run_stage(
        1, model, optimizer, train_loader, val_loader, cfg, run_dir, csv_logger, device,
        start_epoch=stage1_start,
        resume_optimizer_state=resume_ckpt["optimizer"] if resume_stage == 1 else None,
        resume_scaler_state=resume_ckpt["scaler"] if resume_stage == 1 else None,
        ema=ema,
    )

    # rebuild optimizer for Stage 2 with a fresh cosine schedule and updated mu detach flag
    model.detach_mu_for_covariance = cfg["model"]["detach_mu_for_covariance_stage2"]
    opt_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": opt_cfg["lr_backbone"]},
            {
                "params": list(model.heatmap_head.parameters()) + list(model.covariance_head.parameters()),
                "lr": opt_cfg["lr_heads"],
            },
        ],
        weight_decay=opt_cfg["weight_decay"],
    )
    stage2_start = resume_epoch + 1 if resume_stage == 2 else 0
    run_stage(
        2, model, optimizer, train_loader, val_loader, cfg, run_dir, csv_logger, device,
        start_epoch=stage2_start,
        resume_optimizer_state=resume_ckpt["optimizer"] if resume_stage == 2 else None,
        resume_scaler_state=resume_ckpt["scaler"] if resume_stage == 2 else None,
        ema=ema,
    )

    logger.info("Training complete. Best checkpoint: %s", run_dir / "checkpoints" / "best.pt")


if __name__ == "__main__":
    main()