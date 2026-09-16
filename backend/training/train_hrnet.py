"""Two-stage training entrypoint for the probabilistic HRNet keypoint detector.

    python -m training.train_hrnet --config training/configs/hrnet_w32.yaml

Build Prompt v2 §7-§8: Stage 1 (localization warmup, heatmap MSE + L1-on-mu,
covariance head detached from the loss) runs for `training.stage1_epochs`
epochs, then Stage 2 (joint probabilistic training, NLL loss ramped in over
`loss.nll_ramp_epochs`) runs for `training.stage2_epochs` more, with its own
cosine restart and its own early-stopping window (Stage 1's loss and Stage
2's loss are not on the same scale, so a single early-stopping monitor
across both stages would be meaningless — the spec calls this out
explicitly).

IMPORTANT — see training/README.md, "Known limitations": this file was
authored and carefully reviewed line-by-line against the spec, but `torch`
was not installable in the sandbox this codebase was built in, so this loop
has never actually been run. Run the smoke test
(`pytest tests/test_smoke_dummy.py`) before trusting it on real data.
"""

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
from training.losses import gaussian_nll_loss, heatmap_mse_loss, visibility_bce_loss
from training.splitting import load_splits
from training.transforms import build_eval_transform, build_train_transform
from training.utils.config import load_config
from training.utils.logging import MetricCSVLogger, create_run_dir
from training.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Builds the train (augmented) and val (augmentation-free) DataLoaders from config."""
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

    train_ds = BettaKeypointDataset(
        split_image_ids=splits["train"],
        transform_fn=build_train_transform(cfg["augmentation"]["train"], img_cfg["input_size"]),
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
    model_cfg = cfg["model"]
    detach = model_cfg["detach_mu_for_covariance_stage1"] if stage == 1 else model_cfg["detach_mu_for_covariance_stage2"]
    model = HRNetKeypointDetector(backbone_source=model_cfg["backbone_source"], detach_mu_for_covariance=detach)

    opt_cfg = cfg["training"]["optimizer"]
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.heatmap_head.parameters()) + list(model.covariance_head.parameters()) + list(model.visibility_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": opt_cfg["lr_backbone"]},
            {"params": head_params, "lr": opt_cfg["lr_heads"]},
        ],
        weight_decay=opt_cfg["weight_decay"],
    )
    return model, optimizer


def _visibility_target_from_mask(visibility_mask: torch.Tensor) -> torch.Tensor:
    """The visibility BCE target is trained on EVERY keypoint (Build Prompt v2 §7) —
    visibility_mask (1.0 for flags 2/1) doubles directly as that target."""
    return visibility_mask


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
) -> None:
    """Runs one training stage (Stage 1: heatmap+L1 only; Stage 2: + ramped NLL)."""
    loss_cfg = cfg["loss"]
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
    epochs_without_improvement = 0
    checkpoints_dir = run_dir / "checkpoints"

    for epoch in range(n_epochs):
        model.train()
        epoch_start = time.time()
        lambda_nll = 0.0
        if stage == 2:
            ramp = min(1.0, epoch / max(1, loss_cfg["nll_ramp_epochs"]))
            lambda_nll = loss_cfg["lambda_nll_target"] * ramp

        running = {"heatmap": 0.0, "mu_l1": 0.0, "vis": 0.0, "nll": 0.0, "total": 0.0, "n": 0}
        for step, batch in enumerate(train_loader):
            image = batch["image"].to(device)
            gt_crop = batch["keypoints_crop"].to(device)
            visibility_mask = batch["visibility_mask"].to(device)
            heatmap_target = batch["heatmap_target"].to(device)

            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=cfg["training"]["amp"]):
                heatmaps, covariances, visibility_logits = model(image)
                mu_heatmap = soft_argmax(heatmaps)
                mu_crop = HRNetKeypointDetector.mu_to_crop_space(mu_heatmap)

                loss_heatmap = heatmap_mse_loss(heatmaps, heatmap_target, visibility_mask)
                loss_mu = (torch.abs(mu_crop - gt_crop).sum(-1) * visibility_mask).sum(-1) / visibility_mask.sum(-1).clamp_min(1e-8)
                loss_mu = loss_mu.mean()
                loss_vis = visibility_bce_loss(visibility_logits, _visibility_target_from_mask(visibility_mask))

                loss = loss_cfg["lambda_heatmap"] * loss_heatmap + loss_cfg["lambda_mu_l1"] * loss_mu + loss_cfg["lambda_visibility"] * loss_vis

                loss_nll = torch.tensor(0.0, device=device)
                if stage == 2 and lambda_nll > 0:
                    loss_nll = gaussian_nll_loss(mu_crop, covariances, gt_crop, visibility_mask, beta=loss_cfg["beta_nll"])
                    loss = loss + lambda_nll * loss_nll

                loss = loss / cfg["training"]["grad_accum_steps"]

            scaler.scale(loss).backward()
            if (step + 1) % cfg["training"]["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), loss_cfg["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = image.shape[0]
            running["heatmap"] += loss_heatmap.item() * bs
            running["mu_l1"] += loss_mu.item() * bs
            running["vis"] += loss_vis.item() * bs
            running["nll"] += float(loss_nll.item()) * bs
            running["total"] += loss.item() * cfg["training"]["grad_accum_steps"] * bs
            running["n"] += bs

            if step % cfg["logging"]["log_every_n_steps"] == 0:
                logger.info(
                    "stage=%d epoch=%d step=%d loss=%.4f (heatmap=%.4f mu_l1=%.4f vis=%.4f nll=%.4f, lambda_nll=%.3f)",
                    stage, epoch, step, loss.item(), loss_heatmap.item(), loss_mu.item(), loss_vis.item(), float(loss_nll.item()), lambda_nll,
                )

        scheduler.step()

        val_metrics = evaluate_epoch(model, val_loader, device, cfg)
        n = max(1, running["n"])
        row = {
            "stage": stage,
            "epoch": epoch,
            "lambda_nll": lambda_nll,
            "train_loss_heatmap": running["heatmap"] / n,
            "train_loss_mu_l1": running["mu_l1"] / n,
            "train_loss_vis": running["vis"] / n,
            "train_loss_nll": running["nll"] / n,
            "train_loss_total": running["total"] / n,
            "val_radial_error_px": val_metrics["mean_radial_error_px"],
            "val_mean_sigma_px": val_metrics["mean_sigma_px"],
            "epoch_seconds": time.time() - epoch_start,
        }
        csv_logger.log(row)
        logger.info("=== stage=%d epoch=%d val_radial_error_px=%.3f mean_sigma_px=%.3f ===", stage, epoch, row["val_radial_error_px"], row["val_mean_sigma_px"])

        _check_covariance_collapse(row["val_mean_sigma_px"], row["val_radial_error_px"], cfg)

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "stage": stage},
            checkpoints_dir / "last.pt",
        )

        early_cfg = cfg["training"]["early_stopping"]
        if stage >= early_cfg["active_from_stage"]:
            if row["val_radial_error_px"] < best_val_metric:
                best_val_metric = row["val_radial_error_px"]
                epochs_without_improvement = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "stage": stage}, checkpoints_dir / "best.pt")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_cfg["patience"]:
                    logger.warning("Early stopping at stage=%d epoch=%d (patience=%d exhausted).", stage, epoch, early_cfg["patience"])
                    break


def _check_covariance_collapse(mean_sigma_px: float, mean_error_px: float, cfg: dict[str, Any]) -> None:
    """Build Prompt v2 §7 mandatory diagnostic: warn if sigma has collapsed
    toward sigma_min while error stays flat — the run is invalid regardless
    of RMSE if this triggers."""
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
    """Cheap per-epoch validation pass (no TTA — Build Prompt v2 §9.1 says TTA
    is off during training-time validation for speed; full TTA evaluation
    lives in evaluate.py)."""
    model.eval()
    total_error, total_sigma, n = 0.0, 0.0, 0
    for batch in loader:
        image = batch["image"].to(device)
        gt_crop = batch["keypoints_crop"].to(device)
        visibility_mask = batch["visibility_mask"].to(device)

        heatmaps, covariances, _ = model(image)
        mu_crop = HRNetKeypointDetector.mu_to_crop_space(soft_argmax(heatmaps))

        error = torch.linalg.norm(mu_crop - gt_crop, dim=-1)  # (B, K)
        masked_error = (error * visibility_mask).sum() / visibility_mask.sum().clamp_min(1e-8)

        mean_eig = 0.5 * (covariances[..., 0, 0] + covariances[..., 1, 1])
        sigma = torch.sqrt(mean_eig.clamp_min(0))
        masked_sigma = (sigma * visibility_mask).sum() / visibility_mask.sum().clamp_min(1e-8)

        bs = image.shape[0]
        total_error += masked_error.item() * bs
        total_sigma += masked_sigma.item() * bs
        n += bs

    n = max(1, n)
    return {"mean_radial_error_px": total_error / n, "mean_sigma_px": total_sigma / n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None, help="Overrides config's training.device.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    device = args.device or cfg["training"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU. Training will be very slow.")
        device = "cpu"

    seed_everything(cfg["seed"], cfg["deterministic_cudnn"])
    run_dir = create_run_dir(cfg["paths"]["runs_dir"], cfg["experiment_name"], cfg)
    csv_logger = MetricCSVLogger(run_dir, cfg["logging"]["metric_csv_filename"])
    logger.info("Run directory: %s", run_dir)

    train_loader, val_loader = build_dataloaders(cfg)

    model, optimizer = build_model_and_optimizer(cfg, stage=1)
    model.to(device)
    run_stage(1, model, optimizer, train_loader, val_loader, cfg, run_dir, csv_logger, device)

    # Stage 2: fresh optimizer + cosine restart (Build Prompt v2 §8: "restart
    # the cosine schedule at the Stage 2 boundary; document this choice"),
    # same model weights carried over, detach_mu_for_covariance now False.
    model.detach_mu_for_covariance = cfg["model"]["detach_mu_for_covariance_stage2"]
    opt_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": opt_cfg["lr_backbone"]},
            {
                "params": list(model.heatmap_head.parameters())
                + list(model.covariance_head.parameters())
                + list(model.visibility_head.parameters()),
                "lr": opt_cfg["lr_heads"],
            },
        ],
        weight_decay=opt_cfg["weight_decay"],
    )
    run_stage(2, model, optimizer, train_loader, val_loader, cfg, run_dir, csv_logger, device)

    logger.info("Training complete. Best checkpoint: %s", run_dir / "checkpoints" / "best.pt")


if __name__ == "__main__":
    main()
