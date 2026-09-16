"""Throughput + peak VRAM probe — run BEFORE committing to a long training run
(Build Prompt v2 §8.3). Settles batch size and total projected runtime in
under five minutes on the target RTX 4050 Laptop (6GB VRAM).

    python -m training.benchmark --config training/configs/hrnet_w32.yaml

Times ~20 training iterations after a 5-iteration warmup and reports
images/sec, projected seconds/epoch, projected total training hours, and
`torch.cuda.max_memory_allocated()`.
"""

from __future__ import annotations

import argparse
import time

import torch

from app.perception.heatmap import soft_argmax
from app.perception.hrnet import HRNetKeypointDetector
from app.perception.keypoints import NUM_KEYPOINTS
from training.losses import gaussian_nll_loss, heatmap_mse_loss, visibility_bce_loss
from training.utils.config import load_config


def _synthetic_batch(batch_size: int, input_size: int, heatmap_size: int, device: str) -> dict[str, torch.Tensor]:
    """A shape-correct, content-meaningless batch — enough to measure
    throughput/VRAM without needing the real dataset wired up yet."""
    return {
        "image": torch.randn(batch_size, 3, input_size, input_size, device=device),
        "keypoints_crop": torch.rand(batch_size, NUM_KEYPOINTS, 2, device=device) * input_size,
        "visibility_mask": torch.ones(batch_size, NUM_KEYPOINTS, device=device),
        "heatmap_target": torch.rand(batch_size, NUM_KEYPOINTS, heatmap_size, heatmap_size, device=device),
    }


def run_benchmark(config_path: str, device: str, n_warmup: int = 5, n_timed: int = 20) -> dict[str, float]:
    cfg = load_config(config_path)
    img_cfg, loss_cfg, train_cfg = cfg["image"], cfg["loss"], cfg["training"]
    batch_size = train_cfg["batch_size"]

    model = HRNetKeypointDetector(backbone_source=cfg["model"]["backbone_source"]).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg["amp"] and device.startswith("cuda"))

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    def one_step() -> None:
        batch = _synthetic_batch(batch_size, img_cfg["input_size"], img_cfg["heatmap_size"], device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=train_cfg["amp"]):
            heatmaps, covariances, visibility_logits = model(batch["image"])
            mu_crop = HRNetKeypointDetector.mu_to_crop_space(soft_argmax(heatmaps))

            loss = (
                loss_cfg["lambda_heatmap"] * heatmap_mse_loss(heatmaps, batch["heatmap_target"], batch["visibility_mask"])
                + loss_cfg["lambda_mu_l1"] * torch.abs(mu_crop - batch["keypoints_crop"]).mean()
                + loss_cfg["lambda_visibility"] * visibility_bce_loss(visibility_logits, batch["visibility_mask"])
                + loss_cfg["lambda_nll_target"] * gaussian_nll_loss(mu_crop, covariances, batch["keypoints_crop"], batch["visibility_mask"], beta=loss_cfg["beta_nll"])
            )
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), loss_cfg["grad_clip_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(n_warmup):
        one_step()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    start = time.time()
    for _ in range(n_timed):
        one_step()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.time() - start

    images_per_sec = (n_timed * batch_size * train_cfg["grad_accum_steps"]) / elapsed
    n_train_images_estimate = 1750 * cfg["dataset"]["splits"]["train"]  # per the updated ~1,750-image dataset
    steps_per_epoch = n_train_images_estimate / (batch_size * train_cfg["grad_accum_steps"])
    seconds_per_epoch = steps_per_epoch * elapsed / n_timed
    total_epochs = train_cfg["stage1_epochs"] + train_cfg["stage2_epochs"]

    peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.startswith("cuda") else float("nan")

    report = {
        "device": device,
        "batch_size": batch_size,
        "grad_accum_steps": train_cfg["grad_accum_steps"],
        "images_per_sec": images_per_sec,
        "projected_seconds_per_epoch": seconds_per_epoch,
        "projected_total_hours": seconds_per_epoch * total_epochs / 3600,
        "peak_vram_gb": peak_vram_gb,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available — benchmarking on CPU. VRAM figures will be NaN and "
              "throughput will not be representative of the target RTX 4050.")

    report = run_benchmark(args.config, device)
    print("\n--- Benchmark report ---")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if device.startswith("cuda") and report["peak_vram_gb"] > 5.5:
        print(
            "\nWARNING: peak VRAM is within ~0.5GB of a 6GB card's usable ceiling. Follow "
            "Build Prompt v2 §8.2's fallback order: (1) input 320x320, (2) gradient "
            "checkpointing, (3) HRNet-W18."
        )


if __name__ == "__main__":
    main()
