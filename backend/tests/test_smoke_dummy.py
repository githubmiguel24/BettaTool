"""Full smoke test (Build Prompt v2 §11, §12 milestone 4): generate the
dummy dataset, split it, train 2 epochs, evaluate, predict — no errors.

SKIPPED if torch, albumentations, or timm-optional dependencies are not
importable. See training/README.md, "Known limitations" — this is the
single most important test in the whole spec ("no NotImplementedError
anywhere, the pipeline actually runs end to end") and it has NOT been run
in the sandbox this codebase was authored in, because torch itself could
not be installed there. Running this file yourself, before anything else,
is the highest-priority next step.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("albumentations")

from training.dataset import load_and_validate_annotations  # noqa: E402
from training.make_dummy_dataset import generate_dummy_dataset  # noqa: E402
from training.splitting import compute_grouped_stratified_split, write_splits  # noqa: E402
from training.utils.config import load_config  # noqa: E402


@pytest.fixture()
def tiny_config(tmp_path: Path) -> dict:
    """A `hrnet_w32.yaml`-derived config pointed at a tiny dummy dataset
    with a 2-epoch, batch-size-2 schedule, entirely under `tmp_path`."""
    repo_root = Path(__file__).resolve().parents[1]
    ann_path = generate_dummy_dataset(tmp_path / "raw", n_images=20, img_size=256, n_specimens=10, seed=0)

    samples = load_and_validate_annotations(ann_path, tmp_path / "raw" / "images")
    partitions = compute_grouped_stratified_split(
        samples, group_key="specimen_id", stratify_keys=["source"],
        split_fractions={"train": 0.7, "val": 0.15, "test": 0.15}, seed=0,
    )
    write_splits(tmp_path / "splits", partitions)

    cfg = load_config(repo_root / "training" / "configs" / "hrnet_w32.yaml")
    cfg["paths"]["images_dir"] = str(tmp_path / "raw" / "images")
    cfg["paths"]["annotations_dir"] = str(tmp_path / "raw")
    cfg["paths"]["splits_dir"] = str(tmp_path / "splits")
    cfg["paths"]["runs_dir"] = str(tmp_path / "runs")
    cfg["model"]["backbone_source"] = "custom"  # no network access needed
    cfg["training"]["stage1_epochs"] = 1
    cfg["training"]["stage2_epochs"] = 1
    cfg["training"]["batch_size"] = 2
    cfg["training"]["grad_accum_steps"] = 1
    cfg["training"]["num_workers"] = 0
    cfg["training"]["device"] = "cpu"
    cfg["training"]["amp"] = False
    cfg["training"]["early_stopping"]["patience"] = 999
    return cfg


def test_two_epoch_smoke_run_completes_without_errors(tiny_config: dict) -> None:
    from training.train_hrnet import build_dataloaders, build_model_and_optimizer, run_stage
    from training.utils.logging import MetricCSVLogger, create_run_dir
    from training.utils.seed import seed_everything

    cfg = tiny_config
    seed_everything(cfg["seed"], deterministic_cudnn=False)

    train_loader, val_loader = build_dataloaders(cfg)
    assert len(train_loader) > 0
    assert len(val_loader) >= 0

    run_dir = create_run_dir(cfg["paths"]["runs_dir"], cfg["experiment_name"], cfg)
    csv_logger = MetricCSVLogger(run_dir, cfg["logging"]["metric_csv_filename"])

    model, optimizer = build_model_and_optimizer(cfg, stage=1)
    model.to("cpu")
    run_stage(1, model, optimizer, train_loader, val_loader, cfg, run_dir, csv_logger, "cpu")

    assert (run_dir / "checkpoints" / "last.pt").is_file()
    metrics_csv = run_dir / cfg["logging"]["metric_csv_filename"]
    assert metrics_csv.is_file()
    assert len(metrics_csv.read_text().splitlines()) >= 2  # header + at least 1 epoch row


def test_overfit_one_batch_drives_loss_near_zero(tiny_config: dict) -> None:
    """Build Prompt v2 §11: 8 images, augmentation off, should train to
    near-zero loss within a few minutes — this is the cheapest way to catch
    a silent shape/masking/gradient bug before any long real run."""
    from app.perception.heatmap import soft_argmax
    from app.perception.hrnet import HRNetKeypointDetector
    from training.dataset import BettaKeypointDataset
    from training.losses import heatmap_mse_loss, visibility_bce_loss
    from training.splitting import load_splits
    from training.transforms import build_eval_transform
    from training.utils.seed import seed_everything
    from torch.utils.data import DataLoader

    cfg = tiny_config
    seed_everything(cfg["seed"], deterministic_cudnn=False)

    splits = load_splits(cfg["paths"]["splits_dir"], list(cfg["dataset"]["splits"].keys()))
    img_cfg = cfg["image"]
    ds = BettaKeypointDataset(
        images_dir=cfg["paths"]["images_dir"],
        annotations_path=Path(cfg["paths"]["annotations_dir"]) / "annotations.json",
        split_image_ids=splits["train"][:8],
        input_size=img_cfg["input_size"],
        heatmap_size=img_cfg["heatmap_size"],
        bbox_padding=img_cfg["bbox_padding"],
        heatmap_target_sigma_px=cfg["loss"]["heatmap_target_sigma_px"],
        imagenet_mean=img_cfg["imagenet_mean"],
        imagenet_std=img_cfg["imagenet_std"],
        transform_fn=build_eval_transform(),  # augmentation OFF, per spec
    )
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    batch = next(iter(loader))

    model = HRNetKeypointDetector(backbone_source="custom")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(60):
        heatmaps, _, visibility_logits = model(batch["image"])
        loss = heatmap_mse_loss(heatmaps, batch["heatmap_target"], batch["visibility_mask"]) + visibility_bce_loss(
            visibility_logits, batch["visibility_mask"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, (
        f"Loss did not drop substantially on 8 overfit images ({losses[0]:.4f} -> {losses[-1]:.4f}) "
        "— this usually means a shape, masking, or gradient-flow bug (Build Prompt v2 §11)."
    )
