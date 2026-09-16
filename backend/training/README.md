# training/ — Perceptual Tier training pipeline

Implements Build Prompt v2 (`Thesis Model Prompt` in the project docs):
HRNet-W32 + a per-keypoint covariance head + a per-keypoint visibility head,
trained in two stages, with the calibration metrics the downstream GUM/TSI
tier's trust assumption depends on.

## Setup

```bash
pip install -r requirements.txt -r requirements-training.txt
```

## Quickstart on synthetic data (run this first, always)

```bash
# 1. Generate a synthetic dataset so the pipeline can run before real
#    Roboflow annotations exist.
python -m training.make_dummy_dataset --out data/raw_dummy --n 200

# 2. Point training/configs/base.yaml's paths at data/raw_dummy (or copy
#    data/raw_dummy/{images,annotations.json} into data/raw and
#    data/annotations, matching the default config paths), then split:
python -m training.splitting --ann data/raw_dummy/annotations.json \
    --config training/configs/hrnet_w32.yaml

# 3. Sanity-check the annotations (also run this the moment real labels land):
python -m training.validate_annotations --ann data/raw_dummy/annotations.json \
    --images data/raw_dummy/images --plots-out training/runs/_validation_plots

# 4. Before committing to a long run, benchmark on your actual GPU:
python -m training.benchmark --config training/configs/hrnet_w32.yaml

# 5. Train:
python -m training.train_hrnet --config training/configs/hrnet_w32.yaml

# 6. Evaluate (with flip TTA, calibration metrics, overlay export):
python -m training.evaluate --config training/configs/hrnet_w32.yaml \
    --checkpoint training/runs/<run>/checkpoints/best.pt --split test
```

## Swapping in the real dataset

1. Export from Roboflow as **COCO Keypoints JSON**. The loader
   (`training/dataset.py`) expects the same schema
   `training/make_dummy_dataset.py` emits: per-annotation `keypoints` (flat
   `[x, y, v] * 13`), `fish_box` (`[x, y, w, h]`), and `specimen_id`.
2. **`specimen_id` and the extended visibility schema are very unlikely to
   come out of Roboflow automatically** — see "Open questions", below.
   You will very likely need a small post-export script that merges a
   sidecar CSV/JSON (specimen IDs per image, and any `-1`/out-of-frame
   corrections) into the raw export before this loader will accept it. That
   script does not exist yet.
3. Put the merged file at `data/annotations/annotations.json` and images
   under `data/raw/` (or edit `training/configs/base.yaml`'s `paths`).
4. Run steps 2-6 above for real.

## Reading the outputs

Every run lives in `training/runs/<UTC-timestamp>_<experiment_name>/`:

- `config.resolved.json`, `git_commit.txt` — full reproducibility record.
- `metrics.csv` — one row per epoch (both stages; a `stage` column
  distinguishes them).
- `checkpoints/{last,best}.pt` — `best.pt` only exists once Stage 2's
  early-stopping window has run at least one epoch.
- `eval_<split>/metrics.json` — localization + calibration numbers, ready
  to paste into Chapter 4 (Results).
- `eval_<split>/reliability_plot.png`, `eval_<split>/overlay/{best,worst}/`.

## The pipeline change this introduces (Build Prompt v2 §4.3, §13.2)

`HRNetKeypointDetector.forward` now returns a **3-tuple**
`(heatmaps, covariances, visibility)`, not the old 2-tuple. `app/pipeline.py`
has been updated to consume all three and to use `soft_argmax` +
`app/perception/geometry.py`'s affine transforms as the single source of
truth for coordinates, per the units contract. **This was a deliberate,
reviewed change, not an incidental one** — anything else importing
`HRNetKeypointDetector` directly (there is currently nothing else) would
need the same update.

## Known limitations — read this before trusting any of this code

This codebase was authored and reviewed in a sandboxed environment with
**no outbound network access to PyPI beyond a small pre-approved package
set**. Concretely:

- **`torch` itself could not be installed.** Every file under `training/`
  that imports torch (`train_hrnet.py`, `evaluate.py`, `benchmark.py`,
  `app/perception/hrnet.py`, `app/perception/hrnet_backbone.py`,
  `training/dataset.py`'s tensor-conversion step) was written and manually
  traced through very carefully, but **has never actually been executed**.
  `pytest` itself was also uninstallable, so even the test files below were
  only verified by hand-running their assertions as plain Python, not via
  the pytest suite as written.
- **`timm` and `albumentations` could not be installed either.** The HRNet-
  W32 backbone was therefore implemented from scratch in plain PyTorch
  (`app/perception/hrnet_backbone.py`) rather than depending on `timm` for
  the architecture — `app/perception/hrnet.py`'s `build_backbone()` still
  *prefers* `timm`'s ImageNet-pretrained `hrnet_w32` when it is importable
  (satisfying Build Prompt v2 §2's locked decision), falling back to the
  from-scratch, randomly-initialized version with a logged warning
  otherwise. **The self-contained backbone has never been ImageNet-
  pretrained** — training from scratch on ~1,750 images will converge
  slower and likely to a worse optimum than the spec's assumption of
  transfer learning. This is very likely the single biggest risk in this
  deliverable; see "Recommended next step" below.
- **What actually ran and was verified in this session** (pure NumPy/SciPy/
  PIL/Matplotlib, no torch dependency): the dummy dataset generator,
  annotation validation CLI, grouped/stratified splitting (including the
  no-leakage guarantee), the affine-transform geometry module (round-trips
  verified to floating-point precision, well under the spec's 1px
  tolerance), the closed-form 2x2 Gaussian NLL (cross-checked against
  `scipy.stats.multivariate_normal` to machine precision), all of
  `training/metrics/` (RMSE against known-noise ground truth, Mahalanobis
  coverage recovering the nominal level under correctly-specified
  covariance, Wilson/Clopper-Pearson interval boundary cases, Spearman
  sigma-vs-error correlation under heteroscedastic synthetic noise), and
  the overlay renderer. These give real confidence in the *mathematics*.
  They give **no direct evidence about the PyTorch model, training loop,
  or losses**, beyond careful code review.
- **Run `pytest tests/` yourself before doing anything else** with this
  code, in an environment where `pip install -r requirements.txt -r
  requirements-training.txt` actually succeeds (a Kaggle notebook or your
  local GPU machine both should). `tests/test_model.py`,
  `tests/test_losses_torch.py`, `tests/test_augmentation.py`, and
  `tests/test_smoke_dummy.py` are all written to `pytest.importorskip` the
  dependency they need rather than fail the whole suite — so a green run
  in THIS environment (torch missing) is not evidence of anything; you need
  a green run where those imports actually succeed.
- **Recommended next step, in order**: (1) `pip install` the training
  requirements somewhere with real GPU + network access; (2) run
  `pytest tests/ -v` and fix whatever the first real execution surfaces —
  there almost certainly is something, this is untested code; (3) run the
  synthetic-data quickstart above end to end; (4) only then point it at
  real annotations.

## Open questions (Build Prompt v2 §13) — status

1. **Pretrained weights.** Not resolved — see above. `model.backbone_source:
   auto` in config tries `timm`'s ImageNet weights first; COCO-pose weights
   (which the spec suggests transfer better) are not wired up at all — doing
   so would need a state-dict remapping from whatever COCO-pose checkpoint
   source you pick, since this backbone's parameter names do not match
   `timm`'s naming.
2. **`app/pipeline.py`'s return arity.** Resolved: it now accepts the
   3-tuple directly (with a documented fallback for an old 2-tuple
   checkpoint). See "The pipeline change this introduces", above.
3. **Visibility flag encoding in Roboflow.** Not resolved — unknown whether
   Roboflow's export UI can natively record "out of frame" as a 4th state.
   `training/dataset.py`'s loader accepts a `-1` value in the `keypoints`
   array however it gets there, but the sidecar-merge script Build Prompt
   v2 §3.3 describes as the fallback does not exist yet (see "Swapping in
   the real dataset", above).
4. **`specimen_id` availability.** Not resolved — unknown whether the
   current annotation set can identify repeat photographs of the same
   fish. `training/splitting.py` requires this field and will raise
   loudly if it is missing; if it truly cannot be recovered, the fallback
   is `specimen_id = image_id` (each photo is its own "specimen"), which
   `training/README.md` should document as a limitation in the thesis text
   per the spec's own guidance, rather than silently defaulting to it here.
5. **Sigma scale sanity check.** Cannot be checked at all without a real
   training run — flagged as the first thing to look at once one exists.

## Methodology paragraph (drop-in for Chapter 3)

> The perceptual tier is a probabilistic keypoint detector combining an
> HRNet-W32 backbone (Wang et al., 2020) with three parallel prediction
> heads operating on its fused, stride-4 feature map. Input images are
> letterbox-resized to 384x384 to preserve aspect ratio, since downstream
> morphometric measurements are ratios and angles sensitive to distortion.
> The heatmap head produces a 96x96 spatial-softmax distribution per
> landmark, from which a sub-pixel mean is recovered by soft-argmax
> (integral regression). A covariance head samples the backbone feature at
> that mean, via bilinear interpolation, and combines it with a globally
> pooled context feature to regress a 2x2 positive-definite covariance
> matrix per landmark, parameterized through a softplus/tanh
> transformation (sigma_x, sigma_y > 0, |rho| < 0.95) that guarantees
> positive-definiteness by construction. A visibility head independently
> predicts, per landmark, the probability that it is present and
> locatable in the frame at all, distinguishing "occluded" from
> "out-of-frame" failure modes that a coordinate-only prediction cannot
> represent. Training proceeds in two stages: a 40-epoch localization
> warmup (heatmap MSE against a sigma=2px Gaussian target, plus an L1 term
> on the soft-argmax mean, with the covariance head detached from the
> loss) followed by 110 epochs of joint probabilistic training that adds a
> beta-weighted (beta=0.5) heteroscedastic Gaussian negative log-likelihood
> term, ramped in linearly over the first 10 epochs, with sigma clamped to
> [0.5, 64] pixels to guard against the variance-collapse failure mode
> documented for heteroscedastic regression (Seitzer et al., 2022). Both
> stages use AdamW with separate learning rates for the pretrained backbone
> (1e-4) and randomly-initialized heads (1e-3), cosine-annealed with a
> 3-epoch linear warmup and a fresh restart at the Stage 2 boundary, under
> automatic mixed precision with gradient accumulation to an effective
> batch size of 16. Data is split 70/15/15 into train/validation/test,
> grouped by specimen (so repeat photographs of one fish never cross a
> partition boundary) and stratified by annotation source, color morph, and
> occlusion presence. At inference time, flip test-time augmentation
> averages the heatmap (not the extracted coordinate) from the image and
> its horizontal mirror before soft-argmax, with the standard half-pixel
> shift correction; the flip keypoint permutation is the identity, since
> every one of the 13 landmarks is a midline or dorsal/ventral point in
> this lateral-view photography protocol, never a bilateral pair. Model
> calibration is assessed via Mahalanobis-distance coverage against the
> predicted covariance's 68%/95% confidence ellipses, the Spearman
> correlation between predicted sigma and observed radial error, and a
> reliability diagram, all reported on both the validation set (where
> calibration was implicitly tuned via model selection) and the held-out
> test set, so that the resulting optimism gap is measured rather than
> assumed away.

*(Fill in the backbone-pretraining sentence once you know which path —
timm/ImageNet, timm/COCO-pose, or from-scratch — you actually used; the
paragraph above describes the architecture and training procedure, which
holds regardless.)*
