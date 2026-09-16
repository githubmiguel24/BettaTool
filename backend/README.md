# bettafish-backend

Backend service for the **Uncertainty-Aware Neuro-Symbolic Morphometrics for
Rule-Based Halfmoon Longfin Betta Fish Assessment** thesis. Implements the
three-tier confidence-gated pipeline described in Chapter 3 (System
Architecture):

1. **Perceptual Tier** (`app/perception/`) — HRNet-based probabilistic
   keypoint detection; outputs a mean vector + 2x2 covariance matrix per
   anatomical landmark instead of a single fixed coordinate.
2. **Analytical Tier** (`app/analytical/`) — ISO GUM uncertainty propagation
   (Jacobian method, `u_c(y) = sqrt(J * Sigma * J^T)`) and the Threshold
   Sensitivity Index (TSI).
3. **Decisional Tier** (`app/decisional/`) — static IBC rule engine gated by
   a selective abstention rule: if the actual keypoint RMSE exceeds the TSI
   for a criterion, that criterion is deferred to a human judge instead of
   being auto-classified.

`app/pipeline.py` orchestrates all three tiers end-to-end and is exposed over
HTTP via FastAPI (`app/api/`) for the `bettafish-frontend` client.

## Repository layout

```
app/                  the served FastAPI application
  api/                 routes + request/response schemas
  perception/
    hrnet.py             heads (heatmap/covariance/visibility) + build_backbone()
    hrnet_backbone.py     self-contained HRNet-W32 (no timm dependency)
    heatmap.py            soft_argmax (shared train/infer) + legacy moment fit
    geometry.py           affine transforms: crop <-> heatmap <-> original space
    keypoints.py          13-landmark schema, visibility enum, flip map, skeleton
  analytical/          morphometric functions, Jacobian, GUM propagation, TSI
  decisional/          IBC thresholds, rule engine, abstention gate
  reports/             assembling + exporting the Reliability-Annotated Report
  core/                settings, persistence
  pipeline.py          wires perception -> analytical -> decisional together
training/              offline HRNet training — see training/README.md
  configs/               base.yaml, hrnet_w32.yaml, keypoints.yaml
  dataset.py             COCO-Keypoints loader + validation
  transforms.py          Albumentations pipeline (train aug / eval identity)
  targets.py             Gaussian heatmap target rendering
  splitting.py           grouped + stratified train/val/test split (+ CLI)
  make_dummy_dataset.py  synthetic dataset generator
  validate_annotations.py  annotation sanity-check CLI
  losses/                heatmap_mse, gaussian_nll (+beta-NLL), visibility_bce
  metrics/               localization (RMSE/PCK), calibration (coverage/NLL)
  viz/overlay.py         keypoint + covariance-ellipse overlay renderer
  train_hrnet.py         two-stage training loop
  evaluate.py            full eval: flip TTA, calibration, overlay export
  benchmark.py           throughput/VRAM probe
evaluation/            RQ1-RQ3 experiment scripts (Chapter 3 / Appendix 1)
data/                  raw images, annotations, train/val/test splits (gitignored)
tests/                 unit tests — see training/README.md for what's actually verified
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Requires Python 3.10+ (uses PEP 604 union types and builtin generics).

Run tests with:

```bash
pytest
```

## Status

_Last updated 2026-09-15, after implementing Build Prompt v2 (see
`training/README.md` for the full detail behind every line below)._

- **Implemented and unit-tested (verified — pure NumPy/SciPy, no torch
  dependency)**: GUM uncertainty propagation (`app/analytical/
  gum_propagation.py`), the Threshold Sensitivity Index
  (`app/analytical/tsi.py`), the six morphometric measurement functions
  (`app/analytical/morphometrics.py`), the static IBC rule engine +
  selective abstention gate (`app/decisional/`), and the FastAPI
  route/schema layer.
- **Implemented, per Build Prompt v2, but NOT executable-verified in this
  session** (torch could not be installed in the sandbox this was authored
  in — see `training/README.md`, "Known limitations", before trusting any
  of it): a real HRNet-W32 backbone (`app/perception/hrnet_backbone.py`,
  self-contained; `app/perception/hrnet.py` prefers `timm`'s ImageNet
  weights when available), the covariance and visibility heads, the
  two-stage NLL training loop (`training/train_hrnet.py`), the COCO-
  Keypoints dataset loader and Albumentations augmentation pipeline, and
  `evaluate.py`/`benchmark.py`. **No checkpoint has been trained** — there
  is still no annotated real dataset, and even once one exists, this loop
  has to be run and debugged for real before it produces anything usable.
- **Implemented and verified (pure NumPy/SciPy/PIL/Matplotlib)**: the
  synthetic dummy-dataset generator, the annotation validation CLI,
  grouped+stratified train/val/test splitting (no-leakage guaranteed),
  the crop/letterbox/affine geometry module (round-trips to
  floating-point precision), and every localization/calibration metric —
  see `training/README.md` for exactly what was run and what its output
  was.
- **Still stubbed with `TODO`s**: the MFLD-Net baseline
  (`evaluation/baselines/mfld_net.py`) and PDF export
  (`app/reports/exporters.py`).
- **Still unwired end-to-end**: `POST /analyze` (`app/api/routes/
  analyze.py`) still returns HTTP 501 — it does not yet decode an upload,
  crop it, load a checkpoint, or call `AssessmentPipeline.analyze`. This is
  unchanged by Build Prompt v2, which explicitly scoped out the web/API
  layer; it is real remaining work before the frontend can show a live
  result. Report persistence (`app/core/db.py`) is still an in-memory dict.
- **The frontend (`../frontend/`) makes zero HTTP calls to this backend at
  all** — it is a static UI shell (upload button stores a local
  `URL.createObjectURL`, "Pending" placeholders never resolve). Wiring
  `UploadView.jsx` to `POST /analyze` is unstarted work, not something this
  session touched.
- **Needs verification against the IBC Exhibition Standards Book**: the
  exact numeric thresholds for the five fin-ratio criteria in
  `app/decisional/ibc_standards.py` are placeholders — only the caudal
  spread angle bands (Appendix 1, Table 5) are taken directly from the
  document.
- **Dataset size**: the thesis manuscript/Build Prompt v1 cite ~1,500
  images; the project's current figure is ~1,750. `training/configs/
  base.yaml`'s `dataset.expected_total_images` and all split math is
  expressed as fractions (70/15/15) rather than hardcoded counts, so this
  changes nothing structurally — it only changes the split-composition
  table's absolute numbers once real data exists.
