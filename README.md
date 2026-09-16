# bettafish

Monorepo for the **Uncertainty-Aware Neuro-Symbolic Morphometrics for
Rule-Based Halfmoon Longfin Betta Fish Assessment** thesis (Group 4,
PUP-CCIS). Combines what were previously two separate repositories:

- `backend/` — FastAPI service + the offline HRNet training pipeline
  (`backend/training/`). See `backend/README.md` (overall status) and
  `backend/training/README.md` (perceptual-tier detail, and the "Known
  limitations" section — read this one first).
- `frontend/` — the React/Vite dashboard UI. See `frontend/README.md`.

## Running both together

```bash
# Terminal 1 — backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload   # http://localhost:8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev                      # http://localhost:5173, CORS-allowed by default
```

**They are not actually wired together yet.** The frontend currently makes
zero HTTP requests to the backend — `UploadView.jsx` stores the picked file
locally (`URL.createObjectURL`) and every measurement row is a static
"Pending" placeholder. `POST /analyze` on the backend, in turn, returns
HTTP 501 because there is no trained checkpoint to run yet
(`app/api/routes/analyze.py`). Both sides exist and both are individually
reasonable next steps, but connecting them was out of scope for this pass
— see `backend/README.md`'s Status section for the itemized list of what
that would take.

## What changed in this pass

1. **Repository merge**: `bettafish-backend` and `bettafish-frontend` were
   combined into this one repo, unmodified apart from the backend changes
   in (2).
2. **Perceptual tier build-out** (`backend/training/`, plus
   `backend/app/perception/`): implemented the HRNet-W32 + covariance +
   visibility head architecture, the two-stage NLL training loop, the
   COCO-Keypoints dataset/augmentation/splitting pipeline, calibration
   metrics, and a synthetic dummy-dataset generator so all of it can run
   before real annotations exist — per the "Build Prompt v2" specification.
   Full detail, and an honest account of what could and could not be
   verified, is in `backend/training/README.md`.
3. **Dataset size update**: the manuscript/Build Prompt v1 reference
   ~1,500 images; the project's current count is ~1,750. Nothing in the
   split logic hardcodes a count (it's all fractions), so this is a
   one-line config change plus updated comments/docstrings — see
   `backend/training/configs/base.yaml`.
