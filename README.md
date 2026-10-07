# bettafish

**Uncertainty-Aware Neuro-Symbolic Morphometrics for Rule-Based Halfmoon Longfin Betta Fish Assessment**
(Group 4, PUP-CCIS thesis)

A web tool that grades a lateral photo of a Halfmoon Longfin betta against five International Betta Congress (IBC) form
rules (caudal spread angle, dorsal / anal / caudal fin-to-body ratios, anal fin length-to-width ratio). Instead of
forcing a verdict, it tells the user when it is not sure enough to decide.

How it works, in three tiers:

1. **Perceptual:** an HRNet-W32 model finds 13 keypoints on the fish and gives each one a 2x2 covariance that says how
   uncertain its position is.
2. **Analytical:** the five IBC measurements are computed from the keypoints, and the keypoint uncertainty is carried
   into each measurement as a margin of error (Jacobian propagation, ISO GUM). The Threshold Sensitivity Index (TSI)
   gives the largest keypoint error, in pixels, that each rule can tolerate.
3. **Decisional:** the IBC rule engine labels each measurement. If the model's predicted keypoint error is below the TSI the
   result is **Confident Pass** or **Confident Fault**; otherwise it is **Defer to Judge** and a human decides.

The system is decision support, not a replacement for judges. A "Compare with MFLD-Net" tab shows how a deterministic
baseline landmark detector differs from this model.

Layout: `backend/` (FastAPI service and training code), `frontend/` (React dashboard), `analysis/` (confusion-matrix
scripts), `mfld-net/` (the MFLD-Net baseline, a separate repository).
