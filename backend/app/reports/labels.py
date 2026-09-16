"""Human-readable labels for criteria and landmarks.

These are the display strings the API hands the frontend. The frontend used
to keep its own hardcoded copies in `src/data/measurements.js` and
`src/data/landmarks.js`; the landmark list there had drifted out of sync
with the model (it listed a "Caudal Fin Peduncle/Anterior" landmark that
does not exist in `app/perception/keypoints.py`, and was missing
`peduncle_bottom`). Serving the labels from here keeps one source of truth
and makes that class of drift impossible.
"""

from app.perception.keypoints import NUM_KEYPOINTS

CRITERION_LABELS = {
    "caudal-spread-angle": "Caudal Spread Angle",
    "dorsal-body-ratio": "Dorsal Fin / Body Ratio Measurement",
    "anal-body-ratio": "Anal Fin / Body Ratio Measurement",
    "caudal-body-ratio": "Caudal Fin / Body Ratio Measurement",
    "anal-caudal-ratio": "Anal Fin / Caudal Fin Ratio",
    "dorsal-caudal-ratio": "Dorsal Fin / Caudal Fin Ratio",
}

# Index-aligned with app.perception.keypoints.Keypoint / KEYPOINT_SHORT_CODES.
KEYPOINT_LABELS = [
    "Snout Tip",
    "Eye Center",
    "Dorsal Fin Base (Anterior)",
    "Dorsal Fin Base (Posterior)",
    "Dorsal Fin Tip",
    "Caudal Peduncle Top",
    "Caudal Peduncle Bottom",
    "Caudal Fin Tip (Upper)",
    "Caudal Fin Tip (Lower)",
    "Caudal Fin Center",
    "Anal Fin Base (Anterior)",
    "Anal Fin Base (Posterior)",
    "Anal Fin Tip",
]

assert len(KEYPOINT_LABELS) == NUM_KEYPOINTS, (
    f"KEYPOINT_LABELS has {len(KEYPOINT_LABELS)} entries but the schema "
    f"defines {NUM_KEYPOINTS} landmarks."
)
