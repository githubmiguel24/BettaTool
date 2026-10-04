"""Human-readable labels for criteria and landmarks.

These are the display strings the API hands the frontend. The frontend used
to keep its own hardcoded copies in `src/data/measurements.js` and
`src/data/landmarks.js`; the landmark list there had drifted out of sync
with the model (it listed a "Caudal Fin Peduncle/Anterior" landmark that
does not exist in `app/perception/keypoints.py`, and was missing
`peduncle_bottom`). Serving the labels from here keeps one source of truth
and makes that class of drift impossible.
"""

from app.perception.keypoints import NUM_KEYPOINTS, Keypoint

CRITERION_LABELS = {
    "caudal-spread-angle": "Caudal Spread Angle",
    "dorsal-body-ratio": "Dorsal Fin / Body Ratio Measurement",
    "anal-body-ratio": "Anal Fin / Body Ratio Measurement",
    "caudal-body-ratio": "Caudal Fin / Body Ratio Measurement",
    "anal-length-width-ratio": "Anal Fin Length-to-Width Ratio",
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

# Which keypoint indices each criterion in app/analytical/morphometrics.py
# actually reads, so the frontend can highlight them on click without
# duplicating that module's geometry. Kept index-aligned with the functions
# themselves rather than re-derived, since a silent drift here would just
# highlight the wrong landmarks with no visible error.
CRITERION_LANDMARKS: dict[str, list[int]] = {
    "caudal-spread-angle": [
        Keypoint.CAUDAL_PEDUNCLE_TOP,
        Keypoint.CAUDAL_PEDUNCLE_BOTTOM,
        Keypoint.CAUDAL_FIN_CENTER,
        Keypoint.CAUDAL_FIN_TIP_UPPER,
        Keypoint.CAUDAL_FIN_TIP_LOWER,
    ],
    "dorsal-body-ratio": [
        Keypoint.SNOUT_TIP,
        Keypoint.CAUDAL_PEDUNCLE_TOP,
        Keypoint.CAUDAL_PEDUNCLE_BOTTOM,
        Keypoint.DORSAL_FIN_BASE_ANTERIOR,
        Keypoint.DORSAL_FIN_BASE_POSTERIOR,
        Keypoint.DORSAL_FIN_TIP,
    ],
    "anal-body-ratio": [
        Keypoint.SNOUT_TIP,
        Keypoint.CAUDAL_PEDUNCLE_TOP,
        Keypoint.CAUDAL_PEDUNCLE_BOTTOM,
        Keypoint.ANAL_FIN_BASE_ANTERIOR,
        Keypoint.ANAL_FIN_BASE_POSTERIOR,
        Keypoint.ANAL_FIN_TIP,
    ],
    "caudal-body-ratio": [
        Keypoint.SNOUT_TIP,
        Keypoint.CAUDAL_PEDUNCLE_TOP,
        Keypoint.CAUDAL_PEDUNCLE_BOTTOM,
        Keypoint.CAUDAL_FIN_CENTER,
    ],
    "anal-length-width-ratio": [
        Keypoint.ANAL_FIN_BASE_ANTERIOR,
        Keypoint.ANAL_FIN_BASE_POSTERIOR,
        Keypoint.ANAL_FIN_TIP,
    ],
}

assert set(CRITERION_LANDMARKS) == set(CRITERION_LABELS), (
    "CRITERION_LANDMARKS is missing or has extra keys relative to CRITERION_LABELS"
)
