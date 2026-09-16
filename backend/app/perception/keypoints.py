"""Canonical keypoint schema for the Halfmoon Longfin perception tier.

Order and names mirror the 13 anatomical landmarks defined in the thesis
(Chapter 3, System Architecture -> Perceptual Tier) and the frontend's
`src/data/landmarks.js`.
"""

from enum import IntEnum


class Keypoint(IntEnum):
    """Index of each landmark in the (N, 2) keypoint array / (N, 2, 2) covariance array."""

    SNOUT_TIP = 0
    EYE_CENTER = 1
    DORSAL_FIN_BASE_ANTERIOR = 2
    DORSAL_FIN_BASE_POSTERIOR = 3
    DORSAL_FIN_TIP = 4
    CAUDAL_PEDUNCLE_TOP = 5
    CAUDAL_PEDUNCLE_BOTTOM = 6
    CAUDAL_FIN_TIP_UPPER = 7
    CAUDAL_FIN_TIP_LOWER = 8
    CAUDAL_FIN_CENTER = 9
    ANAL_FIN_BASE_ANTERIOR = 10
    ANAL_FIN_BASE_POSTERIOR = 11
    ANAL_FIN_TIP = 12


NUM_KEYPOINTS = len(Keypoint)
KEYPOINT_NAMES = [kp.name for kp in Keypoint]

# Short codes matching training/configs/keypoints.yaml (snake_case, used in
# COCO annotation tooling and file names). Order MUST match `Keypoint` above;
# tests/test_keypoints_consistency.py asserts this against the YAML file.
KEYPOINT_SHORT_CODES: list[str] = [
    "snout_tip",
    "eye_center",
    "dorsal_base_ant",
    "dorsal_base_post",
    "dorsal_tip",
    "peduncle_top",
    "peduncle_bottom",
    "caudal_tip_upper",
    "caudal_tip_lower",
    "caudal_center",
    "anal_base_ant",
    "anal_base_post",
    "anal_tip",
]

# Skeleton edges for overlay visualization only (training/viz/overlay.py).
SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (5, 7),
    (6, 8), (7, 9), (8, 9), (6, 10), (10, 11), (11, 12), (0, 12),
]

# Horizontal-flip permutation map: the IDENTITY, deliberately.
#
# Every one of the 13 landmarks lies on the midline or is a single
# dorsal/ventral point — none are bilateral left/right pairs — because every
# photograph is a lateral-flare shot of one side of the fish. Do NOT paste in
# a human-pose-style left/right swap map here; that would silently corrupt
# every horizontally-flipped augmented sample. tests/test_flip_map.py
# enforces this at both the Python and training/configs/keypoints.yaml level.
FLIP_MAP: list[int] = list(range(NUM_KEYPOINTS))
assert FLIP_MAP == list(range(NUM_KEYPOINTS)), (
    "FLIP_MAP must be the identity permutation: none of the 13 Betta "
    "landmarks are bilateral pairs (see module docstring)."
)


class Visibility(IntEnum):
    """Extended visibility schema (Build Prompt v2 §3.3) — richer than plain
    COCO visibility, because Betta photographs distinguish two failure modes
    that a fixed heatmap/NLL target must treat very differently:

        CLEAR (2)        clearly visible                 -> trained, target 1
        AMBIGUOUS (1)    visible but partially occluded   -> trained, target 1
        OCCLUDED (0)     occluded, position inferable      -> masked,  target 0
        OUT_OF_FRAME (-1) not in the photograph at all      -> masked,  target 0

    OCCLUDED and OUT_OF_FRAME both drop out of the heatmap/NLL loss (soft-
    argmax always returns *some* in-frame coordinate, so there is no valid
    target to regress against for either), but they are NOT the same failure
    mode for the visibility head or for the downstream analytical tier: an
    occluded landmark's position is still physically defined, just hidden,
    while an out-of-frame landmark has no position at all in this photo. Both
    reasons matter differently even though the loss cannot currently express
    that distinction — see `training/metrics/localization.py`'s
    occluded-vs-out-of-frame confusion report (Build Prompt v2 §10.3).
    """

    CLEAR = 2
    AMBIGUOUS = 1
    OCCLUDED = 0
    OUT_OF_FRAME = -1


VISIBLE_FLAGS = frozenset({Visibility.CLEAR, Visibility.AMBIGUOUS})
MASKED_FLAGS = frozenset({Visibility.OCCLUDED, Visibility.OUT_OF_FRAME})
