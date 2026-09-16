"""Keypoint mappings for the betta fish model."""

from enum import IntEnum


class Keypoint(IntEnum):
    # indices for the keypoint array

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

# short labels for yaml configs and coco export
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

# connection lines for debug viz
SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (5, 7),
    (6, 8), (7, 9), (8, 9), (6, 10), (10, 11), (11, 12), (0, 12),
]

# identity map bc lateral view has no left/right pairs, do NOT swap indices
FLIP_MAP: list[int] = list(range(NUM_KEYPOINTS))
assert FLIP_MAP == list(range(NUM_KEYPOINTS)), "dont mess with this flip map"


class Visibility(IntEnum):
    # custom vis flags so we can tell oof vs occluded apart

    CLEAR = 2
    AMBIGUOUS = 1
    OCCLUDED = 0
    OUT_OF_FRAME = -1


# mask out stuff that isnt actually visible in the loss
VISIBLE_FLAGS = frozenset({Visibility.CLEAR, Visibility.AMBIGUOUS})
MASKED_FLAGS = frozenset({Visibility.OCCLUDED, Visibility.OUT_OF_FRAME})