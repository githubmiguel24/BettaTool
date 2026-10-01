# keypoint defs and skeleton setup for the betta fish pose model

from enum import Enum, IntEnum


class Keypoint(IntEnum):  # array indices for each landmark
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

# short names used for yaml configs and exporting to coco
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


class KeypointGroup(str, Enum):  # body parts for UI color coding
    HEAD = "head"
    DORSAL_FIN = "dorsal_fin"
    CAUDAL_FIN = "caudal_fin"
    ANAL_FIN = "anal_fin"


# must match the order in Keypoint enum
KEYPOINT_GROUPS: list[str] = [
    KeypointGroup.HEAD,
    KeypointGroup.HEAD,
    KeypointGroup.DORSAL_FIN,
    KeypointGroup.DORSAL_FIN,
    KeypointGroup.DORSAL_FIN,
    KeypointGroup.CAUDAL_FIN,
    KeypointGroup.CAUDAL_FIN,
    KeypointGroup.CAUDAL_FIN,
    KeypointGroup.CAUDAL_FIN,
    KeypointGroup.CAUDAL_FIN,
    KeypointGroup.ANAL_FIN,
    KeypointGroup.ANAL_FIN,
    KeypointGroup.ANAL_FIN,
]
assert len(KEYPOINT_GROUPS) == NUM_KEYPOINTS, (
    f"KEYPOINT_GROUPS has {len(KEYPOINT_GROUPS)} entries but the schema "
    f"defines {NUM_KEYPOINTS} landmarks."
)

# point pairs to draw the skeleton in debug view
SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (5, 7),
    (6, 8), (7, 9), (8, 9), (6, 10), (10, 11), (11, 12), (0, 12),
]

# side view has no left or right symmetry so dont swap indices on flip
FLIP_MAP: list[int] = list(range(NUM_KEYPOINTS))
assert FLIP_MAP == list(range(NUM_KEYPOINTS)), "dont mess with this flip map"


class Visibility(IntEnum):  # custom flags to separate out of frame from occluded
    CLEAR = 2
    AMBIGUOUS = 1
    OCCLUDED = 0
    OUT_OF_FRAME = -1


# ignore hidden points when calculating loss
VISIBLE_FLAGS = frozenset({Visibility.CLEAR, Visibility.AMBIGUOUS})
MASKED_FLAGS = frozenset({Visibility.OCCLUDED, Visibility.OUT_OF_FRAME})