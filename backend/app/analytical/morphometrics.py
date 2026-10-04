# computes IBC morphometric ratios and angles from keypoint coordinates
from __future__ import annotations

import numpy as np

from app.perception.keypoints import Keypoint


def _point(x: np.ndarray, kp: Keypoint) -> np.ndarray:
    # extract x,y coords for a specific keypoint
    return x[2 * kp : 2 * kp + 2]


def _distance(x: np.ndarray, a: Keypoint, b: Keypoint) -> float:
    # euclidean distance between two keypoints
    return float(np.linalg.norm(_point(x, a) - _point(x, b)))


def _peduncle_mid(x: np.ndarray) -> np.ndarray:
    # midpoint of caudal peduncle for fin vertex origin
    return 0.5 * (_point(x, Keypoint.CAUDAL_PEDUNCLE_TOP) + _point(x, Keypoint.CAUDAL_PEDUNCLE_BOTTOM))


def _signed_angle_deg(axis: np.ndarray, v: np.ndarray) -> float:
    # signed angle in degrees from axis vector to v
    cross = axis[0] * v[1] - axis[1] * v[0]
    return float(np.degrees(np.arctan2(cross, float(np.dot(axis, v)))))


def caudal_spread_angle(x: np.ndarray) -> float:
    # caudal spread angle measured from peduncle midpoint to fin tips
    mid = _peduncle_mid(x)
    axis = _point(x, Keypoint.CAUDAL_FIN_CENTER) - mid  # the fin's mid-ray

    upper = _signed_angle_deg(axis, _point(x, Keypoint.CAUDAL_FIN_TIP_UPPER) - mid)
    lower = _signed_angle_deg(axis, _point(x, Keypoint.CAUDAL_FIN_TIP_LOWER) - mid)
    return abs(float(upper - lower))


def _body_length(x: np.ndarray) -> float:
    # snout tip to caudal peduncle midpoint distance
    return float(np.linalg.norm(_point(x, Keypoint.SNOUT_TIP) - _peduncle_mid(x)))


def _dorsal_base_mid(x: np.ndarray) -> np.ndarray:
    # midpoint of dorsal fin base
    return 0.5 * (
        _point(x, Keypoint.DORSAL_FIN_BASE_ANTERIOR) + _point(x, Keypoint.DORSAL_FIN_BASE_POSTERIOR)
    )


def _anal_base_mid(x: np.ndarray) -> np.ndarray:
    "# midpoint of anal fin base"
    return 0.5 * (
        _point(x, Keypoint.ANAL_FIN_BASE_ANTERIOR) + _point(x, Keypoint.ANAL_FIN_BASE_POSTERIOR)
    )


def _dorsal_length(x: np.ndarray) -> float:
    # dorsal base midpoint to dorsal fin tip
    return float(np.linalg.norm(_point(x, Keypoint.DORSAL_FIN_TIP) - _dorsal_base_mid(x)))


def _anal_length(x: np.ndarray) -> float:
    # anal base midpoint to anal fin tip
    return float(np.linalg.norm(_point(x, Keypoint.ANAL_FIN_TIP) - _anal_base_mid(x)))


def _caudal_length(x: np.ndarray) -> float:
    # caudal peduncle midpoint to caudal fin center
    return float(np.linalg.norm(_point(x, Keypoint.CAUDAL_FIN_CENTER) - _peduncle_mid(x)))


def dorsal_body_ratio(x: np.ndarray) -> float:
    # ratio of dorsal length to body length
    return _dorsal_length(x) / (_body_length(x) + 1e-12)


def anal_body_ratio(x: np.ndarray) -> float:
    # ratio of anal length to body length
    return _anal_length(x) / (_body_length(x) + 1e-12)


def caudal_body_ratio(x: np.ndarray) -> float:
    # ratio of caudal length to body length
    return _caudal_length(x) / (_body_length(x) + 1e-12)


def _anal_width(x: np.ndarray) -> float:
    # anal base front corner to rear corner
    return _distance(x, Keypoint.ANAL_FIN_BASE_ANTERIOR, Keypoint.ANAL_FIN_BASE_POSTERIOR)


def anal_length_width_ratio(x: np.ndarray) -> float:
    # ratio of anal length (base midpoint to tip) to anal width (front to rear corner)
    return _anal_length(x) / (_anal_width(x) + 1e-12)


# matches frontend Keys in measuremnts.js
MORPHOMETRIC_FUNCTIONS = {
    "caudal-spread-angle": caudal_spread_angle,
    "dorsal-body-ratio": dorsal_body_ratio,
    "anal-body-ratio": anal_body_ratio,
    "caudal-body-ratio": caudal_body_ratio,
    "anal-length-width-ratio": anal_length_width_ratio,
}