from __future__ import annotations

import numpy as np

from app.perception.keypoints import Keypoint

# maps flat keypint vectors to the 6 ibc Criteria
def _point(x: np.ndarray, kp: Keypoint) -> np.ndarray:
    return x[2 * kp : 2 * kp + 2]


def _distance(x: np.ndarray, a: Keypoint, b: Keypoint) -> float:
    return float(np.linalg.norm(_point(x, a) - _point(x, b)))


def caudal_spread_angle(x: np.ndarray) -> float:
    # gets iner Angle for caudal spread in degrs
    vertex = _point(x, Keypoint.CAUDAL_FIN_CENTER)
    upper = _point(x, Keypoint.CAUDAL_FIN_TIP_UPPER)
    lower = _point(x, Keypoint.CAUDAL_FIN_TIP_LOWER)

    v_upper = upper - vertex
    v_lower = lower - vertex
    cos_angle = np.dot(v_upper, v_lower) / (
        np.linalg.norm(v_upper) * np.linalg.norm(v_lower) + 1e-12
    )
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _body_length(x: np.ndarray) -> float:
    return _distance(x, Keypoint.SNOUT_TIP, Keypoint.CAUDAL_PEDUNCLE_TOP)


def dorsal_body_ratio(x: np.ndarray) -> float:
    dorsal = _distance(x, Keypoint.DORSAL_FIN_BASE_ANTERIOR, Keypoint.DORSAL_FIN_TIP)
    return dorsal / (_body_length(x) + 1e-12)


def anal_body_ratio(x: np.ndarray) -> float:
    anal = _distance(x, Keypoint.ANAL_FIN_BASE_ANTERIOR, Keypoint.ANAL_FIN_TIP)
    return anal / (_body_length(x) + 1e-12)


def caudal_body_ratio(x: np.ndarray) -> float:
    caudal = _distance(x, Keypoint.CAUDAL_PEDUNCLE_TOP, Keypoint.CAUDAL_FIN_CENTER)
    return caudal / (_body_length(x) + 1e-12)


def anal_caudal_ratio(x: np.ndarray) -> float:
    anal = _distance(x, Keypoint.ANAL_FIN_BASE_ANTERIOR, Keypoint.ANAL_FIN_TIP)
    caudal = _distance(x, Keypoint.CAUDAL_PEDUNCLE_TOP, Keypoint.CAUDAL_FIN_CENTER)
    return anal / (caudal + 1e-12)


def dorsal_caudal_ratio(x: np.ndarray) -> float:
    dorsal = _distance(x, Keypoint.DORSAL_FIN_BASE_ANTERIOR, Keypoint.DORSAL_FIN_TIP)
    caudal = _distance(x, Keypoint.CAUDAL_PEDUNCLE_TOP, Keypoint.CAUDAL_FIN_CENTER)
    return dorsal / (caudal + 1e-12)


# matches frontend Keys in measuremnts.js
MORPHOMETRIC_FUNCTIONS = {
    "caudal-spread-angle": caudal_spread_angle,
    "dorsal-body-ratio": dorsal_body_ratio,
    "anal-body-ratio": anal_body_ratio,
    "caudal-body-ratio": caudal_body_ratio,
    "anal-caudal-ratio": anal_caudal_ratio,
    "dorsal-caudal-ratio": dorsal_caudal_ratio,
}