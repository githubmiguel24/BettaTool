# =============================================================================
# Program Title : morphometrics.py - IBC Morphometric Functions
# Programmers   : Trinidad, C. M. L.; Daanay, K. M. M.; Verceles, R. C. S.
# System Fit    : Betta Fish Quality Assessment Model > Analytical Tier.
#                 app/pipeline.py loops over MORPHOMETRIC_FUNCTIONS and, for each
#                 criterion, passes the 26-value keypoint vector from the
#                 Perceptual Tier to the matching function. The criterion keys
#                 are the same keys used by the API and the frontend.
# Written       : September 10, 2026
# Revised       : October 4, 2026 (last saved);
#                 October 14, 2026 (header block and comments for documentation)
# Purpose       : Computes the five measurements graded under the IBC standards:
#                 the caudal spread angle, the dorsal, anal and caudal
#                 fin-to-body length ratios, and the anal fin length-to-width
#                 ratio.
# Data / Algo   : Data: flat NumPy vector x of 26 values (13 keypoints x 2);
#                 Keypoint enum used as the index; dictionary
#                 MORPHOMETRIC_FUNCTIONS (criterion key -> function).
#                 Algorithms: Euclidean distance; midpoints of the caudal
#                 peduncle and the fin bases; signed angle with
#                 atan2(cross, dot). Caudal spread = |angle of the upper tip -
#                 angle of the lower tip|, both measured from the fin's mid-ray.
#                 Control: sequence of small helper functions; no loops or
#                 branches.
# =============================================================================

from __future__ import annotations

import numpy as np

from app.perception.keypoints import Keypoint

# Added to every divisor so a degenerate (zero-length) measurement cannot
# cause a division by zero.
_EPS = 1e-12


def _point(x: np.ndarray, kp: Keypoint) -> np.ndarray:
    """Return the (x, y) coordinates of one keypoint."""
    return x[2 * kp : 2 * kp + 2]


def _distance(x: np.ndarray, a: Keypoint, b: Keypoint) -> float:
    """Return the Euclidean distance between two keypoints."""
    return float(np.linalg.norm(_point(x, a) - _point(x, b)))


def _peduncle_mid(x: np.ndarray) -> np.ndarray:
    """Return the midpoint of the caudal peduncle (origin of the caudal fin)."""
    top = _point(x, Keypoint.CAUDAL_PEDUNCLE_TOP)
    bottom = _point(x, Keypoint.CAUDAL_PEDUNCLE_BOTTOM)
    return 0.5 * (top + bottom)


def _signed_angle_deg(axis: np.ndarray, v: np.ndarray) -> float:
    """Return the signed angle in degrees from the axis vector to v."""
    cross = axis[0] * v[1] - axis[1] * v[0]
    return float(np.degrees(np.arctan2(cross, float(np.dot(axis, v)))))


def caudal_spread_angle(x: np.ndarray) -> float:
    """Return the caudal spread angle in degrees, from the peduncle midpoint to the fin tips."""
    mid = _peduncle_mid(x)
    axis = _point(x, Keypoint.CAUDAL_FIN_CENTER) - mid  # the fin's mid-ray

    upper = _signed_angle_deg(axis, _point(x, Keypoint.CAUDAL_FIN_TIP_UPPER) - mid)
    lower = _signed_angle_deg(axis, _point(x, Keypoint.CAUDAL_FIN_TIP_LOWER) - mid)
    return abs(float(upper - lower))


def _body_length(x: np.ndarray) -> float:
    """Return the distance from the snout tip to the peduncle midpoint."""
    return float(np.linalg.norm(_point(x, Keypoint.SNOUT_TIP) - _peduncle_mid(x)))


def _dorsal_base_mid(x: np.ndarray) -> np.ndarray:
    """Return the midpoint of the dorsal fin base."""
    front = _point(x, Keypoint.DORSAL_FIN_BASE_ANTERIOR)
    rear = _point(x, Keypoint.DORSAL_FIN_BASE_POSTERIOR)
    return 0.5 * (front + rear)


def _anal_base_mid(x: np.ndarray) -> np.ndarray:
    """Return the midpoint of the anal fin base."""
    front = _point(x, Keypoint.ANAL_FIN_BASE_ANTERIOR)
    rear = _point(x, Keypoint.ANAL_FIN_BASE_POSTERIOR)
    return 0.5 * (front + rear)


def _dorsal_length(x: np.ndarray) -> float:
    """Return the distance from the dorsal base midpoint to the dorsal fin tip."""
    return float(np.linalg.norm(_point(x, Keypoint.DORSAL_FIN_TIP) - _dorsal_base_mid(x)))


def _anal_length(x: np.ndarray) -> float:
    """Return the distance from the anal base midpoint to the anal fin tip."""
    return float(np.linalg.norm(_point(x, Keypoint.ANAL_FIN_TIP) - _anal_base_mid(x)))


def _caudal_length(x: np.ndarray) -> float:
    """Return the distance from the peduncle midpoint to the caudal fin center."""
    return float(np.linalg.norm(_point(x, Keypoint.CAUDAL_FIN_CENTER) - _peduncle_mid(x)))


def dorsal_body_ratio(x: np.ndarray) -> float:
    """Return the dorsal fin length divided by the body length."""
    return _dorsal_length(x) / (_body_length(x) + _EPS)


def anal_body_ratio(x: np.ndarray) -> float:
    """Return the anal fin length divided by the body length."""
    return _anal_length(x) / (_body_length(x) + _EPS)


def caudal_body_ratio(x: np.ndarray) -> float:
    """Return the caudal fin length divided by the body length."""
    return _caudal_length(x) / (_body_length(x) + _EPS)


def _anal_width(x: np.ndarray) -> float:
    """Return the anal fin width: front corner of the base to its rear corner."""
    return _distance(x, Keypoint.ANAL_FIN_BASE_ANTERIOR, Keypoint.ANAL_FIN_BASE_POSTERIOR)


def anal_length_width_ratio(x: np.ndarray) -> float:
    """Return the anal fin length (base midpoint to tip) divided by its width."""
    return _anal_length(x) / (_anal_width(x) + _EPS)


# Lookup table used by app/pipeline.py. The keys must match the criterion keys
# used by the frontend (measurements.js) and by ibc_standards.py.
MORPHOMETRIC_FUNCTIONS = {
    "caudal-spread-angle": caudal_spread_angle,
    "dorsal-body-ratio": dorsal_body_ratio,
    "anal-body-ratio": anal_body_ratio,
    "caudal-body-ratio": caudal_body_ratio,
    "anal-length-width-ratio": anal_length_width_ratio,
}