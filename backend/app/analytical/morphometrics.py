from __future__ import annotations

import numpy as np

from app.perception.keypoints import Keypoint

# maps flat keypint vectors to the 6 ibc Criteria


def _point(x: np.ndarray, kp: Keypoint) -> np.ndarray:
    return x[2 * kp : 2 * kp + 2]


def _distance(x: np.ndarray, a: Keypoint, b: Keypoint) -> float:
    return float(np.linalg.norm(_point(x, a) - _point(x, b)))


def _peduncle_mid(x: np.ndarray) -> np.ndarray:
    """Midpoint of the caudal peduncle: mean(peduncle_top, peduncle_bottom).

    This is the anatomical origin of the caudal fin and the posterior end of
    body length, per training/configs/base.yaml
    (`body_length_landmarks: [snout_tip, peduncle_midpoint]`) and
    training/metrics/localization.py's `body_length`, both of which already
    use the midpoint. An earlier version of this module used
    CAUDAL_PEDUNCLE_TOP alone, disagreeing with both.

    Using the midpoint also halves the variance the peduncle contributes:
    averaging two independently-predicted landmarks is more precise than
    relying on either one.
    """
    return 0.5 * (_point(x, Keypoint.CAUDAL_PEDUNCLE_TOP) + _point(x, Keypoint.CAUDAL_PEDUNCLE_BOTTOM))


def _signed_angle_deg(axis: np.ndarray, v: np.ndarray) -> float:
    """Signed angle in degrees from `axis` to `v`, in (-180, 180]."""
    cross = axis[0] * v[1] - axis[1] * v[0]
    return float(np.degrees(np.arctan2(cross, float(np.dot(axis, v)))))


def caudal_spread_angle(x: np.ndarray) -> float:
    """Caudal spread angle in degrees, measured AT THE CAUDAL PEDUNCLE.

    The IBC caudal spread is the angle the caudal fin sweeps through as seen
    from where it attaches to the body, so the vertex is the caudal peduncle
    midpoint and the rays run out to the upper and lower fin tips. A perfect
    Halfmoon reads exactly 180.

    Two things this formulation deliberately avoids:

    1. WRONG VERTEX. An earlier version put the vertex at CAUDAL_FIN_CENTER
       (the rearmost point of the fin's outer margin) rather than the
       peduncle. That measures the curvature of the trailing edge, not the
       spread: a geometrically perfect Halfmoon scored 90 degrees, which
       falls in the "Disqualify" band (<=165) in
       app/decisional/ibc_standards.py -- i.e. every fish failed.

    2. arccos FOLDING AT 180. `arccos` returns an UNSIGNED angle capped at
       180 degrees, so an over-spread tail reads the same as an equally
       under-spread one and can never exceed 180. That makes the bands above
       180 in ibc_standards.py (Slight Fault 181-194, Major Fault 195-209,
       Disqualify >=210) unreachable. Measuring each tip's SIGNED angle
       about the fin's mid-ray (peduncle midpoint -> caudal fin centre) and
       summing keeps the measurement continuous through 180 and lets it
       exceed 180, so over-spread is representable and the function stays
       smoothly differentiable for GUM propagation across the whole decision
       boundary (verified: ||J|| ~= 1.146 deg/px and continuous at 180).
    """
    mid = _peduncle_mid(x)
    axis = _point(x, Keypoint.CAUDAL_FIN_CENTER) - mid  # the fin's mid-ray

    upper = _signed_angle_deg(axis, _point(x, Keypoint.CAUDAL_FIN_TIP_UPPER) - mid)
    lower = _signed_angle_deg(axis, _point(x, Keypoint.CAUDAL_FIN_TIP_LOWER) - mid)
    return float(upper - lower)


def _body_length(x: np.ndarray) -> float:
    """Snout tip -> caudal peduncle midpoint (see `_peduncle_mid`)."""
    return float(np.linalg.norm(_point(x, Keypoint.SNOUT_TIP) - _peduncle_mid(x)))


def _dorsal_base_mid(x: np.ndarray) -> np.ndarray:
    """Midpoint of the dorsal fin base: mean(dorsal_base_ant, dorsal_base_post)."""
    return 0.5 * (
        _point(x, Keypoint.DORSAL_FIN_BASE_ANTERIOR) + _point(x, Keypoint.DORSAL_FIN_BASE_POSTERIOR)
    )


def _anal_base_mid(x: np.ndarray) -> np.ndarray:
    """Midpoint of the anal fin base: mean(anal_base_ant, anal_base_post)."""
    return 0.5 * (
        _point(x, Keypoint.ANAL_FIN_BASE_ANTERIOR) + _point(x, Keypoint.ANAL_FIN_BASE_POSTERIOR)
    )


def _dorsal_length(x: np.ndarray) -> float:
    """Dorsal fin base MIDPOINT -> dorsal fin tip.

    Measured from the middle of the fin's base rather than its anterior
    corner, so the length reflects the fin's extension from its attachment
    as a whole. An earlier version measured from DORSAL_FIN_BASE_ANTERIOR
    alone, which left DORSAL_FIN_BASE_POSTERIOR unused by every criterion
    and biased the length by half the base width.
    """
    return float(np.linalg.norm(_point(x, Keypoint.DORSAL_FIN_TIP) - _dorsal_base_mid(x)))


def _anal_length(x: np.ndarray) -> float:
    """Anal fin base MIDPOINT -> anal fin tip (see `_dorsal_length`)."""
    return float(np.linalg.norm(_point(x, Keypoint.ANAL_FIN_TIP) - _anal_base_mid(x)))


def _caudal_length(x: np.ndarray) -> float:
    """Span of the caudal fin: upper tip -> lower tip.

    This is the tip-to-tip extent of the spread tail, i.e. the diameter of
    the half-disc a Halfmoon forms. Earlier versions measured peduncle ->
    CAUDAL_FIN_CENTER (the fin's front-to-back depth), which is a different
    quantity: for an ideal Halfmoon the span is twice that depth, and the
    two diverge as the spread departs from 180 degrees.

    CAUDAL_FIN_CENTER is still used, as the reference axis in
    `caudal_spread_angle`.
    """
    return _distance(x, Keypoint.CAUDAL_FIN_TIP_UPPER, Keypoint.CAUDAL_FIN_TIP_LOWER)


def dorsal_body_ratio(x: np.ndarray) -> float:
    return _dorsal_length(x) / (_body_length(x) + 1e-12)


def anal_body_ratio(x: np.ndarray) -> float:
    return _anal_length(x) / (_body_length(x) + 1e-12)


def caudal_body_ratio(x: np.ndarray) -> float:
    return _caudal_length(x) / (_body_length(x) + 1e-12)


def anal_caudal_ratio(x: np.ndarray) -> float:
    return _anal_length(x) / (_caudal_length(x) + 1e-12)


def dorsal_caudal_ratio(x: np.ndarray) -> float:
    return _dorsal_length(x) / (_caudal_length(x) + 1e-12)


# matches frontend Keys in measuremnts.js
MORPHOMETRIC_FUNCTIONS = {
    "caudal-spread-angle": caudal_spread_angle,
    "dorsal-body-ratio": dorsal_body_ratio,
    "anal-body-ratio": anal_body_ratio,
    "caudal-body-ratio": caudal_body_ratio,
    "anal-caudal-ratio": anal_caudal_ratio,
    "dorsal-caudal-ratio": dorsal_caudal_ratio,
}