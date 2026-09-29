"""Sanity checks for the corrected, covariance-consistent decision rule.

See app/analytical/tsi.py's `is_confident_gum` docstring for why this
replaced `is_confident_isotropic` (still covered by test_tsi.py, kept only
as the ablation baseline).
"""

import numpy as np

from app.analytical.gum_propagation import combined_uncertainty, expanded_uncertainty
from app.analytical.tsi import is_confident_gum


def test_confident_when_margin_clears_uncertainty():
    # measurement far from threshold, small uncertainty -> confident
    assert is_confident_gum(measurement=150.0, threshold=180.0, expanded_uncertainty=5.0) is True


def test_defers_when_uncertainty_reaches_threshold():
    # measurement close to threshold, uncertainty reaches it -> defer
    assert is_confident_gum(measurement=179.0, threshold=180.0, expanded_uncertainty=2.0) is False


def test_boundary_is_inclusive_defer():
    # margin exactly equal to expanded uncertainty -> defer (>=), not a
    # silent pass -- ties should not be resolved in the model's favor.
    assert is_confident_gum(measurement=178.0, threshold=180.0, expanded_uncertainty=2.0) is False


def test_uses_real_anisotropic_covariance_not_identity():
    """The whole point of the fix: a criterion whose Jacobian only touches
    keypoints with LOW predicted variance should be confident even if OTHER
    keypoints elsewhere in the image are poorly localized -- because
    combined_uncertainty only sees the keypoints the Jacobian is nonzero
    for. The old actual_rmse (blended across all 13 keypoints) could not
    tell these two cases apart.
    """
    # Jacobian only depends on keypoint 0 (2 coords); keypoint 1 is
    # unrelated to this criterion and has a huge, unrelated variance.
    jacobian = np.array([1.0, 0.0, 0.0, 0.0])
    good_kp_cov = np.array([[0.25, 0.0], [0.0, 0.25]])   # sigma=0.5px on the relevant keypoint
    bad_unrelated_cov = np.array([[400.0, 0.0], [0.0, 400.0]])  # sigma=20px on an irrelevant one
    block_covariance = np.zeros((4, 4))
    block_covariance[0:2, 0:2] = good_kp_cov
    block_covariance[2:4, 2:4] = bad_unrelated_cov

    u_c = combined_uncertainty(jacobian, block_covariance)
    uncertainty = expanded_uncertainty(u_c)

    # Despite the unrelated keypoint's huge variance, the propagated
    # uncertainty for THIS criterion stays small, because the Jacobian
    # zeroes it out of J Sigma J^T.
    assert uncertainty < 2.0
    assert is_confident_gum(measurement=150.0, threshold=180.0, expanded_uncertainty=uncertainty) is True
