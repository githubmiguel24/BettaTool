"""Heteroscedastic Gaussian NLL loss, with beta-NLL weighting and sigma clamping.

Build Prompt v2 §7:

    NLL = 0.5 * log(det Sigma) + 0.5 * (gt - mu)^T Sigma^-1 (gt - mu)

masked over keypoints with visibility 0 or -1, averaged over unmasked
keypoints then over the batch. Sigma's closed-form 2x2 inverse is used
instead of `torch.inverse` for speed and numerical control, per spec.
"""

from __future__ import annotations

import torch


def closed_form_2x2_inverse(cov: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form inverse and determinant of a batch of 2x2 matrices.

    Args:
        cov: (..., 2, 2) tensor.
        eps: added to the determinant before division, guarding against a
            near-singular matrix (should not occur given `CovarianceHead`'s
            construction, but this loss must never emit NaN even if a
            checkpoint from an earlier, less-safe head is loaded).

    Returns:
        inv: (..., 2, 2) inverse.
        det: (...,) determinant (unclamped — used directly in the log-det term).
    """
    a = cov[..., 0, 0]
    b = cov[..., 0, 1]
    c = cov[..., 1, 0]
    d = cov[..., 1, 1]
    det = a * d - b * c

    inv = torch.zeros_like(cov)
    inv_det = 1.0 / (det + eps)
    inv[..., 0, 0] = d * inv_det
    inv[..., 0, 1] = -b * inv_det
    inv[..., 1, 0] = -c * inv_det
    inv[..., 1, 1] = a * inv_det
    return inv, det


def gaussian_nll_loss(
    mu: torch.Tensor,
    cov: torch.Tensor,
    target: torch.Tensor,
    visibility_mask: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """Per-keypoint 2D Gaussian NLL, beta-NLL weighted and visibility-masked.

    Args:
        mu: (B, K, 2) predicted mean, CROP-SPACE pixels.
        cov: (B, K, 2, 2) predicted covariance, CROP-SPACE px^2.
        target: (B, K, 2) ground-truth keypoint location, CROP-SPACE pixels.
        visibility_mask: (B, K) float, 1.0 for flag 2/1, 0.0 for flag 0/-1.
        beta: beta-NLL exponent (Seitzer et al., 2022). Each keypoint's NLL
            term is scaled by `stop_gradient(mean(eigenvalues(Sigma)))^beta`
            (approximated here via `stop_gradient(0.5*trace(Sigma))^beta`,
            which equals the mean eigenvalue for a 2x2 matrix). `beta=0`
            recovers the unweighted NLL; `beta=1` is equivalent to
            unweighted MSE on the residual. Set from
            `loss.beta_nll` in config — 0.5 is the default and single most
            effective defense against variance collapse per the spec.

    Returns:
        Scalar loss, averaged over visible keypoints then over the batch.
        Returns 0.0 (not NaN) for a batch where every keypoint is masked.
    """
    inv_cov, det = closed_form_2x2_inverse(cov)

    residual = (target - mu).unsqueeze(-1)  # (B, K, 2, 1)
    mahalanobis_sq = (residual.transpose(-1, -2) @ inv_cov @ residual).squeeze(-1).squeeze(-1)  # (B, K)

    log_det = torch.log(det.clamp_min(1e-12))
    nll = 0.5 * log_det + 0.5 * mahalanobis_sq  # (B, K)

    if beta > 0:
        mean_eigenvalue = 0.5 * (cov[..., 0, 0] + cov[..., 1, 1])  # trace / 2, for a 2x2 matrix
        weight = mean_eigenvalue.detach().clamp_min(1e-8) ** beta
        nll = nll * weight

    masked = nll * visibility_mask
    denom = visibility_mask.sum(dim=-1).clamp_min(1e-8)
    per_sample = masked.sum(dim=-1) / denom
    return per_sample.mean()
