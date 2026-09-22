"""Heteroscedastic Gaussian NLL loss, with beta-NLL weighting and sigma clamping.

Build Prompt v2 §7:

    NLL = 0.5 * log(det Sigma) + 0.5 * (gt - mu)^T Sigma^-1 (gt - mu)

masked over keypoints with visibility 0 or -1, averaged over unmasked
keypoints then over the batch. Sigma's closed-form 2x2 inverse is used
instead of `torch.inverse` for speed and numerical control, per spec.

TWO NUMERICAL SAFEGUARDS, both load-bearing under AMP (`training.amp: true`):

1. MASK BEFORE SQUARING, not after. A masked keypoint (visibility 0/-1)
   carries a meaningless target -- Roboflow writes unlabeled landmarks as
   (0, 0) -- so its residual against a prediction near the crop centre is
   ~270px. Squared through the Mahalanobis form with inv_cov up to
   1/sigma_min^2 = 4, that is ~590,000, which OVERFLOWS fp16 (max 65,504)
   to `inf`. Multiplying that by a 0.0 mask afterwards gives `inf * 0 =
   NaN`, not 0 -- so keypoints that were supposed to contribute nothing
   instead poison the whole batch, every batch, and the model is NaN within
   one epoch. The residual is therefore zeroed with `torch.where` BEFORE it
   is ever squared, and the per-keypoint NLL is zeroed the same way.

2. COMPUTE IN FLOAT32. Even for visible keypoints, a large early-training
   residual squared against a small predicted sigma can exceed fp16 range.
   The whole loss is evaluated with autocast disabled and inputs upcast, so
   dynamic range is never the failure mode. This costs almost nothing (the
   loss is tiny next to the backbone) and the surrounding forward pass is
   still fp16.

Stage 1 does not exercise either path -- its localization term uses abs(),
not a square -- which is exactly why a NaN here shows up only at the
Stage 2 boundary, as soon as lambda_nll ramps above 0.
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
    # Safeguard 2: evaluate in float32 regardless of the surrounding autocast
    # context, so fp16's 65,504 ceiling is never the failure mode (see module
    # docstring).
    with torch.autocast(device_type=mu.device.type, enabled=False):
        mu = mu.float()
        cov = cov.float()
        target = target.float()
        visibility_mask = visibility_mask.float()

        visible = visibility_mask > 0  # (B, K) bool

        # Safeguard 1: zero the residual for masked keypoints BEFORE it is
        # squared. Multiplying by the mask afterwards would compute inf * 0
        # = NaN for exactly those keypoints (see module docstring).
        residual = target - mu  # (B, K, 2)
        residual = torch.where(visible.unsqueeze(-1), residual, torch.zeros_like(residual))
        residual = residual.unsqueeze(-1)  # (B, K, 2, 1)

        inv_cov, det = closed_form_2x2_inverse(cov)
        mahalanobis_sq = (residual.transpose(-1, -2) @ inv_cov @ residual).squeeze(-1).squeeze(-1)  # (B, K)

        log_det = torch.log(det.clamp_min(1e-12))
        nll = 0.5 * log_det + 0.5 * mahalanobis_sq  # (B, K)

        if beta > 0:
            mean_eigenvalue = 0.5 * (cov[..., 0, 0] + cov[..., 1, 1])  # trace / 2, for a 2x2 matrix
            weight = mean_eigenvalue.detach().clamp_min(1e-8) ** beta
            nll = nll * weight

        # Zero (not multiply) the masked entries, for the same reason as the
        # residual above: the log-det term is finite for a masked keypoint,
        # but this keeps the mask semantics exact and total.
        masked = torch.where(visible, nll, torch.zeros_like(nll))

        denom = visibility_mask.sum(dim=-1).clamp_min(1e-8)
        per_sample = masked.sum(dim=-1) / denom
        return per_sample.mean()