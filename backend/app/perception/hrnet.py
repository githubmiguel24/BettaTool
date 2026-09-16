"""Probabilistic keypoint detector: HRNet-W32 + heatmap/covariance/visibility heads.

Implements Build Prompt v2 §6-§9.1's forward contract exactly:

    heatmaps, covariances, visibility = HRNetKeypointDetector()(image)
    # heatmaps    : (B, 13, 96, 96)   spatial-softmax normalized
    # covariances : (B, 13, 2, 2)     positive-definite, crop-space px^2 units
    # visibility  : (B, 13)           sigmoid logits (NOT probabilities — see below)

`visibility` is returned as raw logits, not probabilities, so a caller can
choose `torch.sigmoid(visibility)` for a probability or feed the logits
directly into `training/losses/visibility_bce.py`'s
`nn.functional.binary_cross_entropy_with_logits` (numerically preferable to
applying sigmoid then plain BCE). `app/pipeline.py` applies the sigmoid.

Positive-definiteness of `covariances` is guaranteed BY CONSTRUCTION via the
softplus/tanh parameterization in `CovarianceHead` — never regress raw
matrix entries (Build Prompt v2 §6). `tests/test_model.py` asserts
`det(Sigma) > 0` and symmetry on random outputs.
"""

from __future__ import annotations

import logging
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from app.perception.geometry import AffineTransform
from app.perception.heatmap import soft_argmax
from app.perception.hrnet_backbone import HRNetW32Backbone
from app.perception.keypoints import NUM_KEYPOINTS

logger = logging.getLogger(__name__)

# --- units contract (Build Prompt v2 §4.4 / training/configs/base.yaml `units`) ---
SIGMA_MIN_PX = 0.5
SIGMA_MAX_PX = 64.0
RHO_MAX = 0.95
COVARIANCE_EPS = 1e-3
HEATMAP_STRIDE = 4  # crop_size / heatmap_size, e.g. 384 / 96


def build_backbone(source: str = "auto") -> tuple[nn.Module, int]:
    """Constructs the HRNet-W32 backbone and returns `(module, out_channels)`.

    Build Prompt v2 §2 locks the ImageNet-pretrained `timm` `hrnet_w32` as the
    backbone. This function honors that by preferring `timm` whenever it is
    importable, and otherwise falls back to the self-contained
    `app/perception/hrnet_backbone.HRNetW32Backbone` (architecturally
    equivalent, but randomly initialized — no ImageNet pretraining).

    Args:
        source: "timm" forces the timm path (raises if timm is unavailable),
            "custom" forces the self-contained path, "auto" tries timm first
            and falls back to custom with a logged warning. Set from
            `model.backbone_source` in config — never hardcode this choice
            at a call site.

    Returns:
        (backbone_module, out_channels). `out_channels` is 480 for W32
        either way — `timm.create_model(..., features_only=True)`'s last
        feature map is at 480 channels @ stride 4 for hrnet_w32, matching
        the custom implementation exactly.
    """
    if source not in {"auto", "timm", "custom"}:
        raise ValueError(f"Unknown backbone_source: {source!r}")

    if source in ("auto", "timm"):
        try:
            import timm  # noqa: PLC0415 (intentionally optional/deferred import)
        except ImportError:
            if source == "timm":
                raise
            warnings.warn(
                "timm is not installed; falling back to the self-contained, "
                "randomly-initialized HRNetW32Backbone (no ImageNet pretraining). "
                "Install `timm` and re-run for a pretrained backbone, which "
                "matters a great deal at ~1,750 training images "
                "(Build Prompt v2 open question §13.1). Set model.backbone_source: "
                "'timm' in config to fail loudly instead of silently falling back.",
                stacklevel=2,
            )
        else:
            backbone = timm.create_model(
                "hrnet_w32", pretrained=True, features_only=True, out_indices=(4,)
            )
            out_channels = backbone.feature_info.channels()[-1]
            if out_channels != HRNetW32Backbone.out_channels:
                warnings.warn(
                    f"timm hrnet_w32 reports {out_channels} output channels; "
                    f"expected {HRNetW32Backbone.out_channels}. Heads are sized "
                    "from this value either way, so shapes stay consistent, but "
                    "double-check the timm version if this surprises you.",
                    stacklevel=2,
                )
            return _TimmFeatureWrapper(backbone), out_channels

    backbone = HRNetW32Backbone()
    return backbone, backbone.out_channels


class _TimmFeatureWrapper(nn.Module):
    """Unwraps timm's `features_only=True` list-of-one-tensor output to a plain tensor."""

    def __init__(self, timm_backbone: nn.Module) -> None:
        super().__init__()
        self.timm_backbone = timm_backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.timm_backbone(x)[-1]


class HeatmapHead(nn.Module):
    """1x1 conv -> per-keypoint heatmap, spatial-softmax normalized."""

    def __init__(self, in_ch: int, num_keypoints: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_keypoints, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.conv(features)
        b, k, h, w = raw.shape
        return torch.softmax(raw.view(b, k, -1), dim=-1).view(b, k, h, w)


class CovarianceHead(nn.Module):
    """Per-keypoint 2x2 positive-definite Sigma from a global feature + the
    local feature sampled at that keypoint's predicted mean (Build Prompt v2 §6).

    Sigma = [[sigma_x^2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y^2]]
    with sigma_x, sigma_y = softplus(.) + eps (clamped to [SIGMA_MIN_PX,
    SIGMA_MAX_PX]) and rho = tanh(.) * RHO_MAX. This guarantees det(Sigma) > 0
    by construction — softplus makes the diagonal strictly positive, and
    |rho| < 1 makes the determinant sigma_x^2*sigma_y^2*(1-rho^2) strictly
    positive too.
    """

    def __init__(self, in_ch: int, num_keypoints: int, hidden_ch: int = 128) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Global context (in_ch) + per-keypoint local feature sampled at mu (in_ch).
        self.mlp = nn.Sequential(
            nn.Linear(in_ch * 2, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, 3),  # (s_x, s_y, r) per keypoint, applied per-keypoint below
        )

    def forward(self, features: torch.Tensor, mu_heatmap_space: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) fused backbone feature map.
            mu_heatmap_space: (B, K, 2) (x, y) coordinates in [0, W-1] x
                [0, H-1] heatmap-space pixels (i.e. `soft_argmax`'s raw
                output, BEFORE multiplying by stride). Whether this carries
                gradient depends on `detach_mu_for_covariance` upstream
                (Build Prompt v2 §6) — this function is agnostic to that.

        Returns:
            (B, K, 2, 2) covariance matrices, in CROP-SPACE pixel units
            (i.e. already scaled by `HEATMAP_STRIDE`; see units contract).
        """
        b, c, h, w = features.shape
        k = mu_heatmap_space.shape[1]

        global_feat = self.global_pool(features).view(b, 1, c).expand(b, k, c)  # (B, K, C)

        # Bilinear-sample the feature map at each keypoint's predicted mean.
        # grid_sample expects normalized [-1, 1] coords, shape (B, K, 1, 2).
        norm_x = (mu_heatmap_space[..., 0] / max(w - 1, 1)) * 2 - 1
        norm_y = (mu_heatmap_space[..., 1] / max(h - 1, 1)) * 2 - 1
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)  # (B, K, 1, 2)
        sampled = F.grid_sample(features, grid, align_corners=True, mode="bilinear", padding_mode="border")
        local_feat = sampled.squeeze(-1).permute(0, 2, 1)  # (B, K, C)

        combined = torch.cat([global_feat, local_feat], dim=-1)  # (B, K, 2C)
        params = self.mlp(combined)  # (B, K, 3)

        sigma_x = self._clamped_sigma(params[..., 0])
        sigma_y = self._clamped_sigma(params[..., 1])
        rho = torch.tanh(params[..., 2]) * RHO_MAX

        # Sigma is predicted directly in crop-space px, per the units contract:
        # sigma clamps [SIGMA_MIN_PX, SIGMA_MAX_PX] are crop-space (384-space) values.
        cov = torch.zeros(b, k, 2, 2, device=params.device, dtype=params.dtype)
        cov[..., 0, 0] = sigma_x**2
        cov[..., 1, 1] = sigma_y**2
        off_diag = rho * sigma_x * sigma_y
        cov[..., 0, 1] = off_diag
        cov[..., 1, 0] = off_diag
        return cov

    @staticmethod
    def _clamped_sigma(raw: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(raw) + COVARIANCE_EPS
        return sigma.clamp(min=SIGMA_MIN_PX, max=SIGMA_MAX_PX)


class VisibilityHead(nn.Module):
    """Global-context MLP -> per-keypoint visibility logit."""

    def __init__(self, in_ch: int, num_keypoints: int, hidden_ch: int = 128) -> None:
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, num_keypoints),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = features.shape
        pooled = self.global_pool(features).view(b, c)
        return self.mlp(pooled)  # (B, K) logits


class HRNetKeypointDetector(nn.Module):
    """Full probabilistic keypoint model: HRNet-W32 backbone -> three heads.

    `forward` ALWAYS returns the 3-tuple `(heatmaps, covariances,
    visibility)` described in the module docstring — Build Prompt v2 §4.3
    explicitly says not to silently drop the visibility output. Callers
    written against the old 2-tuple contract (e.g. any external code that
    still does `heatmaps, covariances = model(x)`) will get a clear
    "too many values to unpack" TypeError rather than silently discarding
    the visibility head; `app/pipeline.py` has been updated to consume all
    three (see its module docstring for the migration note).
    """

    def __init__(
        self,
        num_keypoints: int = NUM_KEYPOINTS,
        backbone_source: str = "auto",
        detach_mu_for_covariance: bool = True,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.detach_mu_for_covariance = detach_mu_for_covariance

        self.backbone, feat_ch = build_backbone(backbone_source)
        self.heatmap_head = HeatmapHead(feat_ch, num_keypoints)
        self.covariance_head = CovarianceHead(feat_ch, num_keypoints)
        self.visibility_head = VisibilityHead(feat_ch, num_keypoints)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (B, 3, 384, 384) normalized crop-space image tensor.

        Returns:
            heatmaps    (B, K, 96, 96)  spatial-softmax normalized.
            covariances (B, K, 2, 2)    positive-definite, CROP-SPACE px^2.
            visibility  (B, K)          sigmoid LOGITS (apply torch.sigmoid
                                        for a probability).
        """
        features = self.backbone(image)
        heatmaps = self.heatmap_head(features)

        mu_heatmap_space = soft_argmax(heatmaps)  # (B, K, 2), heatmap-space px
        mu_for_cov = mu_heatmap_space.detach() if self.detach_mu_for_covariance else mu_heatmap_space

        covariances = self.covariance_head(features, mu_for_cov)
        visibility_logits = self.visibility_head(features)

        return heatmaps, covariances, visibility_logits

    @staticmethod
    def mu_to_crop_space(mu_heatmap_space: torch.Tensor) -> torch.Tensor:
        """Converts `soft_argmax` output (heatmap-space) to crop-space pixels."""
        return mu_heatmap_space * HEATMAP_STRIDE
