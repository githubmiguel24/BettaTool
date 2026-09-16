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

SIGMA_MIN_PX = 0.5
SIGMA_MAX_PX = 64.0
RHO_MAX = 0.95
COVARIANCE_EPS = 1e-3
HEATMAP_STRIDE = 4  


def build_backbone(source: str = "auto") -> tuple[nn.Module, int]:
    # bulds the hrNet w32 backbon. tries timm first then fals back to custom
    if source not in {"auto", "timm", "custom"}:
        raise ValueError(f"Unknown backbone_source: {source!r}")

    if source in ("auto", "timm"):
        try:
            import timm  # noqa: PLC0415
        except ImportError:
            if source == "timm":
                raise
            warnings.warn(
                "timm not instaled. falling back to custom hrnet without pretraining. "
                "u should rly install timm for better results.",
                stacklevel=2,
            )
        else:
            backbone = timm.create_model(
                "hrnet_w32", pretrained=True, features_only=True, out_indices=(4,)
            )
            out_channels = backbone.feature_info.channels()[-1]
            if out_channels != HRNetW32Backbone.out_channels:
                warnings.warn(
                    f"timm hrnet_w32 gave {out_channels} channels but we expected {HRNetW32Backbone.out_channels}. "
                    "shapes should be fine but double chek timm version",
                    stacklevel=2,
                )
            return _TimmFeatureWrapper(backbone), out_channels

    backbone = HRNetW32Backbone()
    return backbone, backbone.out_channels


class _TimmFeatureWrapper(nn.Module):
    # unwraps timm list output to just a singl tensor
    def __init__(self, timm_backbone: nn.Module) -> None:
        super().__init__()
        self.timm_backbone = timm_backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.timm_backbone(x)[-1]


class HeatmapHead(nn.Module):
    # 1x1 conv to get spatial softMax normalized heatmaps
    def __init__(self, in_ch: int, num_keypoints: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_keypoints, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.conv(features)
        b, k, h, w = raw.shape
        return torch.softmax(raw.view(b, k, -1), dim=-1).view(b, k, h, w)


class CovarianceHead(nn.Module):
    # predicts 2x2 covarianc per keypoint using global and local featurs
    def __init__(self, in_ch: int, num_keypoints: int, hidden_ch: int = 128) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # global and locl feats combined
        self.mlp = nn.Sequential(
            nn.Linear(in_ch * 2, hidden_ch),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_ch, 3), 
        )

    def forward(self, features: torch.Tensor, mu_heatmap_space: torch.Tensor) -> torch.Tensor:
        b, c, h, w = features.shape
        k = mu_heatmap_space.shape[1]

        global_feat = self.global_pool(features).view(b, 1, c).expand(b, k, c) 

        # sample feat map at mean
        norm_x = (mu_heatmap_space[..., 0] / max(w - 1, 1)) * 2 - 1
        norm_y = (mu_heatmap_space[..., 1] / max(h - 1, 1)) * 2 - 1
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2) 
        sampled = F.grid_sample(features, grid, align_corners=True, mode="bilinear", padding_mode="border")
        local_feat = sampled.squeeze(-1).permute(0, 2, 1) 

        combined = torch.cat([global_feat, local_feat], dim=-1) 
        params = self.mlp(combined) 

        sigma_x = self._clamped_sigma(params[..., 0])
        sigma_y = self._clamped_sigma(params[..., 1])
        rho = torch.tanh(params[..., 2]) * RHO_MAX

        # predict sigm in crop space directly
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
    # get visbility logits per keypont
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
        return self.mlp(pooled) 


class HRNetKeypointDetector(nn.Module):
    # Main hrnet keypoint moduel. returns heatmaps covs and visbility
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
        features = self.backbone(image)
        heatmaps = self.heatmap_head(features)

        mu_heatmap_space = soft_argmax(heatmaps) 
        mu_for_cov = mu_heatmap_space.detach() if self.detach_mu_for_covariance else mu_heatmap_space

        covariances = self.covariance_head(features, mu_for_cov)
        visibility_logits = self.visibility_head(features)

        return heatmaps, covariances, visibility_logits

    @staticmethod
    def mu_to_crop_space(mu_heatmap_space: torch.Tensor) -> torch.Tensor:
        # convrt soft argmax out to crop space
        return mu_heatmap_space * HEATMAP_STRIDE