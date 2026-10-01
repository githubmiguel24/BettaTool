# hrnet-w32 backbone wrapper using timm with imagenet pretrained weights
# outputs a fused 480 channel feature map at stride 4 (H/4, W/4)

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn


class HRNetW32Backbone(nn.Module):
    # wraps timm hrnet_w32 and fuses the 4 stage4 branches into (B, 480, H/4, W/4)

    out_channels = 480  # 32 + 64 + 128 + 256

    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        # load hrnet_w32 from timm and strip unused classification head layers
        super().__init__()
        self.backbone = timm.create_model(
            "hrnet_w32",
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
        )

        # remove timm classification neck so we dont carry dead weights in memory
        for attr in ("incre_modules", "downsamp_modules", "final_layer", "classifier"):
            if hasattr(self.backbone, attr):
                delattr(self.backbone, attr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # run stem and multi-scale stages then upsample all branches to H/4 and concat
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)

        # stages() runs layer1 and stages 2-4, returning [H/4, H/8, H/16, H/32] tensors
        xs = self.backbone.stages(x)

        target_hw = xs[0].shape[-2:]
        upsampled = [xs[0]] + [
            F.interpolate(branch, size=target_hw, mode="bilinear", align_corners=False)
            for branch in xs[1:]
        ]
        return torch.cat(upsampled, dim=1)