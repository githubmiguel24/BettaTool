# standalone hrnet-w32 backbone in plain pytorch for offline testing and fallback when timm isnt availble
# outputs a fused 480 channel feature map at stride 4 (H/4, W/4) matching the standard w32 config

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_STAGE_CHANNELS: dict[str, list[int]] = {
    "stage2": [32, 64],
    "stage3": [32, 64, 128],
    "stage4": [32, 64, 128, 256],
}
_STAGE_NUM_MODULES: dict[str, int] = {"stage2": 1, "stage3": 4, "stage4": 3}
_STAGE_NUM_BLOCKS: dict[str, int] = {"stage2": 4, "stage3": 4, "stage4": 4}  # blocks per branch per module
_BN_MOMENTUM = 0.1


class BasicBlock(nn.Module):
    # standard 2x 3x3 conv residual block with expansion 1

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        # init conv layers and optional 1x1 downsample shortcut
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch, momentum=_BN_MOMENTUM),
            )
            if stride != 1 or in_ch != out_ch
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply convs and add residual connection
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    # 1x1 -> 3x3 -> 1x1 residual block with expansion 4 used only in stage 1

    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1) -> None:
        # setup bottleneck convs and projection shortcut if dims change
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch, momentum=_BN_MOMENTUM)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch, momentum=_BN_MOMENTUM)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch, momentum=_BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch, momentum=_BN_MOMENTUM),
            )
            if stride != 1 or in_ch != out_ch
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # run bottleneck forward pass
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + residual)


def _make_branch(in_ch: int, out_ch: int, num_blocks: int) -> nn.Sequential:
    # stack basic blocks for a single resolution branch
    layers = [BasicBlock(in_ch, out_ch)]
    layers += [BasicBlock(out_ch, out_ch) for _ in range(num_blocks - 1)]
    return nn.Sequential(*layers)


class HighResolutionModule(nn.Module):
    # runs parallel conv branches and fuses multi-scale features across resolutions

    def __init__(self, num_branches: int, channels: list[int], num_blocks: int) -> None:
        # create per-branch blocks and cross-branch Fusion layers
        super().__init__()
        if len(channels) != num_branches:
            raise ValueError("channels must have one entry per branch")
        self.num_branches = num_branches
        self.branches = nn.ModuleList(
            [_make_branch(channels[i], channels[i], num_blocks) for i in range(num_branches)]
        )
        self.fuse_layers = nn.ModuleList(
            [self._make_fuse_row(i, channels) for i in range(num_branches)]
        )
        self.relu = nn.ReLU(inplace=True)

    def _make_fuse_row(self, target_idx: int, channels: list[int]) -> nn.ModuleList:
        # builds fusion ops to align all source branches to target_idx resolution
        row = nn.ModuleList()
        for src_idx in range(self.num_branches):
            if src_idx == target_idx:
                row.append(nn.Identity())
            elif src_idx > target_idx:
                # lower res source gets 1x1 conv here and bilinear upsample in forward
                row.append(
                    nn.Sequential(
                        nn.Conv2d(channels[src_idx], channels[target_idx], 1, bias=False),
                        nn.BatchNorm2d(channels[target_idx], momentum=_BN_MOMENTUM),
                    )
                )
            else:
                # higher res source downsamples to target res using stride 2 convs
                steps = []
                cur_ch = channels[src_idx]
                num_steps = target_idx - src_idx
                for step in range(num_steps):
                    out_ch = channels[target_idx] if step == num_steps - 1 else cur_ch
                    steps.append(nn.Conv2d(cur_ch, out_ch, 3, stride=2, padding=1, bias=False))
                    steps.append(nn.BatchNorm2d(out_ch, momentum=_BN_MOMENTUM))
                    if step != num_steps - 1:
                        steps.append(nn.ReLU(inplace=True))
                    cur_ch = out_ch
                row.append(nn.Sequential(*steps))
        return row

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        # run each branch then sum and fuse all resolutions together
        branch_outs = [self.branches[i](xs[i]) for i in range(self.num_branches)]
        fused: list[torch.Tensor] = []
        for target_idx in range(self.num_branches):
            target_hw = branch_outs[target_idx].shape[-2:]
            acc = None
            for src_idx in range(self.num_branches):
                contribution = self.fuse_layers[target_idx][src_idx](branch_outs[src_idx])
                if contribution.shape[-2:] != target_hw:
                    contribution = F.interpolate(contribution, size=target_hw, mode="bilinear", align_corners=False)
                acc = contribution if acc is None else acc + contribution
            fused.append(self.relu(acc))
        return fused


def _make_transition_layer(prev_channels: list[int], cur_channels: list[int]) -> nn.ModuleList:
    # transitions between stages and spawns a new lower-res branch via stride 2 conv
    num_prev, num_cur = len(prev_channels), len(cur_channels)
    transitions = nn.ModuleList()
    for i in range(num_cur):
        if i < num_prev:
            if prev_channels[i] != cur_channels[i]:
                transitions.append(
                    nn.Sequential(
                        nn.Conv2d(prev_channels[i], cur_channels[i], 3, padding=1, bias=False),
                        nn.BatchNorm2d(cur_channels[i], momentum=_BN_MOMENTUM),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                transitions.append(nn.Identity())
        else:
            transitions.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels[-1], cur_channels[i], 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(cur_channels[i], momentum=_BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                )
            )
    return transitions


def _apply_transition(transitions: nn.ModuleList, prev_outputs: list[torch.Tensor]) -> list[torch.Tensor]:
    # pass previous stage outputs through transition layers to get new branch inputs
    num_prev = len(prev_outputs)
    return [
        transitions[i](prev_outputs[i] if i < num_prev else prev_outputs[-1])
        for i in range(len(transitions))
    ]


class HRNetW32Backbone(nn.Module):
    # full hrnet-w32 backbone returning a fused (B, 480, H/4, W/4) feature map

    out_channels = sum(_STAGE_CHANNELS["stage4"])  # 32 + 64 + 128 + 256 = 480

    def __init__(self, in_channels: int = 3) -> None:
        # build stem, 4 stages, and transition layers
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            Bottleneck(64, 64),
            Bottleneck(64 * Bottleneck.expansion, 64),
            Bottleneck(64 * Bottleneck.expansion, 64),
            Bottleneck(64 * Bottleneck.expansion, 64),
        )
        stage1_out_ch = 64 * Bottleneck.expansion  # 256

        self.transition1 = _make_transition_layer([stage1_out_ch], _STAGE_CHANNELS["stage2"])
        self.stage2 = nn.ModuleList(
            [
                HighResolutionModule(2, _STAGE_CHANNELS["stage2"], _STAGE_NUM_BLOCKS["stage2"])
                for _ in range(_STAGE_NUM_MODULES["stage2"])
            ]
        )

        self.transition2 = _make_transition_layer(_STAGE_CHANNELS["stage2"], _STAGE_CHANNELS["stage3"])
        self.stage3 = nn.ModuleList(
            [
                HighResolutionModule(3, _STAGE_CHANNELS["stage3"], _STAGE_NUM_BLOCKS["stage3"])
                for _ in range(_STAGE_NUM_MODULES["stage3"])
            ]
        )

        self.transition3 = _make_transition_layer(_STAGE_CHANNELS["stage3"], _STAGE_CHANNELS["stage4"])
        self.stage4 = nn.ModuleList(
            [
                HighResolutionModule(4, _STAGE_CHANNELS["stage4"], _STAGE_NUM_BLOCKS["stage4"])
                for _ in range(_STAGE_NUM_MODULES["stage4"])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # run through stem and stages then upsmple everything to H/4 and concat
        x = self.stem(x)
        x = self.stage1(x)

        xs = _apply_transition(self.transition1, [x])
        for module in self.stage2:
            xs = module(xs)

        xs = _apply_transition(self.transition2, xs)
        for module in self.stage3:
            xs = module(xs)

        xs = _apply_transition(self.transition3, xs)
        for module in self.stage4:
            xs = module(xs)

        target_hw = xs[0].shape[-2:]
        upsampled = [xs[0]] + [
            F.interpolate(branch, size=target_hw, mode="bilinear", align_corners=False) for branch in xs[1:]
        ]
        return torch.cat(upsampled, dim=1)