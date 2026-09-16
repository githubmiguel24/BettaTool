"""MFLD-Net deterministic baseline (Saleh et al., 2023).

A MobileNetV2-backboned landmark detector that outputs a single fixed
coordinate per keypoint, with no covariance/uncertainty. Its coordinates
are routed through the exact same `app.analytical` / `app.decisional`
engine as the proposed pipeline, per the thesis's "Baseline Evaluation
Logic" (Scope and Limitations), so RQ2 is a fair comparison.

TODO: implement or import the actual MFLD-Net architecture/weights; this
is a structural placeholder only.
"""

from __future__ import annotations

import torch
from torch import nn

from app.perception.keypoints import NUM_KEYPOINTS


class MFLDNet(nn.Module):
    def __init__(self, num_keypoints: int = NUM_KEYPOINTS) -> None:
        super().__init__()
        raise NotImplementedError(
            "Implement or import the MobileNetV2-based MFLD-Net architecture "
            "(Saleh et al., 2023) before using this baseline."
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
