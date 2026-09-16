"""Visibility-head loss: binary cross-entropy on the presence/absence label.

Unlike the heatmap and NLL losses, this term is trained on EVERY keypoint,
including occluded (flag 0) and out-of-frame (flag -1) ones — those are
exactly the labels the visibility head exists to learn (Build Prompt v2 §3.3,
§7). Never mask this loss the way the heatmap/NLL losses are masked.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def visibility_bce_loss(visibility_logits: torch.Tensor, visibility_target: torch.Tensor) -> torch.Tensor:
    """
    Args:
        visibility_logits: (B, K) raw logits from `VisibilityHead`.
        visibility_target: (B, K) float tensor in {0.0, 1.0}: 1.0 for
            visibility flag 2 or 1 (clear / ambiguous), 0.0 for flag 0 or -1
            (occluded / out-of-frame).

    Returns:
        Scalar mean BCE-with-logits loss over all keypoints and the batch.
    """
    return F.binary_cross_entropy_with_logits(visibility_logits, visibility_target)
