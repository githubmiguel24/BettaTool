"""Loss terms for the two-stage HRNet training schedule (Build Prompt v2 §7)."""

from training.losses.gaussian_nll import gaussian_nll_loss
from training.losses.heatmap_kl import heatmap_kl_loss

__all__ = ["gaussian_nll_loss", "heatmap_kl_loss"]
