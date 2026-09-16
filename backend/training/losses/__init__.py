"""Loss terms for the two-stage HRNet training schedule (Build Prompt v2 §7)."""

from training.losses.gaussian_nll import gaussian_nll_loss
from training.losses.heatmap_mse import heatmap_mse_loss
from training.losses.visibility_bce import visibility_bce_loss

__all__ = ["gaussian_nll_loss", "heatmap_mse_loss", "visibility_bce_loss"]
