import numpy as np
import pytest
import torch
from torch import nn

from app.analytical.gum_propagation import assemble_block_covariance, combined_uncertainty, expanded_uncertainty
from app.analytical.jacobian import numerical_jacobian
from app.analytical.morphometrics import MORPHOMETRIC_FUNCTIONS
from app.pipeline import AssessmentPipeline


class StubDetector(nn.Module):
    # fixed random heatmaps / SPD covariances so no weights or timm download are needed
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.heatmaps = torch.rand(1, 13, 64, 48, generator=g)
        a = torch.randn(1, 13, 2, 2, generator=g)
        self.cov = a @ a.transpose(-1, -2) * 4 + 4 * torch.eye(2)
        self.vis = torch.full((1, 13), 3.0)

    def forward(self, x):
        return self.heatmaps, self.cov, self.vis


def run(scales):
    pipe = AssessmentPipeline(StubDetector(), uncertainty_scales=scales)
    return pipe.analyze_detailed(torch.zeros(1, 3, 8, 8))


def test_scale_one_matches_unscaled_formula():
    out_default, out_ones = run({}), run({k: 1.0 for k in MORPHOMETRIC_FUNCTIONS})
    x = out_default.keypoints.reshape(-1)
    sigma = assemble_block_covariance(out_default.covariances)
    for a, b in zip(out_default.criteria, out_ones.criteria):
        fn = MORPHOMETRIC_FUNCTIONS[a.criterion_key]
        expected = expanded_uncertainty(combined_uncertainty(numerical_jacobian(fn, x), sigma))
        assert a.uncertainty == expected == b.uncertainty
        assert a.decision == b.decision


def test_scale_multiplies_uncertainty():
    base = run({})
    scaled = run({"dorsal-body-ratio": 3.0})
    for a, b in zip(base.criteria, scaled.criteria):
        factor = 3.0 if a.criterion_key == "dorsal-body-ratio" else 1.0
        assert b.uncertainty == pytest.approx(a.uncertainty * factor)
