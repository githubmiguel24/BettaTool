"""Model shape and positive-definiteness tests (Build Prompt v2 §11).

Requires torch. SKIPPED (not failed) if torch is not importable — see
training/README.md, "Known limitations": torch could not be installed in
the sandbox this codebase was authored in, so this file has never actually
been run. Run it yourself (`pytest tests/test_model.py -v`) before trusting
the model code.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from app.perception.heatmap import soft_argmax  # noqa: E402
from app.perception.hrnet import HRNetKeypointDetector  # noqa: E402
from app.perception.hrnet_backbone import HRNetW32Backbone  # noqa: E402
from app.perception.keypoints import NUM_KEYPOINTS  # noqa: E402


@pytest.fixture(scope="module")
def model() -> "HRNetKeypointDetector":
    # backbone_source="custom" forces the self-contained path so this test
    # suite never depends on network access to download ImageNet weights.
    return HRNetKeypointDetector(backbone_source="custom")


@pytest.mark.parametrize("batch_size", [1, 4])
def test_forward_pass_shapes(model: "HRNetKeypointDetector", batch_size: int) -> None:
    image = torch.randn(batch_size, 3, 384, 384)
    heatmaps, covariances, visibility = model(image)
    assert heatmaps.shape == (batch_size, NUM_KEYPOINTS, 96, 96)
    assert covariances.shape == (batch_size, NUM_KEYPOINTS, 2, 2)
    assert visibility.shape == (batch_size, NUM_KEYPOINTS)


def test_backbone_output_channels() -> None:
    backbone = HRNetW32Backbone()
    x = torch.randn(1, 3, 384, 384)
    features = backbone(x)
    assert features.shape == (1, 480, 96, 96)
    assert backbone.out_channels == 480


def test_heatmaps_are_spatial_probability_distributions(model: "HRNetKeypointDetector") -> None:
    image = torch.randn(2, 3, 384, 384)
    heatmaps, _, _ = model(image)
    mass = heatmaps.sum(dim=(-1, -2))
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-4)
    assert torch.all(heatmaps >= 0)


def test_covariance_is_symmetric_and_positive_definite(model: "HRNetKeypointDetector") -> None:
    image = torch.randn(3, 3, 384, 384)
    _, covariances, _ = model(image)

    assert torch.allclose(covariances, covariances.transpose(-1, -2), atol=1e-5)

    eigvals = torch.linalg.eigvalsh(covariances)
    assert torch.all(eigvals > 0), "Sigma must be positive-definite for every keypoint (Build Prompt v2 §6)."


def test_soft_argmax_returns_exact_peak_for_one_hot_heatmap() -> None:
    heatmap = torch.zeros(1, 1, 10, 10)
    heatmap[0, 0, 3, 7] = 1.0  # row=3 (y), col=7 (x)
    mu = soft_argmax(heatmap)
    assert torch.allclose(mu[0, 0], torch.tensor([7.0, 3.0]), atol=1e-5)
