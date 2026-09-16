"""End-to-end test of POST /analyze.

This is the test that would have caught the Sept-16 integration gap: every
unit test in this suite passed while `/analyze` still returned HTTP 501 and
the frontend made no network calls at all. Unit coverage of the three tiers
says nothing about whether they are wired together.

Runs against randomly initialized weights (no checkpoint required), which is
exactly the point - it asserts the PLUMBING, not the accuracy.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.main import app  # noqa: E402
from app.perception.keypoints import NUM_KEYPOINTS  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _png_bytes(width: int = 800, height: int = 600) -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_rejects_non_image(client):
    response = client.post("/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_rejects_corrupt_image(client):
    response = client.post("/analyze", files={"file": ("broken.png", b"not a png", "image/png")})
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


@pytest.mark.parametrize("size", [(800, 600), (640, 640), (1200, 500)])
def test_analyze_returns_a_complete_report(client, size):
    width, height = size
    response = client.post(
        "/analyze",
        files={"file": ("betta.png", _png_bytes(width, height), "image/png")},
    )
    assert response.status_code == 200, response.text
    report = response.json()

    assert len(report["measurements"]) == 6
    assert len(report["keypoints"]) == NUM_KEYPOINTS
    assert report["image_width"] == width
    assert report["image_height"] == height

    for m in report["measurements"]:
        assert m["decision"] in {"Confident Pass", "Confident Fault", "Defer to Judge"}
        for field in ("value", "uncertainty", "tsi", "rmse"):
            assert np.isfinite(m[field]), f"{m['criterion_key']}.{field} is not finite"
        assert m["uncertainty"] >= 0.0


def test_keypoints_are_in_original_image_pixels(client):
    """The overlay draws directly in uploaded-image coordinates, so predictions
    must be expressed in that space - not the 384x384 crop space. A regression
    here puts every landmark in the top-left corner of the photo."""
    width, height = 1600, 900
    response = client.post(
        "/analyze",
        files={"file": ("betta.png", _png_bytes(width, height), "image/png")},
    )
    report = response.json()

    xs = [kp["x"] for kp in report["keypoints"]]
    ys = [kp["y"] for kp in report["keypoints"]]

    # Generous bounds: an untrained model can predict anywhere, but a
    # crop-space leak would pin everything into [0, 384] on a 1600px image.
    assert max(xs) > 384, "x coordinates never exceed 384 - crop space leaked through"
    for x, y in zip(xs, ys):
        assert -width <= x <= 2 * width
        assert -height <= y <= 2 * height


def test_untrained_model_is_flagged(client):
    """Integrity guard: with no checkpoint on disk the response must say so.
    If this ever fails silently, an untrained demo could be screenshotted and
    presented as a measurement."""
    response = client.post("/analyze", files={"file": ("betta.png", _png_bytes(), "image/png")})
    report = response.json()

    from app.perception.loader import load_model_bundle

    if not load_model_bundle().trained:
        assert report["model_trained"] is False
        assert report["warnings"], "untrained model produced no warning"
        assert "NO TRAINED CHECKPOINT" in report["warnings"][0]


def test_covariance_ellipse_parameters_are_well_formed(client):
    response = client.post("/analyze", files={"file": ("betta.png", _png_bytes(), "image/png")})
    for kp in response.json()["keypoints"]:
        assert kp["sigma_x"] > 0
        assert kp["sigma_y"] > 0
        assert -1.0 <= kp["rho"] <= 1.0
        assert 0.0 <= kp["visibility"] <= 1.0


def test_report_is_persisted_and_retrievable(client):
    response = client.post("/analyze", files={"file": ("betta.png", _png_bytes(), "image/png")})
    report_id = response.json()["id"]

    from app.core.db import get_report

    assert get_report(report_id) is not None, "analyze did not persist the report"
