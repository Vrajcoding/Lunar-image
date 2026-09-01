"""
conftest.py — Pytest fixtures for the Lunar Registration ML Service tests.

Generates synthetic sample images on first run so tests are fully self-contained.
"""
import os
import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Make sure the app can be imported — add ml-service root to sys.path
# ---------------------------------------------------------------------------
import sys
ML_SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ML_SERVICE_ROOT not in sys.path:
    sys.path.insert(0, ML_SERVICE_ROOT)

# Use a local ./data dir during tests (not /data which needs root in Docker)
os.environ.setdefault("STORAGE_DIR", os.path.join(ML_SERVICE_ROOT, "test_data"))

from app.main import app  # noqa: E402  (import after sys.path patch)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    """FastAPI test client — reused across all tests in the session."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_images():
    """
    Returns paths to three synthetic image pairs:
      pair1 — easy:   small rotation (8°) + minor translation
      pair2 — medium: larger rotation (20°) + scale 1.1
      pair3 — same:   identical images (perfect-match baseline)
    """
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    def _make_lunar_base(seed: int = 42, size: int = 512) -> np.ndarray:
        """Generate a synthetic lunar surface with craters and texture."""
        rng = np.random.default_rng(seed)
        img = np.full((size, size), 110, dtype=np.uint8)
        noise = rng.normal(0, 12, (size, size)).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        craters = [
            (150, 150, 45), (320, 180, 65), (200, 360, 55),
            (390, 390, 38), (100, 420, 28), (430, 110, 50),
            (260, 260, 30), (80, 200, 20), (450, 300, 42),
        ]
        for (cx, cy, r) in craters:
            if cx + r < size and cy + r < size:
                cv2.circle(img, (cx, cy), r, (65,), -1)
                cv2.circle(img, (cx + 4, cy + 4), max(r - 5, 2), (155,), -1)
                cv2.circle(img, (cx, cy), r, (25,), 2)
        # Add some ridges/lines for extra SIFT features
        for i in range(0, size, 60):
            cv2.line(img, (i, 0), (i + 30, size), (90,), 1)
        return img

    center = (256, 256)
    base = _make_lunar_base(seed=42, size=512)

    # ── Pair 1: Easy — 8° rotation + small translation ─────────────────────
    ref1_path = os.path.join(SAMPLE_DIR, "ref1.png")
    src1_path = os.path.join(SAMPLE_DIR, "source1.png")
    cv2.imwrite(ref1_path, base)
    M1 = cv2.getRotationMatrix2D(center, 8, 1.02)
    M1[0, 2] += 12
    M1[1, 2] -= 8
    cv2.imwrite(src1_path, cv2.warpAffine(base, M1, (512, 512)))

    # ── Pair 2: Medium — 20° rotation + scale 1.1 ──────────────────────────
    ref2_path = os.path.join(SAMPLE_DIR, "ref2.png")
    src2_path = os.path.join(SAMPLE_DIR, "source2.png")
    cv2.imwrite(ref2_path, base)
    M2 = cv2.getRotationMatrix2D(center, 20, 1.1)
    cv2.imwrite(src2_path, cv2.warpAffine(base, M2, (512, 512)))

    # ── Pair 3: Same — identical images (perfect baseline) ──────────────────
    same_path = os.path.join(SAMPLE_DIR, "same.png")
    cv2.imwrite(same_path, base)

    return {
        "ref1":   ref1_path,
        "source1": src1_path,
        "ref2":   ref2_path,
        "source2": src2_path,
        "same":   same_path,
    }
