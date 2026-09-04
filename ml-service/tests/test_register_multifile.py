"""
test_register_multifile.py — /register with an optional detached label file
per side (source_label / reference_label), added so a real PDS4 product
(.img data + .xml label) can be uploaded through the HTTP API at all — the
single-file-per-side endpoint previously made that impossible regardless of
what app/pipeline/loader.py could already read from disk.
"""
import io
import os

import cv2
import numpy as np
import pytest
import rasterio

pytestmark = pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")


def _make_lunar_base(seed: int, size: int = 256) -> np.ndarray:
    """Small synthetic lunar-looking scene with enough structure for SIFT to
    find real matches — same idea as tests/conftest.py's sample generator."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 110, dtype=np.uint8)
    noise = rng.normal(0, 12, (size, size)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    for (cx, cy, r) in [(60, 60, 22), (160, 90, 30), (100, 180, 26), (200, 200, 18)]:
        cv2.circle(img, (cx, cy), r, (65,), -1)
        cv2.circle(img, (cx + 3, cy + 3), max(r - 4, 2), (155,), -1)
        cv2.circle(img, (cx, cy), r, (25,), 2)
    for i in range(0, size, 30):
        cv2.line(img, (i, 0), (i + 15, size), (90,), 1)
    return img


def _write_pds4_pair(dir_path: str, stem: str, arr: np.ndarray):
    """Writes <stem>.img + <stem>.xml via GDAL's real PDS4 driver."""
    os.makedirs(dir_path, exist_ok=True)
    xml_path = os.path.join(dir_path, f"{stem}.xml")
    h, w = arr.shape
    with rasterio.open(
        xml_path, "w", driver="PDS4", width=w, height=h, count=1, dtype=str(arr.dtype)
    ) as ds:
        ds.write(arr, 1)
    return xml_path, os.path.join(dir_path, f"{stem}.img")


def _mkfile(path):
    return open(path, "rb")


# ---------------------------------------------------------------------------
# 1. PDS4 pair upload (.img + .xml on both sides) — job completes, real metrics
# ---------------------------------------------------------------------------

def test_pds4_pair_upload_completes(client, tmp_path):
    base = _make_lunar_base(seed=10, size=256)
    center = (128, 128)
    M = cv2.getRotationMatrix2D(center, 6, 1.0)
    M[0, 2] += 4
    M[1, 2] -= 3
    warped = cv2.warpAffine(base, M, (256, 256))

    ref_xml, ref_img_path = _write_pds4_pair(
        str(tmp_path / "ref"), "ch2_tmc_ncn_20200101T0100000000_d_img_r01", base
    )
    src_xml, src_img_path = _write_pds4_pair(
        str(tmp_path / "src"), "ch2_tmc_nca_20200101T0100000000_d_img_a01", warped
    )

    with _mkfile(src_img_path) as sf, _mkfile(src_xml) as sl, \
         _mkfile(ref_img_path) as rf, _mkfile(ref_xml) as rl:
        resp = client.post(
            "/register",
            files={
                "source": (os.path.basename(src_img_path), sf, "application/octet-stream"),
                "source_label": (os.path.basename(src_xml), sl, "application/xml"),
                "reference": (os.path.basename(ref_img_path), rf, "application/octet-stream"),
                "reference_label": (os.path.basename(ref_xml), rl, "application/xml"),
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    m = body["metrics"]
    for key in ("rmse_px", "inlier_count", "inlier_ratio", "uniformity_score", "runtime_sec"):
        assert key in m
    assert m["inlier_count"] > 0, "expected real matches on a rotated/translated PDS4 pair"
    assert body["input_metadata"]["source"]["source_format"] == "pds4"
    assert body["input_metadata"]["reference"]["source_format"] == "pds4"
    assert body["input_metadata"]["source"]["product_id"]["instrument"] == "tmc"


# ---------------------------------------------------------------------------
# 2. .img with no label uploaded — clean error, job not left hanging
# ---------------------------------------------------------------------------

def test_img_without_label_fails_cleanly(client, sample_images, tmp_path):
    orphan_dir = tmp_path / "orphan"
    orphan_dir.mkdir()
    orphan_path = orphan_dir / "ch2_ohrc_ncp_20210101T0000000000_d_img_x01.img"
    orphan_path.write_bytes(os.urandom(400))  # raw bytes, no sibling label uploaded

    with _mkfile(sample_images["ref1"]) as rf:
        resp = client.post(
            "/register",
            files={
                "source": (orphan_path.name, io.BytesIO(orphan_path.read_bytes()), "application/octet-stream"),
                "reference": ("reference.png", rf, "image/png"),
            },
        )

    assert resp.status_code != 500, f"must not crash: {resp.text}"
    assert 400 <= resp.status_code < 500, f"expected 4xx, got {resp.status_code}: {resp.text}"
    assert ".xml" in resp.text and ".lbl" in resp.text


# ---------------------------------------------------------------------------
# 3. PNG only, no label fields at all — unchanged behaviour
# ---------------------------------------------------------------------------

def test_png_only_unchanged(client, sample_images):
    with _mkfile(sample_images["source1"]) as sf, _mkfile(sample_images["ref1"]) as rf:
        resp = client.post(
            "/register",
            files={
                "source": ("source1.png", sf, "image/png"),
                "reference": ("ref1.png", rf, "image/png"),
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert body["metrics"]["rmse_px"] == pytest.approx(0.4879, abs=0.01)
    assert body["metrics"]["inlier_count"] == 265
    assert body["metrics"]["uniformity_score"] == pytest.approx(0.9844, abs=0.01)
    assert body["input_metadata"]["source"]["source_format"] == "png"
