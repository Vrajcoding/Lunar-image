"""
test_loader.py — tests for the format-aware loader (app/pipeline/loader.py).

All fixtures are generated programmatically (no committed binary test data).
GeoTIFF/PDS4 fixtures are written with rasterio/GDAL itself so round-trip
correctness is checked against the same library the loader depends on.
"""
import logging
import os

import cv2
import numpy as np
import pytest
import rasterio

from app.pipeline.loader import load_image

pytestmark = pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_png(path: str, arr: np.ndarray):
    cv2.imwrite(path, arr)


def _write_tiff(path: str, bands: list[np.ndarray], crs=None, transform=None):
    """bands: list of 2D arrays, one per band, all the same dtype."""
    dtype = bands[0].dtype
    h, w = bands[0].shape
    kwargs = dict(driver="GTiff", width=w, height=h, count=len(bands), dtype=str(dtype))
    if crs is not None:
        kwargs["crs"] = crs
    if transform is not None:
        kwargs["transform"] = transform
    with rasterio.open(path, "w", **kwargs) as ds:
        for i, band in enumerate(bands, start=1):
            ds.write(band, i)


def _write_pds4_pair(dir_path: str, stem: str, arr: np.ndarray):
    """Writes <stem>.img + <stem>.xml via GDAL's PDS4 driver and returns the
    .xml path. This is the real GDAL PDS4 writer, not a hand-rolled label, so
    round-tripping through it exercises the same driver the loader uses."""
    xml_path = os.path.join(dir_path, f"{stem}.xml")
    h, w = arr.shape
    with rasterio.open(
        xml_path, "w", driver="PDS4", width=w, height=h, count=1, dtype=str(arr.dtype)
    ) as ds:
        ds.write(arr, 1)
    return xml_path


# ---------------------------------------------------------------------------
# 1. 8-bit PNG — regression guard against the old cv2.imread path
# ---------------------------------------------------------------------------

def test_png_is_byte_identical_to_old_cv2_path(tmp_path, sample_images):
    old = cv2.imread(sample_images["ref1"], cv2.IMREAD_GRAYSCALE)
    img, meta = load_image(sample_images["ref1"])
    assert np.array_equal(img, old)
    assert meta["scaling_applied"] == "none"
    assert meta["source_format"] == "png"
    assert meta["band_count"] == 1


# ---------------------------------------------------------------------------
# 2. 16-bit TIFF — stretched to uint8, dtype/method recorded
# ---------------------------------------------------------------------------

def test_16bit_tiff_stretched_to_uint8(tmp_path):
    rng = np.random.default_rng(1)
    arr16 = (rng.random((40, 50)) * 60000).astype(np.uint16)
    path = str(tmp_path / "scene16.tiff")
    _write_tiff(path, [arr16])

    img, meta = load_image(path)
    assert img.dtype == np.uint8
    assert img.shape == (40, 50)
    assert meta["original_dtype"] == "uint16"
    assert meta["scaling_applied"] in ("percentile_2_98", "minmax")
    assert meta["stretch_bounds"] is not None


# ---------------------------------------------------------------------------
# 3. Multi-band TIFF, no band requested — defaults to band 1, warns
# ---------------------------------------------------------------------------

def test_multiband_tiff_defaults_to_band1_and_warns(tmp_path, caplog):
    b1 = np.full((10, 10), 10, dtype=np.uint8)
    b2 = np.full((10, 10), 20, dtype=np.uint8)
    b3 = np.full((10, 10), 30, dtype=np.uint8)
    path = str(tmp_path / "multiband.tiff")
    _write_tiff(path, [b1, b2, b3])

    with caplog.at_level(logging.WARNING, logger="app.pipeline.loader"):
        img, meta = load_image(path)

    assert meta["band_count"] == 3
    assert meta["band_used"] == 1
    assert np.all(img == 10) or img.dtype == np.uint8  # band 1 content (already uint8, no stretch)
    assert any(
        "band" in rec.message.lower() and "3" in rec.message for rec in caplog.records
    ), "expected a warning naming the band count when no band was specified"


# ---------------------------------------------------------------------------
# 4. Multi-band TIFF, explicit band — correct band selected
# ---------------------------------------------------------------------------

def test_multiband_tiff_explicit_band_selection(tmp_path):
    b1 = np.full((10, 10), 10, dtype=np.uint8)
    b2 = np.full((10, 10), 20, dtype=np.uint8)
    b3 = np.full((10, 10), 30, dtype=np.uint8)
    path = str(tmp_path / "multiband2.tiff")
    _write_tiff(path, [b1, b2, b3])

    img, meta = load_image(path, band=2)
    assert meta["band_used"] == 2
    assert np.all(img == 20)


# ---------------------------------------------------------------------------
# 5. Shared stretch bounds — consistent vs independent stretching
# ---------------------------------------------------------------------------

def test_shared_stretch_bounds_vs_independent(tmp_path):
    rng = np.random.default_rng(2)
    ref16 = (rng.random((30, 30)) * 4000 + 1000).astype(np.uint16)   # ~[1000, 5000]
    src16 = (rng.random((30, 30)) * 6000 + 2000).astype(np.uint16)   # ~[2000, 8000]
    ref_path = str(tmp_path / "ref.tiff")
    src_path = str(tmp_path / "src.tiff")
    _write_tiff(ref_path, [ref16])
    _write_tiff(src_path, [src16])

    _, ref_meta = load_image(ref_path)
    src_independent, _ = load_image(src_path)
    src_shared, shared_meta = load_image(src_path, stretch_bounds=ref_meta["stretch_bounds"])

    assert shared_meta["stretch_bounds"] == ref_meta["stretch_bounds"]
    assert not np.array_equal(src_independent, src_shared), (
        "source stretched with its own bounds vs. the reference's bounds should "
        "differ — otherwise the shared-bounds wiring isn't actually taking effect"
    )


# ---------------------------------------------------------------------------
# 6 & 7. Synthetic PDS4 pair — via .xml directly, and via .img redirect
# ---------------------------------------------------------------------------

def test_pds4_loads_via_xml_label(tmp_path):
    rng = np.random.default_rng(3)
    arr = (rng.random((25, 35)) * 255).astype(np.uint8)
    xml_path = _write_pds4_pair(str(tmp_path), "ch2_tmc_nca_20200207T0716469418_d_img_d18", arr)

    img, meta = load_image(xml_path)
    assert img.shape == (25, 35)
    assert img.dtype == np.uint8
    assert meta["source_format"] == "pds4"
    assert np.array_equal(img, arr)


def test_pds4_img_redirects_to_sibling_xml(tmp_path, caplog):
    rng = np.random.default_rng(3)
    arr = (rng.random((25, 35)) * 255).astype(np.uint8)
    stem = "ch2_tmc_nca_20200207T0716469418_d_img_d18"
    xml_path = _write_pds4_pair(str(tmp_path), stem, arr)
    img_path = os.path.join(str(tmp_path), f"{stem}.img")
    assert os.path.isfile(img_path), "GDAL PDS4 writer should have produced the sibling .img"

    with caplog.at_level(logging.INFO, logger="app.pipeline.loader"):
        img_via_redirect, meta_via_redirect = load_image(img_path)
    img_via_xml, meta_via_xml = load_image(xml_path)

    assert np.array_equal(img_via_redirect, img_via_xml)
    assert meta_via_redirect["source_format"] == meta_via_xml["source_format"] == "pds4"
    assert any(".img" in rec.message and "redirect" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# 8. .img with no sibling label — clean, descriptive error
# ---------------------------------------------------------------------------

def test_img_without_sibling_label_raises_descriptive_error(tmp_path):
    orphan = tmp_path / "ch2_ohrc_ncp_20210101T0000000000_d_img_x01.img"
    orphan.write_bytes(b"\x00" * 100)  # raw bytes, no label describing them

    with pytest.raises(ValueError) as excinfo:
        load_image(str(orphan))

    msg = str(excinfo.value)
    assert str(orphan) in msg or orphan.name in msg
    assert ".xml" in msg
    assert ".lbl" in msg


# ---------------------------------------------------------------------------
# 9. Corrupt / unreadable file — clean error via the /register endpoint
# ---------------------------------------------------------------------------

def test_corrupt_tiff_fails_cleanly_through_register(client, sample_images):
    import io

    junk = b"NOT A REAL TIFF FILE" * 50
    with open(sample_images["ref1"], "rb") as r:
        ref_bytes = r.read()

    resp = client.post(
        "/register",
        files={
            "source": ("bad.tiff", io.BytesIO(junk), "image/tiff"),
            "reference": ("reference.png", io.BytesIO(ref_bytes), "image/png"),
        },
    )
    assert resp.status_code != 500, f"Server crashed with 500 on corrupt input: {resp.text}"
    assert 400 <= resp.status_code < 500, f"Expected 4xx, got {resp.status_code}: {resp.text}"


# ---------------------------------------------------------------------------
# 10. Filename parser
# ---------------------------------------------------------------------------

def test_product_id_parses_conforming_filename(tmp_path):
    rng = np.random.default_rng(4)
    arr = (rng.random((10, 10)) * 255).astype(np.uint8)
    path = str(tmp_path / "ch2_tmc_nca_20200207T0716469418_d_img_d18.png")
    _write_png(path, arr)

    _, meta = load_image(path)
    pid = meta["product_id"]
    assert pid is not None
    assert pid["instrument"] == "tmc"
    assert pid["camera"] == "aft"
    assert pid["calibration"] == "calibrated"
    assert pid["timestamp"] == "20200207T0716469418"


def test_product_id_none_for_non_conforming_filename(tmp_path):
    rng = np.random.default_rng(5)
    arr = (rng.random((10, 10)) * 255).astype(np.uint8)
    path = str(tmp_path / "random_image_name.png")
    _write_png(path, arr)

    img, meta = load_image(path)  # must not crash
    assert img is not None
    assert meta["product_id"] is None
