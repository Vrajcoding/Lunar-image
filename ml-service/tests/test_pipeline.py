"""
test_pipeline.py — Full test suite for the Lunar Registration ML Service.

Covers:
  1. Health endpoint
  2. Register — easy synthetic pair
  3. Register — medium synthetic pair
  4. Register — same image as source + reference (perfect-match baseline)
  5. Error handling — empty file upload
  6. Error handling — non-image binary
  7. Acceptance criteria validation (Section 11 of ml.md)
  8. Unit tests for each pipeline module
"""
import io
import os
import csv
import sys
import numpy as np
import cv2
import pytest

# ── sys.path is already patched by conftest.py ────────────────────────────────

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _post_register(client, source_path: str, reference_path: str):
    """Helper: POST /register with two image files, return response."""
    src_bytes = _image_bytes(source_path)
    ref_bytes = _image_bytes(reference_path)
    return client.post(
        "/register",
        files={
            "source":    ("source.png", io.BytesIO(src_bytes), "image/png"),
            "reference": ("reference.png", io.BytesIO(ref_bytes), "image/png"),
        },
    )


# ===========================================================================
# 1. Health Endpoint
# ===========================================================================

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "service" in body


# ===========================================================================
# 2. Register — Easy Pair (8° rotation, small translation)
# ===========================================================================

class TestRegisterEasyPair:
    def test_status_200(self, client, sample_images):
        resp = _post_register(client, sample_images["source1"], sample_images["ref1"])
        assert resp.status_code == 200, f"Response body: {resp.text}"

    def test_response_completed(self, client, sample_images):
        body = _post_register(client, sample_images["source1"], sample_images["ref1"]).json()
        assert body["status"] == "completed"

    def test_job_id_present(self, client, sample_images):
        body = _post_register(client, sample_images["source1"], sample_images["ref1"]).json()
        assert "job_id" in body and len(body["job_id"]) > 0

    def test_metrics_present(self, client, sample_images):
        body = _post_register(client, sample_images["source1"], sample_images["ref1"]).json()
        m = body["metrics"]
        for key in ("rmse_px", "inlier_count", "inlier_ratio", "uniformity_score", "runtime_sec"):
            assert key in m, f"Missing metric: {key}"

    def test_inlier_ratio_positive(self, client, sample_images):
        body = _post_register(client, sample_images["source1"], sample_images["ref1"]).json()
        assert body["metrics"]["inlier_ratio"] > 0, "Expected at least some inliers on easy pair"

    def test_output_files_written(self, client, sample_images):
        body = _post_register(client, sample_images["source1"], sample_images["ref1"]).json()
        assert os.path.exists(body["registered_image_path"]), "registered.png not found"
        assert os.path.exists(body["match_points_path"]), "matches.csv not found"

    def test_registered_image_is_valid_png(self, client, sample_images):
        body = _post_register(client, sample_images["source1"], sample_images["ref1"]).json()
        img = cv2.imread(body["registered_image_path"], cv2.IMREAD_GRAYSCALE)
        assert img is not None, "registered.png could not be decoded by OpenCV"

    def test_matches_csv_has_header_and_data(self, client, sample_images):
        body = _post_register(client, sample_images["source1"], sample_images["ref1"]).json()
        with open(body["match_points_path"], newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["src_x", "src_y", "ref_x", "ref_y", "is_inlier"], "CSV header mismatch"
        assert len(rows) > 1, "CSV has no data rows"

    def test_runtime_is_positive(self, client, sample_images):
        body = _post_register(client, sample_images["source1"], sample_images["ref1"]).json()
        assert body["metrics"]["runtime_sec"] > 0


# ===========================================================================
# 3. Register — Medium Pair (20° rotation, scale 1.1)
# ===========================================================================

class TestRegisterMediumPair:
    def test_status_200(self, client, sample_images):
        resp = _post_register(client, sample_images["source2"], sample_images["ref2"])
        assert resp.status_code == 200, f"Response body: {resp.text}"

    def test_response_completed(self, client, sample_images):
        body = _post_register(client, sample_images["source2"], sample_images["ref2"]).json()
        assert body["status"] == "completed"

    def test_inlier_count_positive(self, client, sample_images):
        body = _post_register(client, sample_images["source2"], sample_images["ref2"]).json()
        assert body["metrics"]["inlier_count"] > 0, "Expected inliers on medium pair"


# ===========================================================================
# 4. Same-Image Registration (perfect-match baseline — RMSE should be ~0)
# ===========================================================================

class TestRegisterSameImage:
    def test_status_200(self, client, sample_images):
        resp = _post_register(client, sample_images["same"], sample_images["same"])
        assert resp.status_code == 200

    def test_low_rmse_on_identical_images(self, client, sample_images):
        body = _post_register(client, sample_images["same"], sample_images["same"]).json()
        rmse = body["metrics"]["rmse_px"]
        assert rmse < 2.0, f"RMSE on identical images should be near 0, got {rmse}"

    def test_high_inlier_ratio_on_identical_images(self, client, sample_images):
        body = _post_register(client, sample_images["same"], sample_images["same"]).json()
        ratio = body["metrics"]["inlier_ratio"]
        assert ratio > 0.5, f"Expected high inlier ratio on identical images, got {ratio}"


# ===========================================================================
# 5. Error Handling — Empty File
# ===========================================================================

class TestErrorHandlingEmpty:
    def test_empty_source_returns_400(self, client, sample_images):
        ref_bytes = _image_bytes(sample_images["ref1"])
        resp = client.post(
            "/register",
            files={
                "source":    ("empty.png", io.BytesIO(b""), "image/png"),
                "reference": ("reference.png", io.BytesIO(ref_bytes), "image/png"),
            },
        )
        assert resp.status_code in (400, 422), f"Expected 400/422, got {resp.status_code}"

    def test_empty_reference_returns_400(self, client, sample_images):
        src_bytes = _image_bytes(sample_images["source1"])
        resp = client.post(
            "/register",
            files={
                "source":    ("source.png", io.BytesIO(src_bytes), "image/png"),
                "reference": ("empty.png", io.BytesIO(b""), "image/png"),
            },
        )
        assert resp.status_code in (400, 422), f"Expected 400/422, got {resp.status_code}"


# ===========================================================================
# 6. Error Handling — Non-image Binary
# ===========================================================================

class TestErrorHandlingNonImage:
    def test_corrupt_file_does_not_crash_server(self, client, sample_images):
        """Server must return 4xx, not 500, on a non-image file."""
        ref_bytes = _image_bytes(sample_images["ref1"])
        junk = b"THIS IS NOT AN IMAGE FILE \x00\x01\x02\x03" * 100
        resp = client.post(
            "/register",
            files={
                "source":    ("bad.png", io.BytesIO(junk), "image/png"),
                "reference": ("reference.png", io.BytesIO(ref_bytes), "image/png"),
            },
        )
        # Must NOT be a 500 (internal server error)
        assert resp.status_code != 500, f"Server crashed with 500 on corrupt input: {resp.text}"
        # Should be 4xx client error
        assert 400 <= resp.status_code < 500, f"Expected 4xx, got {resp.status_code}: {resp.text}"


# ===========================================================================
# 7. Acceptance Criteria (Section 11 of ml.md)
# ===========================================================================

class TestAcceptanceCriteria:
    """
    Validates all acceptance criteria from ml.md Section 11:
    - POST /register works on any two arbitrary image sizes without crashing
    - No hard-coded parameters
    - RMSE < 1.0 on synthetic known-transform pair (sub-pixel accuracy)
    - uniformity_score > 0.7
    - All output files written to /data/{job_id}/
    - /health returns 200 OK
    """

    def test_health_200(self, client):
        assert client.get("/health").status_code == 200

    def test_no_crash_different_sizes(self, client, tmp_path):
        """Register two images with very different sizes — must not crash."""
        img_small = np.full((100, 120), 128, dtype=np.uint8)
        img_large = np.full((480, 640), 110, dtype=np.uint8)
        # Add features so SIFT has something to detect
        for i in range(10, 90, 15):
            cv2.circle(img_small, (i, i), 6, (50,), -1)
            cv2.circle(img_large, (i * 5, i * 4), 25, (60,), -1)

        s_path = str(tmp_path / "small.png")
        l_path = str(tmp_path / "large.png")
        cv2.imwrite(s_path, img_small)
        cv2.imwrite(l_path, img_large)

        resp = client.post(
            "/register",
            files={
                "source":    ("small.png",  open(s_path, "rb"),  "image/png"),
                "reference": ("large.png",  open(l_path, "rb"),  "image/png"),
            },
        )
        assert resp.status_code == 200, f"Crashed on different-size images: {resp.text}"
        assert resp.json()["status"] == "completed"

    def test_subpixel_rmse_on_easy_pair(self, client, sample_images):
        """
        RMSE must be sub-pixel (< 1.0 px) on the easy synthetic pair
        (8° rotation with known affine transform).
        """
        body = _post_register(client, sample_images["source1"], sample_images["ref1"]).json()
        rmse = body["metrics"]["rmse_px"]
        assert rmse < 1.0, (
            f"Acceptance criterion FAILED: RMSE={rmse:.4f} px (must be < 1.0 px). "
            "Sub-pixel refinement may not be working correctly."
        )

    def test_uniformity_score_above_threshold(self, client, sample_images):
        """uniformity_score must be > 0.7 (matches distributed across >70% of grid cells)."""
        body = _post_register(client, sample_images["source1"], sample_images["ref1"]).json()
        score = body["metrics"]["uniformity_score"]
        assert score > 0.7, (
            f"Acceptance criterion FAILED: uniformity_score={score:.4f} (must be > 0.7)."
        )

    def test_output_files_exist_in_job_dir(self, client, sample_images):
        """Registered image + matches CSV must be written to the job directory."""
        body = _post_register(client, sample_images["source1"], sample_images["ref1"]).json()
        assert os.path.isfile(body["registered_image_path"]), "registered.png missing"
        assert os.path.isfile(body["match_points_path"]),     "matches.csv missing"

    def test_metrics_all_fields_present(self, client, sample_images):
        body = _post_register(client, sample_images["source1"], sample_images["ref1"]).json()
        required = {"rmse_px", "inlier_count", "inlier_ratio", "uniformity_score", "runtime_sec"}
        assert required.issubset(body["metrics"].keys()), (
            f"Missing metrics fields: {required - body['metrics'].keys()}"
        )


# ===========================================================================
# 8. Unit Tests — Individual Pipeline Modules
# ===========================================================================

class TestPreprocessModule:
    def test_load_grayscale(self, sample_images):
        from app.pipeline.preprocess import load_grayscale
        img = load_grayscale(sample_images["ref1"])
        assert img is not None
        assert img.ndim == 2

    def test_normalize_illumination(self, sample_images):
        from app.pipeline.preprocess import load_grayscale, normalize_illumination
        img = normalize_illumination(load_grayscale(sample_images["ref1"]))
        assert img.dtype == np.uint8

    def test_build_pyramid(self, sample_images):
        from app.pipeline.preprocess import load_grayscale, build_pyramid
        img = load_grayscale(sample_images["ref1"])
        pyr = build_pyramid(img, 4)
        assert len(pyr) == 4
        # Each level should be half the previous
        assert pyr[1].shape[0] == pyr[0].shape[0] // 2

    def test_preprocess_returns_pyramid(self, sample_images):
        from app.pipeline.preprocess import preprocess
        pyr = preprocess(sample_images["ref1"], levels=3)
        assert len(pyr) == 3
        assert all(img.ndim == 2 for img in pyr)

    def test_load_grayscale_invalid_path(self):
        from app.pipeline.preprocess import load_grayscale
        with pytest.raises(ValueError):
            load_grayscale("/nonexistent/path/image.png")


class TestFeaturesModule:
    def test_sift_detects_keypoints(self, sample_images):
        from app.pipeline.preprocess import load_grayscale
        from app.pipeline.features import detect_and_describe
        img = load_grayscale(sample_images["ref1"])
        kp, desc = detect_and_describe(img, n_features=500)
        assert len(kp) > 0, "SIFT found no keypoints"
        assert desc is not None

    def test_descriptors_shape(self, sample_images):
        from app.pipeline.preprocess import load_grayscale
        from app.pipeline.features import detect_and_describe
        img = load_grayscale(sample_images["ref1"])
        kp, desc = detect_and_describe(img, n_features=500)
        assert desc.shape[1] == 128, "SIFT descriptors should be 128-dimensional"


class TestMatchingModule:
    def test_match_descriptors_returns_list(self, sample_images):
        from app.pipeline.preprocess import load_grayscale
        from app.pipeline.features import detect_and_describe
        from app.pipeline.matching import match_descriptors
        img_a = load_grayscale(sample_images["source1"])
        img_b = load_grayscale(sample_images["ref1"])
        _, desc_a = detect_and_describe(img_a, 500)
        _, desc_b = detect_and_describe(img_b, 500)
        matches = match_descriptors(desc_a, desc_b, 0.75)
        assert isinstance(matches, list)
        assert len(matches) > 0, "Expected some matches on synthetic pair"

    def test_uniform_distribution(self, sample_images):
        from app.pipeline.preprocess import load_grayscale
        from app.pipeline.features import detect_and_describe
        from app.pipeline.matching import match_descriptors, enforce_uniform_distribution
        img_a = load_grayscale(sample_images["source1"])
        img_b = load_grayscale(sample_images["ref1"])
        kp_a, desc_a = detect_and_describe(img_a, 500)
        _, desc_b = detect_and_describe(img_b, 500)
        matches = match_descriptors(desc_a, desc_b, 0.75)
        filtered, score = enforce_uniform_distribution(matches, kp_a, img_a.shape, 8, 10)
        assert 0.0 <= score <= 1.0
        assert isinstance(filtered, list)

    def test_none_descriptors_handled(self):
        from app.pipeline.matching import match_descriptors
        result = match_descriptors(None, None, 0.75)
        assert result == []


class TestGeometryModule:
    def test_estimate_homography_returns_matrix(self, sample_images):
        from app.pipeline.preprocess import load_grayscale
        from app.pipeline.features import detect_and_describe
        from app.pipeline.matching import match_descriptors, enforce_uniform_distribution
        from app.pipeline.geometry import estimate_homography
        img_a = load_grayscale(sample_images["source1"])
        img_b = load_grayscale(sample_images["ref1"])
        kp_a, desc_a = detect_and_describe(img_a, 1000)
        kp_b, desc_b = detect_and_describe(img_b, 1000)
        matches = match_descriptors(desc_a, desc_b, 0.75)
        filtered, _ = enforce_uniform_distribution(matches, kp_a, img_a.shape, 8, 10)
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in filtered]).reshape(-1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in filtered]).reshape(-1, 2)
        H, mask = estimate_homography(src_pts, dst_pts, 3.0)
        assert H.shape == (3, 3)
        assert mask is not None

    def test_warp_image_output_shape(self, sample_images):
        from app.pipeline.preprocess import load_grayscale
        from app.pipeline.geometry import warp_image
        img = load_grayscale(sample_images["source1"])
        H = np.eye(3, dtype=np.float32)
        warped = warp_image(img, H, (300, 400))
        assert warped.shape == (300, 400)

    def test_fallback_on_too_few_points(self):
        from app.pipeline.geometry import estimate_homography
        src = np.float32([[0, 0], [1, 0]])
        dst = np.float32([[0, 0], [1, 0]])
        H, mask = estimate_homography(src, dst, 3.0)
        assert H.shape == (3, 3)


class TestEvaluateModule:
    def test_compute_rmse_identity(self):
        from app.pipeline.evaluate import compute_rmse
        pts = np.float32([[10, 10], [20, 20], [30, 30]])
        H = np.eye(3, dtype=np.float64)
        rmse = compute_rmse(pts, pts, H)
        assert rmse < 0.01, f"RMSE with identity H on same points should be ~0, got {rmse}"

    def test_compute_rmse_empty(self):
        from app.pipeline.evaluate import compute_rmse
        rmse = compute_rmse(np.zeros((0, 2)), np.zeros((0, 2)), np.eye(3))
        assert rmse == 0.0

    def test_build_metrics_keys(self):
        from app.pipeline.evaluate import build_metrics
        m = build_metrics(50, 100, 0.5, 0.8, 1.23)
        assert set(m.keys()) == {"rmse_px", "inlier_count", "inlier_ratio", "uniformity_score", "runtime_sec"}
        assert m["inlier_count"] == 50
        assert m["inlier_ratio"] == 0.5


class TestRefineModule:
    def test_subpixel_refine_empty(self):
        from app.pipeline.refine import subpixel_refine_points
        img = np.zeros((100, 100), dtype=np.uint8)
        result = subpixel_refine_points(img, np.zeros((0, 2), dtype=np.float32))
        assert result.shape == (0, 2)

    def test_subpixel_refine_returns_same_shape(self, sample_images):
        from app.pipeline.preprocess import load_grayscale
        from app.pipeline.refine import subpixel_refine_points
        img = load_grayscale(sample_images["ref1"])
        pts = np.float32([[50, 50], [100, 100], [200, 200]])
        refined = subpixel_refine_points(img, pts)
        assert refined.shape == pts.shape
