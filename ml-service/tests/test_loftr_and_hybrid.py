"""
test_loftr_and_hybrid.py — Comprehensive test suite for LoFTR learned correspondence,
hybrid fallback hierarchy, spatial entropy, independent RMSE, and explicit failure detection.
"""
import os
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.loftr_matcher import LoFTRMatcher
from app.models.model_loader import get_loftr_matcher
from app.pipeline.matching import compute_spatial_entropy, filter_dense_correspondences
from app.pipeline.evaluate import compute_rmse, compute_train_test_rmse, build_metrics
from app.pipeline.confidence import compute_confidence_score
from app.pipeline.refine import subpixel_refine_points
from app.pipeline.geometry import is_valid_homography, estimate_homography
from app.pipeline.registration import run_registration_pipeline
from app.pipeline.loader import _extract_xml_metadata, load_image
from app.evaluation.dataset import SyntheticDatasetGenerator, compute_ground_truth_corner_error

client = TestClient(app)


def test_loftr_matcher_initialization():
    """Verify LoFTRMatcher handles CPU/CUDA gracefully without unhandled exceptions."""
    matcher = LoFTRMatcher(device="cpu")
    assert matcher.device.type == "cpu"
    # Even if weights not yet downloaded on an offline runner, is_available reflects state
    assert isinstance(matcher.is_available, bool)


def test_spatial_entropy_calculation():
    """Verify normalized spatial entropy produces 0 for concentrated points and ~1 for uniform points."""
    img_shape = (500, 500)
    
    # Clustered points in single cell (0, 0)
    clustered_pts = np.array([[10.0, 10.0], [15.0, 12.0], [20.0, 18.0], [25.0, 22.0]], dtype=np.float32)
    u_clustered, ent_clustered, cells = compute_spatial_entropy(clustered_pts, img_shape, grid_size=8)
    assert u_clustered == 0.0
    assert cells == 1

    # Uniformly distributed points across all 64 cells
    grid_pts = []
    for r in range(8):
        for c in range(8):
            grid_pts.append([c * 60.0 + 30.0, r * 60.0 + 30.0])
    grid_pts = np.array(grid_pts, dtype=np.float32)
    u_uniform, ent_uniform, cells_uniform = compute_spatial_entropy(grid_pts, img_shape, grid_size=8)
    assert u_uniform > 0.95
    assert cells_uniform == 64


def test_filter_dense_correspondences():
    """Verify density capping per grid cell."""
    img_shape = (400, 400)
    pts0 = np.tile(np.array([[20.0, 20.0]], dtype=np.float32), (50, 1))
    pts1 = pts0.copy()
    conf = np.linspace(0.1, 0.9, 50, dtype=np.float32)

    f_pts0, f_pts1, f_conf, u_score, s_ent = filter_dense_correspondences(
        pts0, pts1, conf, img_shape, grid_size=8, max_per_cell=10
    )
    assert len(f_pts0) == 10
    assert len(f_conf) == 10
    # Highest confidence retained
    assert f_conf[0] >= f_conf[-1]


def test_train_test_rmse():
    """Verify independent RMSE splits train and test partitions correctly."""
    src = np.array([
        [10.0, 10.0], [50.0, 10.0], [90.0, 10.0], [10.0, 50.0],
        [50.0, 50.0], [90.0, 50.0], [10.0, 90.0], [50.0, 90.0], [90.0, 90.0], [100.0, 100.0]
    ], dtype=np.float64)
    # Perfect identity mapping with slight noise
    dst = src + np.array([[0.1, -0.1]] * len(src))
    H = np.eye(3, dtype=np.float64)

    fit_rmse, test_rmse = compute_train_test_rmse(src, dst, H, test_split=0.20)
    assert fit_rmse > 0.0
    assert test_rmse > 0.0
    assert abs(fit_rmse - test_rmse) < 1.0


def test_confidence_scoring_levels():
    """Verify confidence classification into HIGH, MEDIUM, LOW."""
    # High quality case: low RMSE, high inlier ratio, high uniformity
    score_high, level_high = compute_confidence_score(
        test_rmse=0.45, inlier_ratio=0.85, uniformity_score=0.80, inlier_count=45
    )
    assert level_high == "HIGH"
    assert score_high >= 0.70

    # Medium quality case
    score_med, level_med = compute_confidence_score(
        test_rmse=2.5, inlier_ratio=0.50, uniformity_score=0.45, inlier_count=12
    )
    assert level_med == "MEDIUM"

    # Low quality case
    score_low, level_low = compute_confidence_score(
        test_rmse=18.0, inlier_ratio=0.15, uniformity_score=0.10, inlier_count=3
    )
    assert level_low == "LOW"


def test_subpixel_refinement():
    """Verify subpixel refinement bounds and preservation."""
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 10, 255, -1)
    
    pts = np.array([[50.0, 50.0], [10.0, 10.0]], dtype=np.float32)
    refined = subpixel_refine_points(img, pts)
    assert refined.shape == (2, 2)
    assert np.all(np.isfinite(refined))


def test_synthetic_generator_and_ground_truth():
    """Verify synthetic dataset generator creates valid transformed pairs with exact H_gt."""
    base = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(base, (40, 40), (160, 160), 200, -1)
    cv2.circle(base, (100, 100), 30, 100, -1)

    gen = SyntheticDatasetGenerator(random_seed=42)
    source, ref, H_gt, meta = gen.generate_pair(base, scenario="rotation")

    assert source.shape == (200, 200)
    assert is_valid_homography(H_gt)
    assert "angle_deg" in meta

    # Perfect H_est should yield 0.0 corner error
    err = compute_ground_truth_corner_error(H_gt, H_gt, (200, 200))
    assert err < 1e-5


def test_registration_pipeline_explicit_failure(tmp_path):
    """Verify pipeline rejects unmatchable pairs explicitly without silent identity fallback."""
    # Create two completely blank / uncorrelated images
    img1_path = os.path.join(tmp_path, "blank1.png")
    img2_path = os.path.join(tmp_path, "blank2.png")
    cv2.imwrite(img1_path, np.zeros((200, 200), dtype=np.uint8))
    cv2.imwrite(img2_path, np.ones((200, 200), dtype=np.uint8) * 255)

    job_dir = os.path.join(tmp_path, "job_fail")
    res = run_registration_pipeline(img1_path, img2_path, job_dir=job_dir)

    assert res["status"] == "failed"
    assert res["reason"] in ("INSUFFICIENT_CORRESPONDENCES", "DEGENERATE_TRANSFORM", "LOW_INLIER_RATIO")
    assert "error" in res


def test_registration_pipeline_success(tmp_path):
    """Verify pipeline registers textured synthetic lunar pair successfully."""
    base = np.zeros((300, 300), dtype=np.uint8)
    # Add rich crater-like features
    cv2.circle(base, (150, 150), 60, 220, -1)
    cv2.circle(base, (80, 80), 30, 180, -1)
    cv2.circle(base, (220, 220), 40, 140, -1)
    cv2.circle(base, (220, 80), 25, 190, -1)

    gen = SyntheticDatasetGenerator(random_seed=99)
    source, ref, _, _ = gen.generate_pair(base, scenario="translation")

    src_path = os.path.join(tmp_path, "src.png")
    ref_path = os.path.join(tmp_path, "ref.png")
    cv2.imwrite(src_path, source)
    cv2.imwrite(ref_path, ref)

    job_dir = os.path.join(tmp_path, "job_success")
    res = run_registration_pipeline(src_path, ref_path, job_dir=job_dir, mode="fast")

    assert res["status"] == "completed"
    assert "registered_image_path" in res
    assert os.path.isfile(res["registered_image_path"])
    assert "metrics" in res
    assert res["metrics"]["test_rmse_px"] >= 0.0


def test_api_register_with_modes(tmp_path):
    """Test FastAPI /register endpoint with mode and sensor selections."""
    img = np.zeros((150, 150), dtype=np.uint8)
    cv2.circle(img, (75, 75), 40, 255, -1)
    img_path = os.path.join(tmp_path, "tile.png")
    cv2.imwrite(img_path, img)

    with open(img_path, "rb") as f_src, open(img_path, "rb") as f_ref:
        response = client.post(
            "/register",
            files={"source": ("src.png", f_src, "image/png"), "reference": ("ref.png", f_ref, "image/png")},
            data={"mode": "fast", "source_sensor": "TMC-2", "reference_sensor": "LROC / External"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["metrics"]["inlier_count"] >= 4
    assert data["input_metadata"]["source"]["selected_sensor"] == "TMC-2"
