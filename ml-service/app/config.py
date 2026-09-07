"""
config.py — Central configuration and environment settings for Lunar Registration ML service.
"""
import os

# ─── Device & Accelerator Configuration ──────────────────────────────────────
USE_GPU_ENV = os.getenv("USE_GPU", "").strip().lower()
DEVICE = os.getenv("DEVICE", "auto").strip().lower()

if USE_GPU_ENV in ("true", "1", "yes") and DEVICE == "auto":
    DEVICE = "cuda"
elif USE_GPU_ENV in ("false", "0", "no"):
    DEVICE = "cpu"

# ─── Pipeline Matcher Toggles ────────────────────────────────────────────────
LOFTR_ENABLED = os.getenv("LOFTR_ENABLED", "true").strip().lower() in ("true", "1", "yes")
SIFT_FALLBACK_ENABLED = os.getenv("SIFT_FALLBACK_ENABLED", "true").strip().lower() in ("true", "1", "yes")
ORB_FALLBACK_ENABLED = os.getenv("ORB_FALLBACK_ENABLED", "true").strip().lower() in ("true", "1", "yes")

# ─── Algorithmic Hyperparameters ─────────────────────────────────────────────
LOFTR_CONFIDENCE_THRESHOLD = float(os.getenv("LOFTR_CONFIDENCE_THRESHOLD", "0.20"))
GRID_SIZE = int(os.getenv("GRID_SIZE", "8"))                 # N x N grid for uniform distribution
MAX_MATCHES_PER_CELL = int(os.getenv("MAX_MATCHES_PER_CELL", "15"))
RATIO_TEST_THRESHOLD = float(os.getenv("RATIO_TEST_THRESHOLD", "0.75"))   # Lowe's ratio test
RANSAC_REPROJ_THRESHOLD = float(os.getenv("RANSAC_REPROJ_THRESHOLD", "3.0")) # pixels, initial coarse pass
REFINE_REPROJ_THRESHOLD = float(os.getenv("REFINE_REPROJ_THRESHOLD", "1.5")) # pixels, sub-pixel refined pass
SIFT_N_FEATURES = int(os.getenv("SIFT_N_FEATURES", "5000"))
PYRAMID_LEVELS = int(os.getenv("PYRAMID_LEVELS", "4"))

# ─── Quality Acceptance Thresholds ───────────────────────────────────────────
MIN_RELIABLE_MATCHES = int(os.getenv("MIN_RELIABLE_MATCHES", "8"))
MIN_INLIER_COUNT = int(os.getenv("MIN_INLIER_COUNT", "6"))
MIN_INLIER_RATIO = float(os.getenv("MIN_INLIER_RATIO", "0.20"))
MAX_ACCEPTABLE_RMSE = float(os.getenv("MAX_ACCEPTABLE_RMSE", "15.0"))
