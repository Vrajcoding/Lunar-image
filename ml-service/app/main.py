import time
import os
import uuid
import csv
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from app.pipeline.preprocess import preprocess
from app.pipeline.features import detect_and_describe
from app.pipeline.matching import match_descriptors, enforce_uniform_distribution
from app.pipeline.geometry import estimate_homography, warp_image
from app.pipeline.evaluate import compute_rmse, build_metrics
from app.config import (
    RATIO_TEST_THRESHOLD, GRID_SIZE, MAX_MATCHES_PER_CELL,
    RANSAC_REPROJ_THRESHOLD, SIFT_N_FEATURES, PYRAMID_LEVELS
)

app = FastAPI(title="Lunar Registration ML Service")

STORAGE_DIR = os.getenv("STORAGE_DIR", "/data")
if not os.path.exists(STORAGE_DIR):
    try:
        os.makedirs(STORAGE_DIR, exist_ok=True)
    except Exception:
        STORAGE_DIR = "./data"
        os.makedirs(STORAGE_DIR, exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok", "service": "lunar-registration-ml"}

@app.post("/register")
async def register(source: UploadFile = File(...), reference: UploadFile = File(...)):
    t0 = time.time()
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(STORAGE_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    src_path = os.path.join(job_dir, "source.png")
    ref_path = os.path.join(job_dir, "reference.png")
    
    with open(src_path, "wb") as f:
        f.write(await source.read())
    with open(ref_path, "wb") as f:
        f.write(await reference.read())

    # Preprocess images
    src_pyr = preprocess(src_path, PYRAMID_LEVELS)
    ref_pyr = preprocess(ref_path, PYRAMID_LEVELS)
    src_img, ref_img = src_pyr[0], ref_pyr[0]

    # Detect & Describe
    kp_a, desc_a = detect_and_describe(src_img, SIFT_N_FEATURES)
    kp_b, desc_b = detect_and_describe(ref_img, SIFT_N_FEATURES)

    # Match & Uniform Distribution
    raw_matches = match_descriptors(desc_a, desc_b, RATIO_TEST_THRESHOLD)
    good_matches, uniformity_score = enforce_uniform_distribution(
        raw_matches, kp_a, src_img.shape, GRID_SIZE, MAX_MATCHES_PER_CELL
    )

    if len(good_matches) >= 4:
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Estimate Homography
        H, mask = estimate_homography(src_pts, dst_pts, RANSAC_REPROJ_THRESHOLD)
        inlier_mask = mask.ravel().astype(bool)
        inlier_src, inlier_dst = src_pts[inlier_mask], dst_pts[inlier_mask]

        # Warp
        registered = warp_image(src_img, H, ref_img.shape)
        rmse = compute_rmse(inlier_src, inlier_dst, H)
    else:
        # Fallback if insufficient match points
        src_pts = np.zeros((0, 2), dtype=np.float32)
        dst_pts = np.zeros((0, 2), dtype=np.float32)
        inlier_mask = np.zeros((0,), dtype=bool)
        inlier_src, inlier_dst = src_pts, dst_pts
        registered = ref_img.copy()
        H = np.eye(3)
        rmse = 0.0

    out_img_path = os.path.join(job_dir, "registered.png")
    cv2.imwrite(out_img_path, registered)

    # Save match points to CSV
    match_csv_path = os.path.join(job_dir, "matches.csv")
    with open(match_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src_x", "src_y", "ref_x", "ref_y", "is_inlier"])
        for i, m in enumerate(good_matches):
            is_inl = bool(inlier_mask[i]) if i < len(inlier_mask) else False
            writer.writerow([float(src_pts[i][0]), float(src_pts[i][1]), float(dst_pts[i][0]), float(dst_pts[i][1]), is_inl])

    runtime = time.time() - t0
    metrics = build_metrics(len(inlier_src), len(good_matches), rmse, uniformity_score, runtime)

    return JSONResponse({
        "status": "completed",
        "job_id": job_id,
        "registered_image_path": out_img_path,
        "match_points_path": match_csv_path,
        "metrics": metrics,
    })
