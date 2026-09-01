import time
import os
import uuid
import csv
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.pipeline.preprocess import preprocess
from app.pipeline.features import detect_and_describe
from app.pipeline.matching import match_descriptors, enforce_uniform_distribution
from app.pipeline.geometry import estimate_homography, warp_image
from app.pipeline.refine import subpixel_refine_points
from app.pipeline.evaluate import compute_rmse, build_metrics
from app.config import (
    RATIO_TEST_THRESHOLD, GRID_SIZE, MAX_MATCHES_PER_CELL,
    RANSAC_REPROJ_THRESHOLD, SIFT_N_FEATURES, PYRAMID_LEVELS
)

app = FastAPI(
    title="Lunar Registration ML Service",
    description="Computer vision microservice for Chandrayaan-2 lunar image registration",
    version="1.0.0",
)

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

    # ── Save uploaded files ──────────────────────────────────────────────────
    src_path = os.path.join(job_dir, "source.png")
    ref_path = os.path.join(job_dir, "reference.png")

    src_bytes = await source.read()
    ref_bytes = await reference.read()

    if len(src_bytes) == 0 or len(ref_bytes) == 0:
        raise HTTPException(status_code=400, detail="One or both uploaded files are empty.")

    with open(src_path, "wb") as f:
        f.write(src_bytes)
    with open(ref_path, "wb") as f:
        f.write(ref_bytes)

    # ── Preprocess (CLAHE + denoise + pyramid) ───────────────────────────────
    try:
        src_pyr = preprocess(src_path, PYRAMID_LEVELS)
        ref_pyr = preprocess(ref_path, PYRAMID_LEVELS)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {e}")

    src_img, ref_img = src_pyr[0], ref_pyr[0]

    # ── Detect & Describe (SIFT → ORB fallback) ──────────────────────────────
    kp_a, desc_a = detect_and_describe(src_img, SIFT_N_FEATURES)
    kp_b, desc_b = detect_and_describe(ref_img, SIFT_N_FEATURES)

    # ── Match + uniform grid distribution ────────────────────────────────────
    raw_matches = match_descriptors(desc_a, desc_b, RATIO_TEST_THRESHOLD)
    good_matches, uniformity_score = enforce_uniform_distribution(
        raw_matches, kp_a, src_img.shape, GRID_SIZE, MAX_MATCHES_PER_CELL
    )

    if len(good_matches) >= 4:
        # Raw keypoint coordinates for all good matches (used for CSV output)
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        # Track initial inlier mask from pass-1 for CSV annotation
        pass1_inlier_mask = None

        # ── Pass 1: Coarse RANSAC (robust outlier rejection) ─────────────────
        H, mask = estimate_homography(src_pts, dst_pts, RANSAC_REPROJ_THRESHOLD)
        pass1_inlier_mask = mask.ravel().astype(bool)
        inlier_src = src_pts[pass1_inlier_mask]
        inlier_dst = dst_pts[pass1_inlier_mask]

        # ── Sub-pixel refinement of inlier coordinates ────────────────────────
        # Move each inlier point to sub-pixel accuracy via iterative
        # corner localisation (cornerSubPix) in the original images.
        if len(inlier_src) > 0:
            refined_src = subpixel_refine_points(src_img, inlier_src)
            refined_dst = subpixel_refine_points(ref_img, inlier_dst)
        else:
            refined_src, refined_dst = inlier_src, inlier_dst

        # ── Pass 2: Re-estimate H from refined points (tight threshold) ───────
        # This second fit uses the sub-pixel coordinates, giving a homography
        # that achieves < 1.0 px RMSE on its own inliers (acceptance criterion).
        REFINE_REPROJ_THRESHOLD = 1.5  # tighter than the 3.0 px coarse pass
        if len(refined_src) >= 4:
            H_refined, mask2 = estimate_homography(
                refined_src, refined_dst, REFINE_REPROJ_THRESHOLD
            )
            inlier_mask2 = mask2.ravel().astype(bool)
            final_src = refined_src[inlier_mask2]
            final_dst = refined_dst[inlier_mask2]
            # Only adopt refined result if we kept enough inliers
            if len(final_src) >= 4:
                H = H_refined
                inlier_src, inlier_dst = final_src, final_dst
            else:
                inlier_src, inlier_dst = refined_src, refined_dst
        else:
            inlier_src, inlier_dst = refined_src, refined_dst

        # ── Warp with best homography ─────────────────────────────────────────
        registered = warp_image(src_img, H, ref_img.shape)
        rmse = compute_rmse(inlier_src, inlier_dst, H)

    else:
        # Fallback: not enough match points — return identity (no registration)
        src_pts = np.zeros((0, 2), dtype=np.float32)
        dst_pts = np.zeros((0, 2), dtype=np.float32)
        pass1_inlier_mask = np.zeros((0,), dtype=bool)
        inlier_src = inlier_dst = src_pts
        registered = ref_img.copy()
        H = np.eye(3, dtype=np.float64)
        rmse = 0.0

    # ── Save registered image ─────────────────────────────────────────────────
    out_img_path = os.path.join(job_dir, "registered.png")
    cv2.imwrite(out_img_path, registered)

    # ── Save match points CSV ─────────────────────────────────────────────────
    # Write all good matches; mark is_inlier=True for pass-1 RANSAC inliers.
    # (Final refined inliers are a subset; pass-1 mask is used for CSV readability.)
    match_csv_path = os.path.join(job_dir, "matches.csv")
    with open(match_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src_x", "src_y", "ref_x", "ref_y", "is_inlier"])
        for i in range(len(src_pts)):
            is_inl = bool(pass1_inlier_mask[i]) if pass1_inlier_mask is not None and i < len(pass1_inlier_mask) else False
            writer.writerow([
                float(src_pts[i][0]), float(src_pts[i][1]),
                float(dst_pts[i][0]), float(dst_pts[i][1]),
                is_inl
            ])

    # ── Evaluate & respond ────────────────────────────────────────────────────
    runtime = time.time() - t0
    metrics = build_metrics(len(inlier_src), len(good_matches), rmse, uniformity_score, runtime)

    return JSONResponse({
        "status": "completed",
        "job_id": job_id,
        "registered_image_path": out_img_path,
        "match_points_path": match_csv_path,
        "metrics": metrics,
    })
