"""
registration.py — Central hybrid image registration orchestration engine.
Coordinates PDS4 loading, CLAHE preprocessing, LoFTR learned matching with SIFT/ORB fallbacks,
spatial entropy filtering, RANSAC homography, sub-pixel refinement, and independent RMSE validation.
"""
import csv
import logging
import os
import time
import cv2
import numpy as np

from app.pipeline.loader import load_image
from app.pipeline.preprocess import normalize_illumination, denoise, build_pyramid
from app.pipeline.features import detect_and_describe
from app.pipeline.matching import (
    match_descriptors,
    enforce_uniform_distribution,
    filter_dense_correspondences,
    compute_spatial_entropy,
)
from app.pipeline.geometry import estimate_homography, warp_image, is_valid_homography
from app.pipeline.refine import subpixel_refine_points
from app.pipeline.evaluate import compute_rmse, compute_train_test_rmse, build_metrics
from app.pipeline.confidence import compute_confidence_score
from app.models.model_loader import get_loftr_matcher
from app.config import (
    RATIO_TEST_THRESHOLD,
    GRID_SIZE,
    MAX_MATCHES_PER_CELL,
    RANSAC_REPROJ_THRESHOLD,
    REFINE_REPROJ_THRESHOLD,
    SIFT_N_FEATURES,
    PYRAMID_LEVELS,
    LOFTR_ENABLED,
    SIFT_FALLBACK_ENABLED,
    ORB_FALLBACK_ENABLED,
    LOFTR_CONFIDENCE_THRESHOLD,
    MIN_RELIABLE_MATCHES,
    MIN_INLIER_COUNT,
    MIN_INLIER_RATIO,
    MAX_ACCEPTABLE_RMSE,
)

logger = logging.getLogger(__name__)


def run_registration_pipeline(
    src_path: str,
    ref_path: str,
    job_dir: str,
    band: int | None = None,
    mode: str = "auto",
    src_img_u8: np.ndarray | None = None,
    src_meta: dict | None = None,
    ref_img_u8: np.ndarray | None = None,
    ref_meta: dict | None = None,
) -> dict:
    """Execute end-to-end hybrid registration."""
    t0 = time.time()
    os.makedirs(job_dir, exist_ok=True)

    # ── 1. Load images & extract metadata if not already loaded ───────────────
    if src_img_u8 is None or ref_img_u8 is None:
        try:
            ref_img_u8, ref_meta = load_image(ref_path, band=band)
            src_img_u8, src_meta = load_image(
                src_path, band=band, stretch_bounds=ref_meta.get("stretch_bounds")
            )
        except Exception as e:
            logger.error("Failed to load images: %s", e)
            raise ValueError(f"Could not decode image: {e}") from e

    # ── 2. Radiometric Preprocessing ──────────────────────────────────────────
    src_pyr = build_pyramid(denoise(normalize_illumination(src_img_u8)), PYRAMID_LEVELS)
    ref_pyr = build_pyramid(denoise(normalize_illumination(ref_img_u8)), PYRAMID_LEVELS)
    src_img, ref_img = src_pyr[0], ref_pyr[0]

    # ── 3. Hybrid Feature Correspondence (LoFTR -> SIFT -> ORB) ──────────────
    method_used = "UNKNOWN"
    src_pts = np.zeros((0, 2), dtype=np.float32)
    dst_pts = np.zeros((0, 2), dtype=np.float32)
    conf_array = np.zeros((0,), dtype=np.float32)
    uniformity_score = 0.0
    spatial_entropy = 0.0

    # Phase A: LoFTR Learned Correspondence
    try_loftr = LOFTR_ENABLED and mode.lower() in ("auto", "accuracy", "loftr")
    if try_loftr:
        try:
            loftr = get_loftr_matcher()
            if loftr is not None and loftr.is_available:
                logger.info("Executing LoFTR correspondence matching...")
                l_res = loftr.match(src_img, ref_img, confidence_threshold=LOFTR_CONFIDENCE_THRESHOLD)
                l_pts0 = l_res.get("points0", np.zeros((0, 2), dtype=np.float32))
                l_pts1 = l_res.get("points1", np.zeros((0, 2), dtype=np.float32))
                l_conf = l_res.get("confidence", np.zeros((0,), dtype=np.float32))

                if len(l_pts0) >= MIN_RELIABLE_MATCHES:
                    f_pts0, f_pts1, f_conf, u_score, s_entropy = filter_dense_correspondences(
                        l_pts0, l_pts1, l_conf, src_img.shape, GRID_SIZE, MAX_MATCHES_PER_CELL
                    )
                    if len(f_pts0) >= 4:
                        method_used = "LoFTR"
                        src_pts, dst_pts, conf_array = f_pts0, f_pts1, f_conf
                        uniformity_score, spatial_entropy = u_score, s_entropy
                        logger.info("LoFTR correspondence succeeded with %d filtered matches", len(src_pts))
        except Exception as e:
            logger.warning("LoFTR matching raised exception: %s, falling back to classical", e)

    # Phase B: SIFT Classical Fallback (or if mode == "fast")
    if (len(src_pts) < 4 and SIFT_FALLBACK_ENABLED) or mode.lower() == "fast":
        logger.info("Engaging SIFT classical pipeline...")
        try:
            kp_a, desc_a = detect_and_describe(src_img, SIFT_N_FEATURES)
            kp_b, desc_b = detect_and_describe(ref_img, SIFT_N_FEATURES)
            raw_matches = match_descriptors(desc_a, desc_b, RATIO_TEST_THRESHOLD)
            good_matches, u_score = enforce_uniform_distribution(
                raw_matches, kp_a, src_img.shape, GRID_SIZE, MAX_MATCHES_PER_CELL
            )
            if len(good_matches) >= 4:
                method_used = "SIFT"
                src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                conf_array = np.float32([max(1.0 - m.distance / 200.0, 0.1) for m in good_matches])
                uniformity_score = u_score
                _, spatial_entropy, _ = compute_spatial_entropy(src_pts, src_img.shape, GRID_SIZE)
                logger.info("SIFT established %d matches", len(src_pts))
        except Exception as e:
            logger.warning("SIFT detection/matching failed: %s", e)

    # Phase C: ORB Secondary Fallback
    if len(src_pts) < 4 and ORB_FALLBACK_ENABLED:
        logger.info("Engaging ORB secondary fallback...")
        try:
            orb = cv2.ORB_create(nfeatures=SIFT_N_FEATURES)
            kp_a, desc_a = orb.detectAndCompute(src_img, None)
            kp_b, desc_b = orb.detectAndCompute(ref_img, None)
            raw_matches = match_descriptors(desc_a, desc_b, RATIO_TEST_THRESHOLD)
            good_matches, u_score = enforce_uniform_distribution(
                raw_matches, kp_a, src_img.shape, GRID_SIZE, MAX_MATCHES_PER_CELL
            )
            if len(good_matches) >= 4:
                method_used = "ORB"
                src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                conf_array = np.float32([max(1.0 - m.distance / 100.0, 0.1) for m in good_matches])
                uniformity_score = u_score
                _, spatial_entropy, _ = compute_spatial_entropy(src_pts, src_img.shape, GRID_SIZE)
                logger.info("ORB established %d matches", len(src_pts))
        except Exception as e:
            logger.warning("ORB matching failed: %s", e)

    # ── 4. Correspondence Verification (Reject on Failure) ────────────────────
    if len(src_pts) < 4:
        logger.error("Registration rejected: Insufficient correspondences found across all matchers.")
        return {
            "status": "failed",
            "reason": "INSUFFICIENT_CORRESPONDENCES",
            "error": "Could not establish at least 4 reliable correspondences across lunar image pair.",
            "metrics": build_metrics(0, len(src_pts), 0.0, uniformity_score, time.time() - t0, method="None", test_rmse=0.0, spatial_entropy=0.0, confidence_score=0.0, confidence_level="LOW"),
            "input_metadata": {"source": src_meta, "reference": ref_meta},
        }

    # ── 5. Pass 1: Coarse RANSAC Homography ───────────────────────────────────
    H, mask = estimate_homography(src_pts, dst_pts, RANSAC_REPROJ_THRESHOLD)
    pass1_mask = mask.ravel().astype(bool)
    inlier_src = src_pts[pass1_mask]
    inlier_dst = dst_pts[pass1_mask]

    inlier_count = len(inlier_src)
    inlier_ratio = inlier_count / max(len(src_pts), 1)

    if inlier_count < 4 or not is_valid_homography(H):
        logger.error("Registration rejected: Degenerate geometric transformation or insufficient inliers.")
        return {
            "status": "failed",
            "reason": "DEGENERATE_TRANSFORM",
            "error": "RANSAC could not compute a non-singular geometric homography.",
            "metrics": build_metrics(inlier_count, len(src_pts), 0.0, uniformity_score, time.time() - t0, method=method_used, test_rmse=0.0, spatial_entropy=spatial_entropy, confidence_score=0.0, confidence_level="LOW"),
            "input_metadata": {"source": src_meta, "reference": ref_meta},
        }

    if inlier_count < MIN_INLIER_COUNT or inlier_ratio < MIN_INLIER_RATIO:
        logger.error("Registration rejected: Low inlier ratio (inliers=%d, ratio=%.2f)", inlier_count, inlier_ratio)
        return {
            "status": "failed",
            "reason": "LOW_INLIER_RATIO",
            "error": f"Inlier count ({inlier_count}) or ratio ({inlier_ratio:.2f}) below scientific acceptance threshold.",
            "metrics": build_metrics(inlier_count, len(src_pts), 0.0, uniformity_score, time.time() - t0, method=method_used, test_rmse=0.0, spatial_entropy=spatial_entropy, confidence_score=0.0, confidence_level="LOW"),
            "input_metadata": {"source": src_meta, "reference": ref_meta},
        }

    # ── 6. Sub-pixel Refinement (Iterative Corner Localisation) ────────────────
    refined_src = subpixel_refine_points(src_img, inlier_src)
    refined_dst = subpixel_refine_points(ref_img, inlier_dst)

    # ── 7. Pass 2: Re-estimate Homography from Refined Points ─────────────────
    H_refined, mask2 = estimate_homography(refined_src, refined_dst, REFINE_REPROJ_THRESHOLD)
    pass2_mask = mask2.ravel().astype(bool)
    final_src = refined_src[pass2_mask]
    final_dst = refined_dst[pass2_mask]

    if len(final_src) >= 4 and is_valid_homography(H_refined):
        H = H_refined
        eval_src, eval_dst = final_src, final_dst
    else:
        eval_src, eval_dst = refined_src, refined_dst

    # ── 8. Image Warping & Alignment Difference ───────────────────────────────
    registered = warp_image(src_img, H, ref_img.shape)

    # Pixel-wise difference image
    diff_raw = cv2.absdiff(ref_img, registered)
    diff_vis = cv2.applyColorMap(diff_raw, cv2.COLORMAP_VIRIDIS)

    # ── 9. Independent Evaluation & Confidence ────────────────────────────────
    fit_rmse, test_rmse = compute_train_test_rmse(eval_src, eval_dst, H)

    conf_score, conf_level = compute_confidence_score(
        test_rmse, inlier_ratio, uniformity_score, len(eval_src)
    )

    runtime_sec = time.time() - t0
    metrics = build_metrics(
        inlier_count=len(eval_src),
        total_matches=len(src_pts),
        rmse=fit_rmse,
        uniformity_score=uniformity_score,
        runtime_sec=runtime_sec,
        test_rmse=test_rmse,
        spatial_entropy=spatial_entropy,
        confidence_score=conf_score,
        confidence_level=conf_level,
        method=method_used,
    )

    # ── 10. Persist Artifacts ─────────────────────────────────────────────────
    out_img_path = os.path.join(job_dir, "registered.png")
    cv2.imwrite(out_img_path, registered)

    diff_img_path = os.path.join(job_dir, "difference.png")
    cv2.imwrite(diff_img_path, diff_vis)

    match_csv_path = os.path.join(job_dir, "matches.csv")
    with open(match_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src_x", "src_y", "ref_x", "ref_y", "is_inlier"])
        for i in range(len(src_pts)):
            is_inl = bool(pass1_mask[i]) if i < len(pass1_mask) else False
            writer.writerow([
                float(src_pts[i][0]), float(src_pts[i][1]),
                float(dst_pts[i][0]), float(dst_pts[i][1]),
                is_inl
            ])

    return {
        "status": "completed",
        "method": method_used,
        "registered_image_path": out_img_path,
        "difference_image_path": diff_img_path,
        "match_points_path": match_csv_path,
        "metrics": metrics,
        "input_metadata": {"source": src_meta, "reference": ref_meta},
    }
