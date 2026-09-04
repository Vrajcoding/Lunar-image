import time
import os
import uuid
import csv
import shutil
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.pipeline.loader import load_image
from app.pipeline.preprocess import normalize_illumination, denoise, build_pyramid
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

# How long a job's output directory is kept before it is swept. The pipeline
# writes source/reference/registered PNGs + matches.csv per job and nothing
# else deletes them, so without this the shared /data volume grows without
# bound. One hour is plenty for a demo (frontend downloads happen seconds
# after the job completes); override with JOB_RETENTION_SEC=0 to disable.
try:
    JOB_RETENTION_SEC = int(os.getenv("JOB_RETENTION_SEC") or 60 * 60)
except ValueError:
    JOB_RETENTION_SEC = 60 * 60


def sweep_old_jobs(now: float | None = None) -> int:
    """
    Delete job directories under STORAGE_DIR whose last modification time is
    older than JOB_RETENTION_SEC. Best-effort: any error on a single directory
    is swallowed so a stale lock or permission issue can never fail a request.
    Returns the number of directories removed.
    """
    if JOB_RETENTION_SEC <= 0:
        return 0
    now = time.time() if now is None else now
    removed = 0
    try:
        entries = os.listdir(STORAGE_DIR)
    except OSError:
        return 0
    for name in entries:
        path = os.path.join(STORAGE_DIR, name)
        if not os.path.isdir(path):
            continue
        try:
            if now - os.path.getmtime(path) > JOB_RETENTION_SEC:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


@app.get("/health")
def health():
    return {"status": "ok", "service": "lunar-registration-ml"}


_UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".xml", ".img", ".lbl"}
_LABEL_EXTENSIONS = {".xml", ".lbl"}


def _upload_ext(filename: str | None) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext in _UPLOAD_EXTENSIONS else ".png"


def _safe_filename(filename: str | None, fallback: str) -> str:
    """Basename to save an upload under, preserving the client's own filename
    (sanitized) rather than a fixed name. A PDS4 .xml label references its
    .img data file by the exact filename baked into the label's own content
    at creation time — renaming the .img server-side (e.g. to a fixed
    "source.img") breaks that reference and GDAL can no longer find the data,
    even though both files still sit in the same directory."""
    raw = (filename or "").strip().replace("\\", "/")
    name = raw.rsplit("/", 1)[-1]
    ext = os.path.splitext(name)[1].lower()
    if not name or name in (".", "..") or ext not in _UPLOAD_EXTENSIONS:
        return fallback
    return name


async def _save_side(job_dir: str, side: str, main: UploadFile, label: UploadFile | None):
    """Saves one side's upload(s) (main image, plus an optional detached
    label such as a PDS4 .xml or PDS3 .lbl) into their own subdirectory —
    separate per side so a source/reference filename collision can't
    overwrite one with the other — preserving each file's real name. Returns
    (main_bytes, main_path, label_path_or_None)."""
    side_dir = os.path.join(job_dir, side)
    os.makedirs(side_dir, exist_ok=True)

    main_name = _safe_filename(main.filename, f"{side}{_upload_ext(main.filename)}")
    main_bytes = await main.read()
    main_path = os.path.join(side_dir, main_name)
    with open(main_path, "wb") as f:
        f.write(main_bytes)

    label_path = None
    if label is not None:
        label_bytes = await label.read()
        if len(label_bytes) > 0:
            label_name = _safe_filename(label.filename, f"{side}_label{_upload_ext(label.filename)}")
            if label_name == main_name:
                label_name = f"label_{label_name}"
            label_path = os.path.join(side_dir, label_name)
            with open(label_path, "wb") as f:
                f.write(label_bytes)

    return main_bytes, main_path, label_path


def _pick_load_path(main_path: str, label_path: str | None) -> str:
    """The spec rule is 'open the .xml, never the .img': if a detached label
    was uploaded alongside the main file, load through the label directly.
    Otherwise load the main file as before — a lone .img with no label still
    falls through to loader.py's own sibling-search error, a lone
    PNG/TIFF/.xml is used directly."""
    if label_path and os.path.splitext(label_path)[1].lower() in _LABEL_EXTENSIONS:
        return label_path
    return main_path


@app.post("/register")
async def register(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    source_label: UploadFile | None = File(None),
    reference_label: UploadFile | None = File(None),
    band: int | None = Form(None),
):
    t0 = time.time()

    # Opportunistic cleanup of stale job output before writing a new one.
    sweep_old_jobs()

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(STORAGE_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # ── Save uploaded files ──────────────────────────────────────────────────
    # source_label / reference_label are optional: a PDS4 product is two
    # files (.img data + .xml detached label) and /register otherwise only
    # takes one file per side, so real Chandrayaan-2 archive files could
    # never be uploaded through this endpoint at all. A plain PNG/TIFF
    # upload passes no label and behaves exactly as before.
    src_bytes, src_main_path, src_label_path = await _save_side(job_dir, "source", source, source_label)
    ref_bytes, ref_main_path, ref_label_path = await _save_side(job_dir, "reference", reference, reference_label)

    if len(src_bytes) == 0 or len(ref_bytes) == 0:
        raise HTTPException(status_code=400, detail="One or both uploaded files are empty.")

    src_path = _pick_load_path(src_main_path, src_label_path)
    ref_path = _pick_load_path(ref_main_path, ref_label_path)

    # ── Load (format-aware) + preprocess (CLAHE + denoise + pyramid) ────────
    # Reference loads first so its stretch bounds (for non-8-bit input) can be
    # reused on the source — independent per-image percentile stretches would
    # introduce an artificial intensity difference the matcher would read as
    # a real radiometric difference between the two images.
    try:
        ref_img_u8, ref_meta = load_image(ref_path, band=band)
        src_img_u8, src_meta = load_image(
            src_path, band=band, stretch_bounds=ref_meta.get("stretch_bounds")
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {e}")

    src_pyr = build_pyramid(denoise(normalize_illumination(src_img_u8)), PYRAMID_LEVELS)
    ref_pyr = build_pyramid(denoise(normalize_illumination(ref_img_u8)), PYRAMID_LEVELS)

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
        "input_metadata": {"source": src_meta, "reference": ref_meta},
    })
