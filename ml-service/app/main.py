"""
main.py — FastAPI microservice for Chandrayaan-2 lunar image registration.
Provides REST API endpoints that delegate computer vision pipeline tasks to modular engines.
"""
import logging
import os
import shutil
import time
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.pipeline.loader import load_image
from app.pipeline.registration import run_registration_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ml-service")

app = FastAPI(
    title="LunarMatch AI — ML Microservice",
    description="Multi-modal, Sun angle & scale-invariant correspondence engine for Chandrayaan-2 imagery",
    version="2.0.0",
)

STORAGE_DIR = os.getenv("STORAGE_DIR", "/data")
if not os.path.exists(STORAGE_DIR):
    try:
        os.makedirs(STORAGE_DIR, exist_ok=True)
    except Exception:
        STORAGE_DIR = "./data"
        os.makedirs(STORAGE_DIR, exist_ok=True)

try:
    JOB_RETENTION_SEC = int(os.getenv("JOB_RETENTION_SEC") or 60 * 60)
except ValueError:
    JOB_RETENTION_SEC = 60 * 60


def sweep_old_jobs(now: float | None = None) -> int:
    """Delete stale job directories older than JOB_RETENTION_SEC."""
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
    return {
        "status": "ok",
        "service": "lunar-registration-ml",
        "version": "2.0.0",
        "storage": STORAGE_DIR,
    }


_UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".xml", ".img", ".lbl"}
_LABEL_EXTENSIONS = {".xml", ".lbl"}


def _upload_ext(filename: str | None) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext in _UPLOAD_EXTENSIONS else ".png"


def _safe_filename(filename: str | None, fallback: str) -> str:
    raw = (filename or "").strip().replace("\\", "/")
    name = raw.rsplit("/", 1)[-1]
    ext = os.path.splitext(name)[1].lower()
    if not name or name in (".", "..") or ext not in _UPLOAD_EXTENSIONS:
        return fallback
    return name


async def _save_side(job_dir: str, side: str, main: UploadFile, label: UploadFile | None):
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
    mode: str = Form("auto"),
    source_sensor: str | None = Form(None),
    reference_sensor: str | None = Form(None),
):
    sweep_old_jobs()

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(STORAGE_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    src_bytes, src_main_path, src_label_path = await _save_side(job_dir, "source", source, source_label)
    ref_bytes, ref_main_path, ref_label_path = await _save_side(job_dir, "reference", reference, reference_label)

    if len(src_bytes) == 0 or len(ref_bytes) == 0:
        raise HTTPException(status_code=400, detail="One or both uploaded files are empty.")

    src_path = _pick_load_path(src_main_path, src_label_path)
    ref_path = _pick_load_path(ref_main_path, ref_label_path)

    # Decode uploaded images; raise HTTP 422 if decoding/format is invalid
    try:
        ref_img_u8, ref_meta = load_image(ref_path, band=band)
        src_img_u8, src_meta = load_image(
            src_path, band=band, stretch_bounds=ref_meta.get("stretch_bounds")
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {e}")

    result = run_registration_pipeline(
        src_path=src_path,
        ref_path=ref_path,
        job_dir=job_dir,
        band=band,
        mode=mode,
        src_img_u8=src_img_u8,
        src_meta=src_meta,
        ref_img_u8=ref_img_u8,
        ref_meta=ref_meta,
    )

    result["job_id"] = job_id
    if source_sensor and "input_metadata" in result and result["input_metadata"].get("source"):
        result["input_metadata"]["source"]["selected_sensor"] = source_sensor
    if reference_sensor and "input_metadata" in result and result["input_metadata"].get("reference"):
        result["input_metadata"]["reference"]["selected_sensor"] = reference_sensor

    return JSONResponse(result)
