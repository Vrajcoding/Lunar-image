import uuid
import os
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.jobs import create_job, update_job, get_job
from app.ml_client import call_ml_register
from app.schemas import RegisterAccepted, ResultResponse, Metrics

router = APIRouter(prefix="/api")

@router.get("/health")
def health():
    return {"status": "ok", "service": "lunar-registration-backend"}

@router.post("/register", response_model=RegisterAccepted)
async def register(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    source_label: UploadFile | None = File(None),
    reference_label: UploadFile | None = File(None),
):
    job_id = str(uuid.uuid4())
    create_job(job_id)

    source_bytes = await source.read()
    reference_bytes = await reference.read()
    source_label_bytes = await source_label.read() if source_label is not None else None
    reference_label_bytes = await reference_label.read() if reference_label is not None else None

    asyncio.create_task(
        process_job(
            job_id,
            source_bytes, source.filename,
            reference_bytes, reference.filename,
            source_label_bytes, source_label.filename if source_label is not None else None,
            reference_label_bytes, reference_label.filename if reference_label is not None else None,
        )
    )
    return RegisterAccepted(job_id=job_id, status="processing")

async def process_job(
    job_id: str,
    source_bytes: bytes, source_name: str,
    reference_bytes: bytes, reference_name: str,
    source_label_bytes: bytes | None = None, source_label_name: str | None = None,
    reference_label_bytes: bytes | None = None, reference_label_name: str | None = None,
):
    try:
        result = await call_ml_register(
            source_bytes, source_name,
            reference_bytes, reference_name,
            source_label_bytes, source_label_name,
            reference_label_bytes, reference_label_name,
        )
        update_job(job_id, {
            "status": "completed",
            "registered_image_path": result["registered_image_path"],
            "match_points_path": result["match_points_path"],
            "metrics": result["metrics"],
        })
    except Exception as e:
        update_job(job_id, {"status": "failed", "error": str(e)})

@router.get("/status/{job_id}")
def status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, "status": job["status"]}

@router.get("/result/{job_id}", response_model=ResultResponse)
def result(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "failed":
        return ResultResponse(job_id=job_id, status="failed", error=job.get("error", "Unknown error during registration"))
    if job["status"] != "completed":
        return ResultResponse(job_id=job_id, status=job["status"])
    
    return ResultResponse(
        job_id=job_id,
        status="completed",
        registered_image_url=f"/api/download/{job_id}/registered",
        match_points_url=f"/api/download/{job_id}/matches",
        metrics=Metrics(**job["metrics"]),
    )

@router.get("/download/{job_id}/{file_type}")
def download(job_id: str, file_type: str):
    job = get_job(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "Result not ready or job not found")
    
    path_map = {
        "registered": job.get("registered_image_path"),
        "matches": job.get("match_points_path"),
    }
    
    if file_type not in path_map:
        raise HTTPException(400, "Invalid file_type requested")
        
    file_path = path_map[file_type]
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, f"File for {file_type} not found at {file_path}")
        
    media_type = "image/png" if file_type == "registered" else "text/csv"
    filename = f"lunar_registered_{job_id[:8]}.png" if file_type == "registered" else f"match_points_{job_id[:8]}.csv"
    
    return FileResponse(file_path, media_type=media_type, filename=filename)
