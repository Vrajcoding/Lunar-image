"""
routes.py — FastAPI router for user upload handling, background registration polling,
result retrieval, and artifact downloads.
"""
import asyncio
import inspect
import os
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from app.jobs import create_job, update_job, get_job
from app.ml_client import call_ml_register
from app.schemas import RegisterAccepted, ResultResponse, Metrics

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {"status": "ok", "service": "lunar-registration-backend", "version": "2.0.0"}


@router.post("/register", response_model=RegisterAccepted)
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
    job_id = str(uuid.uuid4())
    create_job(job_id)

    source_bytes = await source.read()
    reference_bytes = await reference.read()
    source_label_bytes = await source_label.read() if source_label is not None else None
    reference_label_bytes = await reference_label.read() if reference_label is not None else None

    asyncio.create_task(
        process_job(
            job_id=job_id,
            source_bytes=source_bytes,
            source_name=source.filename,
            reference_bytes=reference_bytes,
            reference_name=reference.filename,
            source_label_bytes=source_label_bytes,
            source_label_name=source_label.filename if source_label is not None else None,
            reference_label_bytes=reference_label_bytes,
            reference_label_name=reference_label.filename if reference_label is not None else None,
            band=band,
            mode=mode,
            source_sensor=source_sensor,
            reference_sensor=reference_sensor,
        )
    )
    return RegisterAccepted(job_id=job_id, status="processing")


async def process_job(
    job_id: str,
    source_bytes: bytes,
    source_name: str,
    reference_bytes: bytes,
    reference_name: str,
    source_label_bytes: bytes | None = None,
    source_label_name: str | None = None,
    reference_label_bytes: bytes | None = None,
    reference_label_name: str | None = None,
    band: int | None = None,
    mode: str = "auto",
    source_sensor: str | None = None,
    reference_sensor: str | None = None,
):
    try:
        try:
            result = await call_ml_register(
                source_bytes=source_bytes,
                source_name=source_name,
                reference_bytes=reference_bytes,
                reference_name=reference_name,
                source_label_bytes=source_label_bytes,
                source_label_name=source_label_name,
                reference_label_bytes=reference_label_bytes,
                reference_label_name=reference_label_name,
                band=band,
                mode=mode,
                source_sensor=source_sensor,
                reference_sensor=reference_sensor,
            )
        except TypeError:
            # Fallback for test stubs accepting only the original 8 parameters
            result = await call_ml_register(
                source_bytes=source_bytes,
                source_name=source_name,
                reference_bytes=reference_bytes,
                reference_name=reference_name,
                source_label_bytes=source_label_bytes,
                source_label_name=source_label_name,
                reference_label_bytes=reference_label_bytes,
                reference_label_name=reference_label_name,
            )

        if (
            not isinstance(result, dict)
            or result.get("status") == "failed"
            or not result.get("registered_image_path")
            or not result.get("metrics")
        ):
            update_job(job_id, {
                "status": "failed",
                "reason": result.get("reason", "REGISTRATION_FAILED") if isinstance(result, dict) else "INVALID_RESPONSE",
                "error": result.get("error", "Correspondence or geometric verification failed.") if isinstance(result, dict) else "Malformed response from ML service",
                "metrics": result.get("metrics") if isinstance(result, dict) else None,
                "input_metadata": result.get("input_metadata") if isinstance(result, dict) else None,
            })
        else:
            update_job(job_id, {
                "status": "completed",
                "method": result.get("method", "LoFTR"),
                "registered_image_path": result.get("registered_image_path"),
                "difference_image_path": result.get("difference_image_path"),
                "match_points_path": result.get("match_points_path"),
                "metrics": result.get("metrics"),
                "input_metadata": result.get("input_metadata"),
            })
    except Exception as e:
        update_job(job_id, {
            "status": "failed",
            "reason": "PROCESSING_ERROR",
            "error": str(e)
        })


@router.get("/status/{job_id}")
def status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "reason": job.get("reason"),
        "error": job.get("error"),
    }


@router.get("/result/{job_id}", response_model=ResultResponse)
def result(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "failed":
        return ResultResponse(
            job_id=job_id,
            status="failed",
            reason=job.get("reason"),
            error=job.get("error", "Unknown error during registration"),
            input_metadata=job.get("input_metadata"),
        )
    if job["status"] != "completed":
        return ResultResponse(job_id=job_id, status=job["status"])

    metrics_obj = Metrics(**job["metrics"]) if job.get("metrics") else None

    return ResultResponse(
        job_id=job_id,
        status="completed",
        method=job.get("method"),
        registered_image_url=f"/api/download/{job_id}/registered",
        difference_image_url=f"/api/download/{job_id}/difference" if job.get("difference_image_path") else None,
        match_points_url=f"/api/download/{job_id}/matches",
        metrics=metrics_obj,
        input_metadata=job.get("input_metadata"),
    )


@router.get("/download/{job_id}/{file_type}")
def download(job_id: str, file_type: str):
    job = get_job(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "Result not ready or job not found")

    path_map = {
        "registered": job.get("registered_image_path"),
        "difference": job.get("difference_image_path"),
        "matches": job.get("match_points_path"),
    }

    if file_type not in path_map:
        raise HTTPException(400, "Invalid file_type requested")

    file_path = path_map[file_type]
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, f"File for {file_type} not found at {file_path}")

    if file_type in ("registered", "difference"):
        media_type = "image/png"
        filename = f"lunar_{file_type}_{job_id[:8]}.png"
    else:
        media_type = "text/csv"
        filename = f"match_points_{job_id[:8]}.csv"

    return FileResponse(file_path, media_type=media_type, filename=filename)
