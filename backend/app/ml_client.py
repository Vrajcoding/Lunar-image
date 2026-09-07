"""
ml_client.py — Asynchronous HTTP client communicating with the ML computer vision microservice.
"""
import os
import httpx

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8001")


async def call_ml_register(
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
) -> dict:
    """Send multipart registration request to the ML service."""
    files = {
        "source": (source_name or "source.png", source_bytes, "application/octet-stream"),
        "reference": (reference_name or "reference.png", reference_bytes, "application/octet-stream"),
    }
    if source_label_bytes:
        files["source_label"] = (
            source_label_name or "source_label.xml",
            source_label_bytes,
            "application/octet-stream",
        )
    if reference_label_bytes:
        files["reference_label"] = (
            reference_label_name or "reference_label.xml",
            reference_label_bytes,
            "application/octet-stream",
        )

    data = {
        "mode": mode or "auto",
    }
    if band is not None:
        data["band"] = str(band)
    if source_sensor:
        data["source_sensor"] = source_sensor
    if reference_sensor:
        data["reference_sensor"] = reference_sensor

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{ML_SERVICE_URL}/register", files=files, data=data)
        resp.raise_for_status()
        return resp.json()
