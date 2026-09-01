import httpx
import os

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8001")

async def call_ml_register(source_bytes: bytes, source_name: str,
                            reference_bytes: bytes, reference_name: str) -> dict:
    files = {
        "source": (source_name or "source.png", source_bytes, "image/png"),
        "reference": (reference_name or "reference.png", reference_bytes, "image/png"),
    }
    # Allow generous timeout for ML registration on large lunar images
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{ML_SERVICE_URL}/register", files=files)
        resp.raise_for_status()
        return resp.json()
