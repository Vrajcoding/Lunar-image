import httpx
import os

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8001")

async def call_ml_register(source_bytes: bytes, source_name: str,
                            reference_bytes: bytes, reference_name: str,
                            source_label_bytes: bytes | None = None, source_label_name: str | None = None,
                            reference_label_bytes: bytes | None = None, reference_label_name: str | None = None,
                            ) -> dict:
    # Content-type left generic: ml-service dispatches by filename extension
    # (see app/pipeline/loader.py there), not by this header, and a real
    # upload here may be .tif/.xml/.img rather than always a PNG.
    files = {
        "source": (source_name or "source.png", source_bytes, "application/octet-stream"),
        "reference": (reference_name or "reference.png", reference_bytes, "application/octet-stream"),
    }
    # source_label / reference_label are optional detached labels (e.g. a
    # PDS4 .xml next to a .img) — forwarded only when the client sent one, so
    # a plain single-file PNG/TIFF upload behaves exactly as before.
    if source_label_bytes:
        files["source_label"] = (
            source_label_name or "source_label.xml", source_label_bytes, "application/octet-stream"
        )
    if reference_label_bytes:
        files["reference_label"] = (
            reference_label_name or "reference_label.xml", reference_label_bytes, "application/octet-stream"
        )
    # Allow generous timeout for ML registration on large lunar images
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{ML_SERVICE_URL}/register", files=files)
        resp.raise_for_status()
        return resp.json()
