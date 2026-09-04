"""
conftest.py — Pytest fixtures for the Lunar Registration Backend (API gateway).

These tests exercise the backend in isolation: the ML microservice is replaced
with a stub (monkeypatched ``call_ml_register``) so the suite is fast, offline,
and deterministic. The goal is to prove the gateway's own contract:

  * ``POST /api/register`` returns a ``job_id`` immediately (never blocks on ML)
  * ``GET  /api/status/{job_id}`` tracks ``processing -> completed / failed``
  * ``GET  /api/result/{job_id}`` returns metrics + download URLs when done
  * ``GET  /api/download/{job_id}/{file_type}`` streams the output files
  * the ML service being down / slow / broken surfaces as ``failed`` — the
    backend does not hang and does not 500
"""
import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

# --- make the ``app`` package importable (backend/ is the project root here) ---
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.main import app  # noqa: E402
from app import jobs as jobs_module  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_jobs():
    """Every test starts with an empty job store."""
    with jobs_module._lock:
        jobs_module._jobs.clear()
    yield
    with jobs_module._lock:
        jobs_module._jobs.clear()


@pytest.fixture
def sample_png():
    """
    Dummy upload payload. The backend is a pass-through gateway — it never
    decodes the image itself (that is the ml-service's job, and the ml-service
    is stubbed here) — so any non-empty bytes stand in for a real lunar tile.
    """
    return b"\x89PNG\r\n\x1a\n" + b"pretend-this-is-a-lunar-image" * 32


@pytest.fixture
def fake_ml_outputs(tmp_path):
    """
    Create the files a real ml-service run would have written to the shared
    /data volume, and return the dict its ``/register`` endpoint returns.
    """
    job_dir = tmp_path / "ml_job"
    job_dir.mkdir()
    registered = job_dir / "registered.png"
    matches = job_dir / "matches.csv"
    registered.write_bytes(b"\x89PNG\r\n\x1a\nFAKE-REGISTERED-IMAGE")
    matches.write_text("src_x,src_y,ref_x,ref_y,is_inlier\n1.0,2.0,1.1,2.1,True\n")
    return {
        "status": "completed",
        "job_id": "ml-side-id",
        "registered_image_path": str(registered),
        "match_points_path": str(matches),
        "metrics": {
            "rmse_px": 0.42,
            "inlier_count": 187,
            "inlier_ratio": 0.81,
            "uniformity_score": 0.93,
            "runtime_sec": 4.7,
        },
    }


def wait_for_status(client, job_id, target=("completed", "failed"), timeout=10.0):
    """Poll /api/status until the job leaves 'processing' (or time runs out)."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/api/status/{job_id}")
        assert resp.status_code == 200
        last = resp.json()["status"]
        if last in target:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} stuck in '{last}' after {timeout}s")
