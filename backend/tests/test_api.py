"""
test_api.py — Backend API gateway tests.

Run from the ``backend/`` directory:

    venv\\Scripts\\python.exe -m pytest tests/ -q         (Windows)
    venv/bin/python -m pytest tests/ -q                   (*nix)

The ml-service is stubbed via monkeypatching ``app.routes.call_ml_register``,
so no network / Docker is required.
"""
import asyncio
import time

import httpx
import pytest

from app import routes
from tests.conftest import wait_for_status


# ===========================================================================
# 1. Health
# ===========================================================================

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_root_ok(self, client):
        assert client.get("/").status_code == 200


# ===========================================================================
# 2. Happy path: register -> status -> result -> download
# ===========================================================================

class TestRegisterHappyPath:
    def test_full_round_trip(self, client, monkeypatch, sample_png, fake_ml_outputs):
        async def fake_call(*args, **kwargs):
            return fake_ml_outputs

        monkeypatch.setattr(routes, "call_ml_register", fake_call)

        # --- register: must return a job_id *immediately* ---
        t0 = time.time()
        resp = client.post(
            "/api/register",
            files={
                "source": ("source.png", sample_png, "image/png"),
                "reference": ("reference.png", sample_png, "image/png"),
            },
        )
        elapsed = time.time() - t0
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "processing"
        job_id = body["job_id"]
        assert job_id
        assert elapsed < 2.0, f"/api/register blocked for {elapsed:.2f}s"

        # --- status flips to completed ---
        assert wait_for_status(client, job_id) == "completed"

        # --- result: metrics + download URLs ---
        result = client.get(f"/api/result/{job_id}")
        assert result.status_code == 200
        rbody = result.json()
        assert rbody["status"] == "completed"
        assert rbody["registered_image_url"] == f"/api/download/{job_id}/registered"
        assert rbody["match_points_url"] == f"/api/download/{job_id}/matches"
        assert rbody["metrics"]["rmse_px"] == pytest.approx(0.42)
        assert rbody["metrics"]["inlier_count"] == 187
        assert rbody["error"] is None

        # --- download: both files stream back with the right content ---
        reg = client.get(f"/api/download/{job_id}/registered")
        assert reg.status_code == 200
        assert reg.headers["content-type"] == "image/png"
        assert reg.content == b"\x89PNG\r\n\x1a\nFAKE-REGISTERED-IMAGE"

        csv = client.get(f"/api/download/{job_id}/matches")
        assert csv.status_code == 200
        assert csv.headers["content-type"].startswith("text/csv")
        assert csv.text.splitlines()[0] == "src_x,src_y,ref_x,ref_y,is_inlier"

    def test_result_before_completion_reports_processing(
        self, client, monkeypatch, sample_png
    ):
        async def slow_call(*args, **kwargs):
            await asyncio.sleep(0.5)
            return {
                "registered_image_path": "/nope",
                "match_points_path": "/nope",
                "metrics": {
                    "rmse_px": 0, "inlier_count": 0, "inlier_ratio": 0,
                    "uniformity_score": 0, "runtime_sec": 0,
                },
            }

        monkeypatch.setattr(routes, "call_ml_register", slow_call)
        job_id = client.post(
            "/api/register",
            files={
                "source": ("s.png", sample_png, "image/png"),
                "reference": ("r.png", sample_png, "image/png"),
            },
        ).json()["job_id"]

        # immediately after register, the job is still processing
        early = client.get(f"/api/result/{job_id}").json()
        assert early["status"] == "processing"
        assert early["metrics"] is None
        # and it does eventually finish
        assert wait_for_status(client, job_id) == "completed"


# ===========================================================================
# 3. ML service down / timing out  ->  job 'failed', backend stays alive
# ===========================================================================

class TestMlServiceUnavailable:
    def _register(self, client, sample_png):
        return client.post(
            "/api/register",
            files={
                "source": ("s.png", sample_png, "image/png"),
                "reference": ("r.png", sample_png, "image/png"),
            },
        )

    def test_ml_connection_refused_marks_job_failed(
        self, client, monkeypatch, sample_png
    ):
        async def boom(*args, **kwargs):
            raise httpx.ConnectError("All connection attempts failed")

        monkeypatch.setattr(routes, "call_ml_register", boom)

        t0 = time.time()
        resp = self._register(client, sample_png)
        assert resp.status_code == 200
        assert time.time() - t0 < 2.0, "register hung when ml-service was down"
        job_id = resp.json()["job_id"]

        assert wait_for_status(client, job_id) == "failed"
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "failed"
        assert result["error"]  # a non-empty error message is surfaced
        assert result["metrics"] is None

        # backend is still healthy after the failure
        assert client.get("/api/health").status_code == 200

    def test_ml_timeout_marks_job_failed(self, client, monkeypatch, sample_png):
        async def timeout(*args, **kwargs):
            raise httpx.ReadTimeout("timed out")

        monkeypatch.setattr(routes, "call_ml_register", timeout)

        job_id = self._register(client, sample_png).json()["job_id"]
        assert wait_for_status(client, job_id) == "failed"
        assert client.get(f"/api/result/{job_id}").json()["error"]

    def test_ml_http_500_marks_job_failed(self, client, monkeypatch, sample_png):
        async def http_error(*args, **kwargs):
            request = httpx.Request("POST", "http://ml-service:8001/register")
            response = httpx.Response(500, request=request, text="ml exploded")
            raise httpx.HTTPStatusError("500", request=request, response=response)

        monkeypatch.setattr(routes, "call_ml_register", http_error)

        job_id = self._register(client, sample_png).json()["job_id"]
        assert wait_for_status(client, job_id) == "failed"

    def test_ml_malformed_response_marks_job_failed(
        self, client, monkeypatch, sample_png
    ):
        async def missing_keys(*args, **kwargs):
            return {"status": "completed"}  # no paths, no metrics

        monkeypatch.setattr(routes, "call_ml_register", missing_keys)

        job_id = self._register(client, sample_png).json()["job_id"]
        assert wait_for_status(client, job_id) == "failed"


# ===========================================================================
# 4. Error handling for unknown jobs / bad input
# ===========================================================================

class TestErrorHandling:
    def test_status_unknown_job_404(self, client):
        assert client.get("/api/status/does-not-exist").status_code == 404

    def test_result_unknown_job_404(self, client):
        assert client.get("/api/result/does-not-exist").status_code == 404

    def test_download_unknown_job_404(self, client):
        assert client.get("/api/download/does-not-exist/registered").status_code == 404

    def test_register_missing_file_422(self, client, sample_png):
        # only 'source' supplied, 'reference' missing
        resp = client.post(
            "/api/register",
            files={"source": ("s.png", sample_png, "image/png")},
        )
        assert resp.status_code == 422

    def test_download_invalid_file_type(
        self, client, monkeypatch, sample_png, fake_ml_outputs
    ):
        async def fake_call(*args, **kwargs):
            return fake_ml_outputs

        monkeypatch.setattr(routes, "call_ml_register", fake_call)
        job_id = client.post(
            "/api/register",
            files={
                "source": ("s.png", sample_png, "image/png"),
                "reference": ("r.png", sample_png, "image/png"),
            },
        ).json()["job_id"]
        assert wait_for_status(client, job_id) == "completed"

        resp = client.get(f"/api/download/{job_id}/report")
        assert resp.status_code == 400

    def test_download_before_completion_404(self, client, monkeypatch, sample_png):
        async def slow_call(*args, **kwargs):
            await asyncio.sleep(0.3)
            return {
                "registered_image_path": "/nope",
                "match_points_path": "/nope",
                "metrics": {
                    "rmse_px": 0, "inlier_count": 0, "inlier_ratio": 0,
                    "uniformity_score": 0, "runtime_sec": 0,
                },
            }

        monkeypatch.setattr(routes, "call_ml_register", slow_call)
        job_id = client.post(
            "/api/register",
            files={
                "source": ("s.png", sample_png, "image/png"),
                "reference": ("r.png", sample_png, "image/png"),
            },
        ).json()["job_id"]
        # download attempted while still processing
        assert client.get(f"/api/download/{job_id}/registered").status_code == 404
