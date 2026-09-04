"""
test_register_multifile.py — /api/register with an optional detached label
file per side (source_label / reference_label), forwarded through to
ml-service. ml-service itself is stubbed (as in test_api.py) so this proves
the backend's own wiring: the label bytes/filenames actually reach
call_ml_register, absence of a label doesn't break anything, and a rejection
from ml-service (e.g. a lone .img with no label) surfaces as a failed job
rather than a hang — the same contract test_api.py already checks for other
ml-service failure modes.
"""
import httpx
import pytest

from app import routes
from tests.conftest import wait_for_status


# ---------------------------------------------------------------------------
# 1. PDS4-style pair (main + label on both sides) — args reach call_ml_register,
#    job completes
# ---------------------------------------------------------------------------

def test_pds4_pair_forwarded_and_completes(client, monkeypatch, fake_ml_outputs):
    captured = {}

    async def spy_call(source_bytes, source_name, reference_bytes, reference_name,
                        source_label_bytes=None, source_label_name=None,
                        reference_label_bytes=None, reference_label_name=None):
        captured.update(
            source_name=source_name, reference_name=reference_name,
            source_label_name=source_label_name, reference_label_name=reference_label_name,
            source_label_bytes=source_label_bytes, reference_label_bytes=reference_label_bytes,
        )
        return fake_ml_outputs

    monkeypatch.setattr(routes, "call_ml_register", spy_call)

    img_bytes = b"\x00" * 200
    xml_bytes = b"<?xml version='1.0'?><Product_Observational/>"

    resp = client.post(
        "/api/register",
        files={
            "source": ("ch2_tmc_nca_20200207T0716469418_d_img_d18.img", img_bytes, "application/octet-stream"),
            "source_label": ("ch2_tmc_nca_20200207T0716469418_d_img_d18.xml", xml_bytes, "application/xml"),
            "reference": ("ch2_tmc_ncn_20200207T0716469418_d_img_d19.img", img_bytes, "application/octet-stream"),
            "reference_label": ("ch2_tmc_ncn_20200207T0716469418_d_img_d19.xml", xml_bytes, "application/xml"),
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    assert wait_for_status(client, job_id) == "completed"

    assert captured["source_label_name"] == "ch2_tmc_nca_20200207T0716469418_d_img_d18.xml"
    assert captured["reference_label_name"] == "ch2_tmc_ncn_20200207T0716469418_d_img_d19.xml"
    assert captured["source_label_bytes"] == xml_bytes
    assert captured["reference_label_bytes"] == xml_bytes


# ---------------------------------------------------------------------------
# 2. .img with no label — ml-service's rejection surfaces as a failed job,
#    not a hang
# ---------------------------------------------------------------------------

def test_img_without_label_marks_job_failed(client, monkeypatch):
    async def rejects_like_real_ml_service(*args, **kwargs):
        request = httpx.Request("POST", "http://ml-service:8001/register")
        response = httpx.Response(
            422, request=request,
            text='{"detail":"Could not decode image: \'...img\' is a raw .img file with no detached label..."}',
        )
        raise httpx.HTTPStatusError("422 Unprocessable Entity", request=request, response=response)

    monkeypatch.setattr(routes, "call_ml_register", rejects_like_real_ml_service)

    import time
    t0 = time.time()
    resp = client.post(
        "/api/register",
        files={
            "source": ("ch2_ohrc_ncp_20210101T0000000000_d_img_x01.img", b"\x00" * 100, "application/octet-stream"),
            "reference": ("reference.png", b"\x89PNG\r\n\x1a\nfake", "image/png"),
        },
    )
    assert resp.status_code == 200  # register() never blocks on ml-service
    assert time.time() - t0 < 2.0, "register hung waiting on ml-service"
    job_id = resp.json()["job_id"]

    assert wait_for_status(client, job_id) == "failed"
    result = client.get(f"/api/result/{job_id}").json()
    assert result["status"] == "failed"
    assert result["error"]
    assert result["metrics"] is None


# ---------------------------------------------------------------------------
# 3. PNG only, no label fields — unchanged behaviour, labels default to None
# ---------------------------------------------------------------------------

def test_png_only_no_label_fields_unchanged(client, monkeypatch, sample_png, fake_ml_outputs):
    captured = {}

    async def spy_call(source_bytes, source_name, reference_bytes, reference_name,
                        source_label_bytes=None, source_label_name=None,
                        reference_label_bytes=None, reference_label_name=None):
        captured.update(source_label_bytes=source_label_bytes, reference_label_bytes=reference_label_bytes)
        return fake_ml_outputs

    monkeypatch.setattr(routes, "call_ml_register", spy_call)

    resp = client.post(
        "/api/register",
        files={
            "source": ("source.png", sample_png, "image/png"),
            "reference": ("reference.png", sample_png, "image/png"),
        },
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    assert wait_for_status(client, job_id) == "completed"

    assert captured["source_label_bytes"] is None
    assert captured["reference_label_bytes"] is None

    result = client.get(f"/api/result/{job_id}").json()
    assert result["status"] == "completed"
    assert result["metrics"]["rmse_px"] == pytest.approx(0.42)
