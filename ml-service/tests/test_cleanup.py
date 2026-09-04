"""
test_cleanup.py — storage retention sweep for the ML service.

The pipeline writes one directory of artefacts per job into STORAGE_DIR and
nothing else removes them. `sweep_old_jobs()` runs opportunistically at the
start of every /register call and deletes job directories older than
JOB_RETENTION_SEC.
"""
import os
import time

import pytest

import app.main as ml_main


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_main, "STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(ml_main, "JOB_RETENTION_SEC", 3600)
    return tmp_path


def _make_job_dir(root, name, age_sec):
    d = root / name
    d.mkdir()
    (d / "registered.png").write_bytes(b"x")
    (d / "matches.csv").write_text("src_x,src_y,ref_x,ref_y,is_inlier\n")
    old = time.time() - age_sec
    os.utime(d, (old, old))
    return d


def test_sweep_removes_only_stale_dirs(storage):
    fresh = _make_job_dir(storage, "fresh-job", age_sec=60)          # 1 min old
    stale = _make_job_dir(storage, "stale-job", age_sec=7200)        # 2 h old

    removed = ml_main.sweep_old_jobs()

    assert removed == 1
    assert fresh.exists()
    assert not stale.exists()


def test_sweep_disabled_when_retention_zero(storage, monkeypatch):
    monkeypatch.setattr(ml_main, "JOB_RETENTION_SEC", 0)
    stale = _make_job_dir(storage, "stale-job", age_sec=99999)

    assert ml_main.sweep_old_jobs() == 0
    assert stale.exists()


def test_sweep_ignores_loose_files_and_missing_dir(storage):
    (storage / "not-a-job.txt").write_text("hello")
    removed = ml_main.sweep_old_jobs()
    assert removed == 0
    assert (storage / "not-a-job.txt").exists()


def test_register_triggers_sweep(client, sample_images, tmp_path, monkeypatch):
    """A real /register call cleans up a pre-existing stale job dir."""
    monkeypatch.setattr(ml_main, "STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(ml_main, "JOB_RETENTION_SEC", 3600)
    stale = _make_job_dir(tmp_path, "old-job", age_sec=7200)

    with open(sample_images["source1"], "rb") as s, open(sample_images["ref1"], "rb") as r:
        resp = client.post(
            "/register",
            files={
                "source": ("source.png", s, "image/png"),
                "reference": ("reference.png", r, "image/png"),
            },
        )
    assert resp.status_code == 200
    assert not stale.exists(), "stale job dir should have been swept by /register"
