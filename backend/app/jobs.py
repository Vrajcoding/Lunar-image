from threading import Lock
from typing import Optional, Dict, Any

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = Lock()

def create_job(job_id: str):
    with _lock:
        _jobs[job_id] = {"status": "processing"}

def update_job(job_id: str, data: Dict[str, Any]):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(data)

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            return job.copy()
        return None
