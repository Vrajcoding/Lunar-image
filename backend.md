# Backend Module — Lunar Image Registration
## Owner: Person 2 (Backend Engineer) — Dockerized API Gateway + Orchestrator

> **Your job in one sentence:** Give the Frontend a simple, stable REST API to upload two images and get results back — while internally calling the ML engineer's microservice to do the actual work. You also own the root `docker-compose.yml` that wires all three services (frontend, backend, ml-service) together.

---

# 1. What You're Building

A **backend API service** (`backend`) that:

1. Accepts image uploads from the Frontend.
2. Forwards them to the ML microservice (`ml-service`) over the internal Docker network.
3. Tracks job status (so the frontend can show a progress spinner).
4. Returns registered image URLs, match-point file URLs, and metrics to the Frontend.
5. Serves the output files for download.
6. Owns the **root `docker-compose.yml`** that starts frontend + backend + ml-service together with one command.

You sit exactly in the middle: **Frontend ↔ Backend ↔ ML-service**. You do not implement any computer vision — you call the ML engineer's `/register` endpoint and reshape the response for the Frontend.

---

# 2. What To Download / Install

| Tool | Version | Download |
|---|---|---|
| Python | 3.11.x | https://www.python.org/downloads/ |
| Git | latest | https://git-scm.com/downloads |
| Docker Desktop | latest | https://www.docker.com/products/docker-desktop/ |
| Postman (optional, for API testing) | latest | https://www.postman.com/downloads/ |
| VS Code | latest | https://code.visualstudio.com/ |

Check installs:
```bash
python --version
docker --version
docker compose version
```

---

# 3. Folder Structure (create exactly this, at the project root — this is the root repo layout everyone shares)

```text
lunar-registration/
├── docker-compose.yml          # YOU own this — the single file that starts everything
├── .env                        # shared env vars (ports, service URLs)
├── ml-service/                 # ML engineer's folder (already exists — don't touch internals)
├── backend/                    # YOUR folder
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI entrypoint
│   │   ├── routes.py           # /register, /status, /result, /download endpoints
│   │   ├── ml_client.py        # HTTP client that calls ml-service
│   │   ├── jobs.py             # in-memory / SQLite job store
│   │   └── schemas.py          # request/response models
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
└── frontend/                   # Frontend engineer's folder (already exists — don't touch internals)
```

Create your part:
```bash
mkdir -p lunar-registration/backend/app
cd lunar-registration/backend
```

---

# 4. Dependencies — `backend/requirements.txt`

```text
fastapi==0.115.0
uvicorn[standard]==0.30.6
httpx==0.27.2
python-multipart==0.0.9
pydantic==2.9.2
aiofiles==24.1.0
```

Install locally:
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

# 5. The API Contract You Expose to the Frontend

This is what Frontend engineer will call. **Do not change these paths/response shapes without telling Frontend.**

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/register` | Upload source + reference image, start registration |
| `GET` | `/api/status/{job_id}` | Poll job status (`processing` / `completed` / `failed`) |
| `GET` | `/api/result/{job_id}` | Get metrics + file URLs once completed |
| `GET` | `/api/download/{job_id}/{file_type}` | Download `registered`, `matches`, or `report` file |
| `GET` | `/api/health` | Health check |

### Example: `POST /api/register` response
```json
{
  "job_id": "abc123",
  "status": "processing"
}
```

### Example: `GET /api/result/{job_id}` response
```json
{
  "job_id": "abc123",
  "status": "completed",
  "registered_image_url": "/api/download/abc123/registered",
  "match_points_url": "/api/download/abc123/matches",
  "metrics": {
    "rmse_px": 0.42,
    "inlier_count": 187,
    "inlier_ratio": 0.81,
    "uniformity_score": 0.93,
    "runtime_sec": 4.7
  }
}
```

---

# 6. Code — What Each File Does & Starter Code

## 6.1 `app/schemas.py`
```python
from pydantic import BaseModel
from typing import Optional

class RegisterAccepted(BaseModel):
    job_id: str
    status: str

class Metrics(BaseModel):
    rmse_px: float
    inlier_count: int
    inlier_ratio: float
    uniformity_score: float
    runtime_sec: float

class ResultResponse(BaseModel):
    job_id: str
    status: str
    registered_image_url: Optional[str] = None
    match_points_url: Optional[str] = None
    metrics: Optional[Metrics] = None
    error: Optional[str] = None
```

## 6.2 `app/jobs.py` — simple in-memory job store (fine for hackathon; swap for SQLite/Redis later)
```python
from threading import Lock

_jobs: dict[str, dict] = {}
_lock = Lock()

def create_job(job_id: str):
    with _lock:
        _jobs[job_id] = {"status": "processing"}

def update_job(job_id: str, data: dict):
    with _lock:
        _jobs[job_id].update(data)

def get_job(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)
```

## 6.3 `app/ml_client.py` — calls the ML microservice
```python
import httpx
import os

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8001")

async def call_ml_register(source_bytes: bytes, source_name: str,
                            reference_bytes: bytes, reference_name: str) -> dict:
    files = {
        "source": (source_name, source_bytes, "image/png"),
        "reference": (reference_name, reference_bytes, "image/png"),
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{ML_SERVICE_URL}/register", files=files)
        resp.raise_for_status()
        return resp.json()
```

## 6.4 `app/routes.py`
```python
import uuid
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.jobs import create_job, update_job, get_job
from app.ml_client import call_ml_register
from app.schemas import RegisterAccepted, ResultResponse, Metrics

router = APIRouter(prefix="/api")

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/register", response_model=RegisterAccepted)
async def register(source: UploadFile = File(...), reference: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    create_job(job_id)

    source_bytes = await source.read()
    reference_bytes = await reference.read()

    asyncio.create_task(
        process_job(job_id, source_bytes, source.filename, reference_bytes, reference.filename)
    )
    return RegisterAccepted(job_id=job_id, status="processing")

async def process_job(job_id, source_bytes, source_name, reference_bytes, reference_name):
    try:
        result = await call_ml_register(source_bytes, source_name, reference_bytes, reference_name)
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
    if job["status"] != "completed":
        return ResultResponse(job_id=job_id, status=job["status"], error=job.get("error"))
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
    if not job or job["status"] != "completed":
        raise HTTPException(404, "Result not ready")
    path_map = {
        "registered": job["registered_image_path"],
        "matches": job["match_points_path"],
    }
    if file_type not in path_map:
        raise HTTPException(400, "Invalid file_type")
    return FileResponse(path_map[file_type])
```

## 6.5 `app/main.py`
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router

app = FastAPI(title="Lunar Registration Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production; fine for hackathon demo
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
```

Run locally to test (needs ml-service running too, either locally on 8001 or via Docker):
```bash
uvicorn app.main:app --reload --port 8000
```

Test with curl:
```bash
curl -X POST http://localhost:8000/api/register \
  -F "source=@../ml-service/tests/sample_images/source1.png" \
  -F "reference=@../ml-service/tests/sample_images/ref1.png"

curl http://localhost:8000/api/status/<job_id>
curl http://localhost:8000/api/result/<job_id>
```

---

# 7. Dockerfile (`backend/Dockerfile`)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

# 8. The Root `docker-compose.yml` (YOU own and maintain this — place at `lunar-registration/docker-compose.yml`)

```yaml
version: "3.9"

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    networks:
      - lunar-net

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - ML_SERVICE_URL=http://ml-service:8001
    depends_on:
      - ml-service
    volumes:
      - shared-data:/data
    networks:
      - lunar-net

  ml-service:
    build: ./ml-service
    ports:
      - "8001:8001"
    volumes:
      - shared-data:/data
    networks:
      - lunar-net

volumes:
  shared-data:

networks:
  lunar-net:
    driver: bridge
```

### One command starts the entire project:
```bash
docker compose up --build
```

- Frontend → http://localhost:3000
- Backend API → http://localhost:8000/api
- ML service (internal, for debugging) → http://localhost:8001

Stop everything:
```bash
docker compose down
```

> **Important networking note for the team:** inside Docker, services talk to each other by **service name**, not `localhost`. That's why `ML_SERVICE_URL=http://ml-service:8001` and not `http://localhost:8001`. `localhost` only works for calls from your own host machine's browser/curl.

---

# 9. `.env` (root level, shared config)

```env
BACKEND_PORT=8000
ML_SERVICE_PORT=8001
FRONTEND_PORT=3000
```

(Wire these into `docker-compose.yml` with `${BACKEND_PORT}` syntax if you want configurable ports — optional polish for later.)

---

# 10. Your Day-by-Day Checklist

**Day 1**
- [ ] Set up `backend/` folder structure and venv
- [ ] Implement `schemas.py`, `jobs.py`, `ml_client.py`, `routes.py`, `main.py`
- [ ] Run backend locally (`uvicorn`), confirm `/api/health` returns 200
- [ ] Coordinate with ML engineer: confirm their `/register` contract matches `ml_client.py` exactly
- [ ] Test `/api/register` against ML engineer's locally-running service (before Docker)

**Day 2**
- [ ] Write `backend/Dockerfile`
- [ ] Write the root `docker-compose.yml` (frontend service can be a placeholder/stub container until Frontend engineer's folder is ready)
- [ ] Run `docker compose up --build` — confirm backend ↔ ml-service communicate correctly inside Docker
- [ ] Add error handling: what happens if ml-service is down, times out, or returns malformed data
- [ ] Hand Frontend engineer the API contract (Section 5) — confirm they can call you

**Day 3**
- [ ] Add proper job status states (`processing`, `completed`, `failed`) end-to-end
- [ ] Test with real large image files (make sure upload size limits aren't hit — see Section 12)
- [ ] Add basic request validation (reject non-image files, missing fields)
- [ ] Full three-service integration test: upload from Frontend → see result appear correctly

**Final Day**
- [ ] Freeze API, confirm `docker compose up --build` works from a clean clone (test on a teammate's machine)
- [ ] Write `backend/README.md` with run instructions
- [ ] Prepare a 2-minute explanation of the backend's role for the demo

---

# 11. Acceptance Criteria

- [ ] `docker compose up --build` starts all three services with zero manual steps
- [ ] `POST /api/register` accepts two image files and returns a `job_id` immediately (does not block/hang)
- [ ] `GET /api/status/{job_id}` correctly reflects `processing` → `completed`/`failed`
- [ ] `GET /api/result/{job_id}` returns correct metrics and working download URLs once completed
- [ ] Downloaded files (`registered`, `matches`) actually open and are correct
- [ ] Backend survives ml-service being briefly slow (no crash, job just stays `processing`)
- [ ] CORS is configured so the Frontend (different port) can call the API without browser errors

---

# 12. Common Pitfalls

- Using `localhost` instead of `ml-service` as the hostname when calling from inside Docker — this is the #1 integration bug on these projects.
- Not setting an `httpx` timeout — ML processing can take several seconds; a low default timeout will cause false failures.
- FastAPI's default upload size limits: for large lunar images, make sure the ASGI server (`uvicorn`) and any reverse proxy don't silently truncate uploads. Test with a real multi-MB image, not just a small sample.
- Forgetting `CORSMiddleware` — the Frontend running on port 3000 calling the Backend on port 8000 will be blocked by the browser without it.
- Blocking the `/register` endpoint on the full ML call — always kick it off as a background task (`asyncio.create_task`) so the API responds instantly with a `job_id`, otherwise the Frontend's upload button will look frozen.
