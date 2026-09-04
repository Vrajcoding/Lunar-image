# Lunar Registration Backend Service

FastAPI API Gateway and job orchestrator for the Lunar Image Registration system.

## Setup & Running

### Local Development
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Docker
```bash
docker build -t lunar-backend .
docker run -p 8000:8000 -e ML_SERVICE_URL=http://localhost:8001 lunar-backend
```

## Endpoints

- `GET /api/health` — Health check
- `POST /api/register` — Upload source & reference images (`multipart/form-data`)
- `GET /api/status/{job_id}` — Check status (`processing` | `completed` | `failed`)
- `GET /api/result/{job_id}` — Get metrics & output download URLs
- `GET /api/download/{job_id}/{file_type}` — Download `registered` image or `matches` CSV

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q          # from the backend/ directory
```

The suite stubs the ML microservice (no network / Docker needed) and covers the
`register → status → result → download` happy path plus the failure paths where
`ml-service` is down, times out, returns HTTP 500, or returns a malformed body —
in every case the job must end `failed` without the gateway hanging or 500-ing.

## Known limitations (acceptable for the hackathon demo — call these out in the pitch)

- **Job store is in-memory.** `app/jobs.py` keeps jobs in a plain process-local
  `dict`. If the backend container restarts, all job history is lost and
  `GET /api/status/{job_id}` will 404 for previously issued IDs. A running demo is
  unaffected; only a mid-demo restart hurts. The natural upgrade (noted in
  `backend.md` §6.2) is SQLite — a single-file store that survives restarts —
  but it was deliberately left out to keep the moving parts minimal.
- **No auth / rate limiting.** Any client can post jobs.
- **Output file retention.** The backend serves job artefacts straight off the
  shared `/data` volume; it never writes or deletes them. The `ml-service` now
  sweeps job directories older than `JOB_RETENTION_SEC` (default 1 h) at the
  start of each `/register`, so the volume no longer grows without bound.
