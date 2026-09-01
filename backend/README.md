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
