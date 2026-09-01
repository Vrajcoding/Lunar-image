# Lunar Image Co-Registration Platform

An end-to-end automated sub-pixel image registration solution for Chandrayaan-2 TMC-2 / OHRC orbital imagery and lunar basemaps. Built for SIH Hackathon.

## Architecture Overview

```text
[ React UI (Port 3000) ] ── (HTTP REST) ──> [ FastAPI Gateway (Port 8000) ]
                                                    │
                                            (Docker Network)
                                                    ▼
                                         [ ML Service (Port 8001) ]
```

- **`frontend/`**: React + Vite + Nginx UI with dark space theme, interactive blend slider, match points canvas, and performance metrics panel.
- **`backend/`**: FastAPI API gateway managing job queues, REST endpoints, and async microservice client.
- **`ml-service/`**: Microservice executing CLAHE preprocessing, SIFT feature extraction, RANSAC homography, grid uniformity filtering, and sub-pixel alignment.

---

## 🚀 How to Run with Docker

Run the single command below from the project root directory:

```bash
docker compose up --build
```

Access the services in your browser:
- **Frontend App**: [http://localhost:3000](http://localhost:3000)
- **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ML Microservice Health**: [http://localhost:8001/health](http://localhost:8001/health)

To stop all services:
```bash
docker compose down
```

---

## 💻 Local Development (Without Docker)

### 1. Start ML Microservice
```bash
cd ml-service
python -m venv venv
# Windows: venv\Scripts\activate | Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8001 --reload
```

### 2. Start Backend Service
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate | Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000 --reload
```

### 3. Start Frontend UI
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.
