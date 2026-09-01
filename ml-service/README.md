# Lunar Registration ML Service

A self-contained FastAPI microservice that aligns a **source** image (Chandrayaan‑2)
to a **reference** image (a lunar map / LROC mosaic) and returns the warped image,
the match‑point list, and accuracy metrics.

It is stateless over HTTP and owns exactly one contract: `POST /register`. The
backend calls it at `http://ml-service:8001` on the Docker network; nothing on the
ML side of that line affects the frontend or backend tech choices.

---

## Pipeline

```
preprocess → detect+describe → match → uniform-grid filter → RANSAC (pass 1)
          → sub-pixel refine → RANSAC (pass 2, tight) → warp → evaluate
```

| Stage | File | What it does |
|---|---|---|
| Preprocess | `app/pipeline/preprocess.py` | grayscale load, CLAHE illumination normalisation, non‑local‑means denoise, Gaussian pyramid |
| Features | `app/pipeline/features.py` | SIFT keypoints + 128‑D descriptors (`opencv-contrib`), ORB fallback |
| Matching | `app/pipeline/matching.py` | BF kNN match, Lowe ratio test (0.75), N×N grid cap for spatial uniformity |
| Geometry | `app/pipeline/geometry.py` | `findHomography` with RANSAC, `warpPerspective` (cubic) |
| Refine | `app/pipeline/refine.py` | `cornerSubPix` sub‑pixel localisation of inlier points |
| Evaluate | `app/pipeline/evaluate.py` | RMSE on inliers, inlier ratio, uniformity score, runtime |

**Two‑pass RANSAC.** Pass 1 rejects gross outliers at a 3.0 px reprojection
threshold. The surviving inliers are refined to sub‑pixel accuracy, then pass 2
re‑fits the homography at a tight 1.5 px threshold. The refined fit is only
adopted if it retains ≥ 4 inliers, otherwise the pass‑1 result stands. This is
what drives RMSE below 1.0 px on the synthetic pairs.

Tunable constants live in `app/config.py`:

| Constant | Default | Meaning |
|---|---|---|
| `GRID_SIZE` | 8 | N×N grid for match distribution |
| `MAX_MATCHES_PER_CELL` | 10 | cap per grid cell |
| `RATIO_TEST_THRESHOLD` | 0.75 | Lowe ratio test |
| `RANSAC_REPROJ_THRESHOLD` | 3.0 | pixels, coarse pass |
| `SIFT_N_FEATURES` | 5000 | SIFT feature budget per image |
| `PYRAMID_LEVELS` | 4 | pyramid depth (base level is used for matching) |

---

## API

### `GET /health`

```json
{ "status": "ok", "service": "lunar-registration-ml" }
```

### `POST /register`

`multipart/form-data` with two file fields:

| Field | Required | Notes |
|---|---|---|
| `source` | yes | image to be warped (Chandrayaan‑2) |
| `reference` | yes | target frame (lunar map) |

Any decodable image format / size is accepted. Output is written to
`${STORAGE_DIR}/{job_id}/` (default `STORAGE_DIR=/data`, the shared Docker volume).

**200 response:**

```json
{
  "status": "completed",
  "job_id": "3f2b1c9e-...",
  "registered_image_path": "/data/3f2b1c9e-.../registered.png",
  "match_points_path": "/data/3f2b1c9e-.../matches.csv",
  "metrics": {
    "rmse_px": 0.42,
    "inlier_count": 213,
    "inlier_ratio": 0.87,
    "uniformity_score": 0.94,
    "runtime_sec": 1.83
  }
}
```

Files written per job: `source.png`, `reference.png`, `registered.png`,
`matches.csv` (`src_x,src_y,ref_x,ref_y,is_inlier`).

**Errors** (the server returns 4xx, never 500, on bad input):

| Code | Cause |
|---|---|
| 400 | one or both uploads are empty |
| 422 | an upload could not be decoded as an image |

If fewer than 4 matches are found the service does not crash: it returns
`status: "completed"` with an identity transform and empty match set.

---

## Run it

### Local dev (no Docker)

```bash
cd ml-service
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate on *nix
pip install -r requirements.txt

# STORAGE_DIR falls back to ./data automatically when /data is not writable
uvicorn app.main:app --reload --port 8001
```

```bash
curl http://localhost:8001/health
curl -X POST http://localhost:8001/register \
  -F "source=@tests/sample_images/source1.png" \
  -F "reference=@tests/sample_images/ref1.png"
```

### Docker (standalone)

```bash
cd ml-service
docker build -t ml-service .

# Windows CMD:
docker run --rm -p 8001:8001 -v "%cd%\data:/data" ml-service
# PowerShell:
docker run --rm -p 8001:8001 -v "${PWD}\data:/data" ml-service
# bash:
docker run --rm -p 8001:8001 -v "$(pwd)/data:/data" ml-service

curl http://localhost:8001/health
```

Mount a volume at `/data` or the job outputs vanish when the container stops.

### Full stack (from repo root)

```bash
cd ..
docker-compose up --build
```

| Service | URL |
|---|---|
| frontend | http://localhost:3000 |
| backend | http://localhost:8000/api/health |
| ml-service | http://localhost:8001/health |

End‑to‑end through the backend:

```bash
curl -X POST http://localhost:8000/api/register \
  -F "source=@ml-service/tests/sample_images/source1.png" \
  -F "reference=@ml-service/tests/sample_images/ref1.png"
```

The backend runs registration as a background job, so this returns
`{ "job_id": "...", "status": "processing" }`. Poll for the result:

```bash
curl http://localhost:8000/api/status/<job_id>
curl http://localhost:8000/api/result/<job_id>
```

Because backend and ml-service share the `shared-data` volume at `/data`, the
paths ml-service reports are directly readable by the backend for download.

---

## Tests

```bash
cd ml-service
venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v
```

44 tests: health, register (easy / medium / identical pairs), error handling
(empty upload, corrupt file, mismatched sizes), the ml.md Section 11 acceptance
criteria (sub‑pixel RMSE < 1.0 px, uniformity > 0.7, output files written), and
unit tests for every pipeline module. Sample images are generated on first run by
`tests/conftest.py` (see also `tests/create_sample_images.py`), so the suite is
self‑contained.

---

## Dependencies & the Docker build

`requirements.txt` uses `>=` pins so the same file resolves on Python 3.11 (the
`python:3.11-slim` image) and on a newer local interpreter. The Docker image
needs only `libgl1` + `libglib2.0-0` for OpenCV — the headless wheel still
`dlopen`s them at import time.

`rasterio`, `scipy`, and `scikit-image` are **commented out**: no pipeline module
imports them, and `rasterio` drags in a GDAL toolchain that is the usual cause of
a failed image build. If you add GeoTIFF I/O later, re‑enable them in
`requirements.txt` and restore `gdal-bin libgdal-dev gcc g++` in the `Dockerfile`
(noted inline in both files).
