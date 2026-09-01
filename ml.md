# ML / Computer Vision Module — Lunar Image Registration
## Owner: Person 1 (ML Engineer) — Standalone, Dockerized Microservice

> **Your job in one sentence:** Take two images (source = Chandrayaan-2, reference = lunar map) and return a warped/aligned image + match points + accuracy numbers — exposed as a small FastAPI microservice so the Backend engineer can call you over HTTP. You do not touch the frontend or the public-facing API.

---

# 1. What You're Building

A self-contained **ML microservice** (`ml-service`) that:

1. Accepts two images (source + reference) via an internal HTTP API.
2. Runs the full registration pipeline: preprocess → detect → match → filter → transform → refine → warp.
3. Returns: the registered image, a match-point list, and evaluation metrics (RMSE, inlier ratio, etc.) as JSON + files.
4. Runs entirely inside its own Docker container, independent of the backend/frontend tech choices.

You will hand the Backend engineer exactly **one contract**: `POST /register` in → JSON + files out (Section 6). Everything on your side of that line is yours to design.

---

# 2. What To Download / Install (Local Dev, Before Docker)

Install these once on your machine so you can develop and test before containerizing:

| Tool | Version | Download |
|---|---|---|
| Python | 3.11.x | https://www.python.org/downloads/ |
| Git | latest | https://git-scm.com/downloads |
| Docker Desktop | latest | https://www.docker.com/products/docker-desktop/ |
| VS Code (recommended) | latest | https://code.visualstudio.com/ |
| VS Code Python extension | — | install from VS Code Extensions tab |

Check installs:
```bash
python --version      # should show 3.11.x
docker --version
git --version
```

---

# 3. Folder Structure (create exactly this)

```text
lunar-registration/
└── ml-service/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py                # FastAPI entrypoint
    │   ├── pipeline/
    │   │   ├── __init__.py
    │   │   ├── preprocess.py      # CLAHE, denoise, pyramids
    │   │   ├── features.py        # SIFT/ORB/AKAZE detection+description
    │   │   ├── matching.py        # kNN match, ratio test, grid uniformity
    │   │   ├── geometry.py        # RANSAC, homography/TPS, warp
    │   │   ├── refine.py          # sub-pixel refinement
    │   │   └── evaluate.py        # RMSE, inlier ratio, uniformity score
    │   ├── schemas.py             # Pydantic request/response models
    │   └── config.py              # constants (grid size, ratio threshold, etc.)
    ├── tests/
    │   ├── test_pipeline.py
    │   └── sample_images/         # 2-3 test image pairs (put real Chandrayaan-2 + LROC pairs here)
    ├── requirements.txt
    ├── Dockerfile
    └── README.md
```

Create it:
```bash
mkdir -p lunar-registration/ml-service/app/pipeline
mkdir -p lunar-registration/ml-service/tests/sample_images
cd lunar-registration/ml-service
```

---

# 4. Dependencies — `requirements.txt`

```text
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.9
opencv-contrib-python==4.10.0.84
numpy==1.26.4
scipy==1.13.1
scikit-image==0.24.0
rasterio==1.3.10
pydantic==2.9.2
pillow==10.4.0
```

Install locally:
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Note: `opencv-contrib-python` (not plain `opencv-python`) is required — SIFT lives in the `contrib` build.
> Stretch goal only (learned matching): add `torch`, `kornia` later once the classical path works end-to-end — these are large downloads (~2GB), don't block Day 1 on them.

---

# 5. The Pipeline — What Each File Does & Starter Code

## 5.1 `app/config.py`
```python
GRID_SIZE = 8                 # N x N grid for uniform match distribution
MAX_MATCHES_PER_CELL = 10
RATIO_TEST_THRESHOLD = 0.75   # Lowe's ratio test
RANSAC_REPROJ_THRESHOLD = 3.0 # pixels, initial pass
SIFT_N_FEATURES = 5000
PYRAMID_LEVELS = 4
```

## 5.2 `app/pipeline/preprocess.py`
```python
import cv2
import numpy as np

def load_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img

def normalize_illumination(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(img, h=7)

def build_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(pyramid[-1])
        pyramid.append(img)
    return pyramid

def preprocess(path: str, levels: int) -> list[np.ndarray]:
    img = load_grayscale(path)
    img = normalize_illumination(img)
    img = denoise(img)
    return build_pyramid(img, levels)
```

## 5.3 `app/pipeline/features.py`
```python
import cv2
import numpy as np

def detect_and_describe(img: np.ndarray, n_features: int):
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def detect_orb_fallback(img: np.ndarray, n_features: int):
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors
```

## 5.4 `app/pipeline/matching.py`
```python
import cv2
import numpy as np
from collections import defaultdict

def match_descriptors(desc_a, desc_b, ratio_thresh: float):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(desc_a, desc_b, k=2)
    good = []
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good

def enforce_uniform_distribution(matches, kp_a, img_shape, grid_size: int, max_per_cell: int):
    h, w = img_shape
    cell_h, cell_w = h / grid_size, w / grid_size
    cells = defaultdict(list)
    for m in matches:
        x, y = kp_a[m.queryIdx].pt
        cell_id = (int(y // cell_h), int(x // cell_w))
        cells[cell_id].append(m)

    filtered = []
    for cell_matches in cells.values():
        cell_matches.sort(key=lambda m: m.distance)
        filtered.extend(cell_matches[:max_per_cell])

    covered_cells = len(cells)
    total_cells = grid_size * grid_size
    uniformity_score = covered_cells / total_cells
    return filtered, uniformity_score
```

## 5.5 `app/pipeline/geometry.py`
```python
import cv2
import numpy as np

def estimate_homography(src_pts, dst_pts, reproj_thresh: float):
    H, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, reproj_thresh, maxIters=5000, confidence=0.995
    )
    return H, mask

def warp_image(img: np.ndarray, H: np.ndarray, out_shape):
    h, w = out_shape
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC)
```

## 5.6 `app/pipeline/refine.py`
```python
import cv2
import numpy as np

def subpixel_refine_points(img: np.ndarray, points: np.ndarray, win_size=(5, 5)):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    refined = cv2.cornerSubPix(img, pts, win_size, (-1, -1), criteria)
    return refined.reshape(-1, 2)
```

## 5.7 `app/pipeline/evaluate.py`
```python
import numpy as np

def compute_rmse(src_pts, dst_pts, H) -> float:
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    projected = (H @ src_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected - dst_pts, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))

def build_metrics(inlier_count, total_matches, rmse, uniformity_score, runtime_sec):
    return {
        "rmse_px": round(rmse, 4),
        "inlier_count": int(inlier_count),
        "inlier_ratio": round(inlier_count / max(total_matches, 1), 4),
        "uniformity_score": round(uniformity_score, 4),
        "runtime_sec": round(runtime_sec, 3),
    }
```

---

# 6. The API Contract (this is what Backend calls — do not change without telling Backend)

## `app/schemas.py`
```python
from pydantic import BaseModel

class RegistrationMetrics(BaseModel):
    rmse_px: float
    inlier_count: int
    inlier_ratio: float
    uniformity_score: float
    runtime_sec: float

class RegistrationResponse(BaseModel):
    status: str
    registered_image_path: str
    match_points_path: str
    metrics: RegistrationMetrics
```

## `app/main.py`
```python
import time, os, uuid, csv
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from app.pipeline.preprocess import preprocess
from app.pipeline.features import detect_and_describe
from app.pipeline.matching import match_descriptors, enforce_uniform_distribution
from app.pipeline.geometry import estimate_homography, warp_image
from app.pipeline.evaluate import compute_rmse, build_metrics
from app.config import RATIO_TEST_THRESHOLD, GRID_SIZE, MAX_MATCHES_PER_CELL, \
    RANSAC_REPROJ_THRESHOLD, SIFT_N_FEATURES, PYRAMID_LEVELS

app = FastAPI(title="Lunar Registration ML Service")
STORAGE_DIR = "/data"  # shared Docker volume, see docker-compose.yml
os.makedirs(STORAGE_DIR, exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/register")
async def register(source: UploadFile = File(...), reference: UploadFile = File(...)):
    t0 = time.time()
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(STORAGE_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    src_path = os.path.join(job_dir, "source.png")
    ref_path = os.path.join(job_dir, "reference.png")
    with open(src_path, "wb") as f:
        f.write(await source.read())
    with open(ref_path, "wb") as f:
        f.write(await reference.read())

    # Preprocess (base resolution only for baseline; pyramid list available for multi-scale)
    src_pyr = preprocess(src_path, PYRAMID_LEVELS)
    ref_pyr = preprocess(ref_path, PYRAMID_LEVELS)
    src_img, ref_img = src_pyr[0], ref_pyr[0]

    # Detect + describe
    kp_a, desc_a = detect_and_describe(src_img, SIFT_N_FEATURES)
    kp_b, desc_b = detect_and_describe(ref_img, SIFT_N_FEATURES)

    # Match + uniform distribution
    good_matches = match_descriptors(desc_a, desc_b, RATIO_TEST_THRESHOLD)
    good_matches, uniformity_score = enforce_uniform_distribution(
        good_matches, kp_a, src_img.shape, GRID_SIZE, MAX_MATCHES_PER_CELL
    )

    src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Outlier rejection + transform
    H, mask = estimate_homography(src_pts, dst_pts, RANSAC_REPROJ_THRESHOLD)
    inlier_mask = mask.ravel().astype(bool)
    inlier_src, inlier_dst = src_pts[inlier_mask], dst_pts[inlier_mask]

    # Warp
    registered = warp_image(src_img, H, ref_img.shape)
    out_img_path = os.path.join(job_dir, "registered.png")
    cv2.imwrite(out_img_path, registered)

    # Match points file
    match_csv_path = os.path.join(job_dir, "matches.csv")
    with open(match_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src_x", "src_y", "ref_x", "ref_y", "is_inlier"])
        for i, m in enumerate(good_matches):
            writer.writerow([*src_pts[i], *dst_pts[i], bool(inlier_mask[i])])

    # Evaluate
    rmse = compute_rmse(inlier_src, inlier_dst, H)
    runtime = time.time() - t0
    metrics = build_metrics(len(inlier_src), len(good_matches), rmse, uniformity_score, runtime)

    return JSONResponse({
        "status": "completed",
        "job_id": job_id,
        "registered_image_path": out_img_path,
        "match_points_path": match_csv_path,
        "metrics": metrics,
    })
```

Run locally to test before Docker:
```bash
uvicorn app.main:app --reload --port 8001
```
Test it:
```bash
curl -X POST http://localhost:8001/register \
  -F "source=@tests/sample_images/source1.png" \
  -F "reference=@tests/sample_images/ref1.png"
```

---

# 7. Dockerfile (`ml-service/Dockerfile`)

```dockerfile
FROM python:3.11-slim

# System deps for OpenCV + rasterio/GDAL
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p /data
EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

Build & run standalone (before full docker-compose is wired up by Backend):
```bash
docker build -t ml-service .
docker run -p 8001:8001 -v $(pwd)/data:/data ml-service
```

---

# 8. Your Section of `docker-compose.yml` (Backend owns the full file — this is your service block)

```yaml
  ml-service:
    build: ./ml-service
    ports:
      - "8001:8001"
    volumes:
      - shared-data:/data
    networks:
      - lunar-net
```

Tell the Backend engineer: **"call me at `http://ml-service:8001/register` from inside Docker"** (service name = hostname on the Docker network — not `localhost`).

---

# 9. Test Data — Where To Get It

- Chandrayaan-2 TMC-2 / OHRC browse images: ISRO **PRADAN** portal (https://pradan.issdc.gov.in/) — register, download 2–3 browse/reduced-data image tiles of the same region under different passes.
- Reference lunar imagery: **LROC WAC/NAC mosaics** — https://quickmap.lroc.asu.edu/ or https://wms.lroc.asu.edu/lroc — export a matching region as PNG/TIFF.
- Put 2–3 real pairs + 1 synthetic pair (apply a known rotation/scale/shift to any single image using `cv2.warpAffine` to create a source with a known ground-truth transform, for exact RMSE validation) in `tests/sample_images/`.

---

# 10. Your Day-by-Day Checklist

**Day 1**
- [ ] Set up venv, install requirements, confirm `cv2.SIFT_create()` works (contrib build)
- [ ] Implement `preprocess.py`, `features.py`, basic `matching.py` (no grid distribution yet)
- [ ] Implement `geometry.py` (RANSAC + homography) and `main.py` `/register` endpoint
- [ ] Run end-to-end on one real image pair, even roughly

**Day 2**
- [ ] Add grid-based uniform match distribution
- [ ] Add sub-pixel refinement (`refine.py`) integrated into the pipeline
- [ ] Add `evaluate.py` metrics (RMSE, inlier ratio, uniformity score)
- [ ] Write Dockerfile, build image, confirm `curl` test works against the container
- [ ] Hand Backend the API contract (Section 6) — confirm they can call you

**Day 3**
- [ ] Test on a hard pair (strong shadow difference or big scale gap) — fix failures
- [ ] Build synthetic ground-truth test to prove sub-pixel RMSE claim
- [ ] (Stretch) integrate pretrained LoFTR/LightGlue as `/register?method=learned` alt path
- [ ] Add `/health` monitoring, clean logs, finalize `README.md`

**Final Day**
- [ ] Freeze pipeline, run full test set, record final metrics table
- [ ] Confirm your container runs cleanly via `docker-compose up ml-service`
- [ ] Prepare a 2-minute explanation of your pipeline for the demo

---

# 11. Acceptance Criteria (you're not done until all of these pass)

- [ ] `POST /register` works on any two arbitrary image sizes without crashing
- [ ] No image-specific hard-coded parameters
- [ ] RMSE reported on inlier points is sub-pixel (< 1.0) on at least 2 of 3 test pairs
- [ ] Match distribution uniformity score > 0.7 (covers most grid cells)
- [ ] Registered image, match CSV, and metrics JSON are all written to `/data/{job_id}/`
- [ ] Runs correctly inside Docker via `docker run` and via `docker-compose up`
- [ ] `/health` returns 200 OK

---

# 12. Common Pitfalls

- `opencv-python` instead of `opencv-contrib-python` → `cv2.SIFT_create()` fails. Always use contrib.
- Forgetting `-v $(pwd)/data:/data` when testing Docker locally → outputs vanish when container stops.
- Using `localhost` instead of the Docker service name (`ml-service`) when Backend calls you inside Docker Compose.
- Reporting RMSE on the *same* points used to fit the homography (inflates accuracy) — always fine on inliers, but be honest that a truly independent check-point set is a stretch-goal improvement.
