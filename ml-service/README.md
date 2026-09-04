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
| Load | `app/pipeline/loader.py` | format-aware ingestion (PNG/JPEG/BMP, GeoTIFF, PDS4, PDS3) → uint8 grayscale + metadata |
| Preprocess | `app/pipeline/preprocess.py` | CLAHE illumination normalisation, non‑local‑means denoise, Gaussian pyramid |
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

## Supported input formats

`app/pipeline/loader.py` dispatches on file extension. All branches converge on
the same thing downstream gets today: a uint8, single-channel 2D array.

| Format | Extension(s) | Notes |
|---|---|---|
| PNG / JPEG / BMP | `.png`, `.jpg`, `.jpeg`, `.bmp` | `cv2.imread`, unchanged from before — byte-identical output |
| TIFF / GeoTIFF | `.tif`, `.tiff` | via `rasterio`; `crs`/`transform` captured into metadata when the TIFF is georeferenced |
| PDS4 (Chandrayaan-2) | `.xml` (detached label) | via `rasterio`; open the **`.xml`**, not the `.img` — the label is what tells GDAL how to interpret the binary |
| PDS4 raw data | `.img` | has no self-describing header — automatically redirected to the sibling `.xml` label in the same directory (this is the natural mistake: a user has the `.img` in hand and points at that) |
| PDS3 (pre-Chandrayaan-2 era) | `.lbl` | via `rasterio`; same idea as PDS4, kept minimal since the target is Chandrayaan-2 |

**Bit depth.** 8-bit input passes through unchanged (`scaling_applied: "none"`).
16-/32-bit input is never truncated by a bit-shift — it goes through a 2–98
percentile linear stretch to uint8, because a full-range map wastes most of
the 8-bit space on values that never occur in a real lunar scene (shadowed
crater floors and sunlit rims sit at opposite histogram extremes). The bounds
actually used are recorded in `metadata["stretch_bounds"]`.

**Both images of a pair are stretched with the same bounds.** `POST /register`
loads the reference first, reads back its stretch bounds, and reuses them for
the source. Stretching each image independently would introduce an artificial
intensity difference that the matcher would read as a real radiometric
difference between the two scenes — silently, and hard to debug after the
fact.

**Multi-band input.** Only TMC-2/OHRC (both single-band panchromatic) are the
intended input, so this path exists to fail loudly rather than silently if
something else (e.g. an IIRS cube) is fed in. Band 1 is used unless `band` is
given; whenever a file has more than one band and `band` is not specified, a
WARNING is logged naming the file and its band count. Same for a PDS4 label
that exposes multiple `Array` objects as GDAL subdatasets rather than bands —
the first subdataset is used and a warning names which one.

**`.img` → `.xml` redirect.** A raw PDS4 `.img` carries no header of its own;
the detached `.xml` label next to it says how to interpret the bytes. Pointing
the loader at the `.img` is redirected to the sibling `.xml` automatically (an
INFO log line records this). If no `.xml` (or PDS3 `.lbl`) sibling exists, the
error names both paths it looked for rather than failing opaquely.

`POST /register` accepts an optional `band` form field (1-indexed) to force a
specific band instead of the multi-band default.

---

## API

### `GET /health`

```json
{ "status": "ok", "service": "lunar-registration-ml" }
```

### `POST /register`

`multipart/form-data`:

| Field | Required | Notes |
|---|---|---|
| `source` | yes | image to be warped (Chandrayaan‑2) |
| `reference` | yes | target frame (lunar map) |
| `band` | no | 1-indexed band to use for multi-band input; defaults to band 1 |

Any format in [Supported input formats](#supported-input-formats) is accepted,
at any size. The upload's own filename extension decides which loader branch
handles it — `.tif`/`.xml`/etc., not just `.png` — so keep the real extension
on the uploaded file. Output is written to `${STORAGE_DIR}/{job_id}/` (default
`STORAGE_DIR=/data`, the shared Docker volume).

**Retention.** At the start of every `/register` call the service sweeps job
directories older than `JOB_RETENTION_SEC` (env var, default `3600` = 1 hour) so
the shared volume does not grow without bound. Set `JOB_RETENTION_SEC=0` to keep
everything.

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
  },
  "input_metadata": {
    "source": { "source_format": "png", "original_dtype": "uint8", "band_count": 1, "band_used": 1, "scaling_applied": "none", "stretch_bounds": null, "product_id": null },
    "reference": { "source_format": "png", "original_dtype": "uint8", "band_count": 1, "band_used": 1, "scaling_applied": "none", "stretch_bounds": null, "product_id": null }
  }
}
```

`input_metadata` mirrors what `loader.load_image()` returned for each input —
see [Supported input formats](#supported-input-formats) for the full key list
(`crs`, `transform`, `subdataset_used`, `original_shape`, `product_id` parsed
from a `ch2_*` filename, etc.).

Files written per job: `source.<ext>`, `reference.<ext>` (the upload's own
extension), `registered.png`, `matches.csv`
(`src_x,src_y,ref_x,ref_y,is_inlier`).

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

Note: `input_metadata` (see above) is currently only in ml-service's own
`/register` response. Surfacing it through the backend's `GET /api/result/{job_id}`
would be a `backend/` change, out of scope here — see "ingestion" in `ml.md`
if that's picked up later.

---

## Tests

```bash
cd ml-service
venv\Scripts\python.exe -m pytest tests/ -v
```

59 tests total:

- `tests/test_pipeline.py` (44): health, register (easy / medium / identical
  pairs), error handling (empty upload, corrupt file, mismatched sizes), the
  ml.md Section 11 acceptance criteria (sub‑pixel RMSE < 1.0 px, uniformity >
  0.7, output files written), and unit tests for every pipeline module. This
  count is a regression guard — it must stay 44; the algorithm itself was not
  touched to add format support.
- `tests/test_cleanup.py` (4): storage retention sweep.
- `tests/test_loader.py` (11): the format-aware loader — PNG regression
  (byte-identical to the old `cv2.imread` path), 16-bit stretch, multi-band
  default + explicit selection + warning, shared stretch bounds across a
  pair, synthetic PDS4 pair via `.xml` and via the `.img` redirect, a missing
  sibling label, a corrupt upload through `/register`, and the `ch2_*`
  filename parser.

Sample images and PDS4/TIFF fixtures are generated programmatically on test
run (`tests/conftest.py`, `tests/test_loader.py`) — no binary test data is
committed.

---

## Dependencies & the Docker build

`requirements.txt` uses `>=` pins so the same file resolves on Python 3.11 (the
`python:3.11-slim` image) and on a newer local interpreter. The Docker image
needs only `libgl1` + `libglib2.0-0` for OpenCV — the headless wheel still
`dlopen`s them at import time.

`rasterio` is pinned exactly to `1.4.3` — **not** the latest release — because
its manylinux wheels dropped Python 3.11 support starting at `1.5.x` (Linux
wheels there start at `cp312`), and the base image here is `python:3.11-slim`.
`1.4.3` is the newest release that still ships a `cp311` wheel, so `pip
install` stays a wheel install with no system GDAL toolchain needed — no
`gdal-bin`/`libgdal-dev` in the `Dockerfile`. If the base image's Python
version is ever bumped to 3.12+, `rasterio` can be bumped to `1.5.x` at the
same time. Confirm the PDS4 driver is present after any `rasterio` version
change: `python -c "from osgeo import gdal; print(gdal.GetDriverByName('PDS4'))"`.

`scipy` and `scikit-image` stay commented out: no pipeline module imports them.
