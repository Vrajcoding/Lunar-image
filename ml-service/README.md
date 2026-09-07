# LunarMatch AI — ML Service Specification

## 1. Overview
The `ml-service` is the core scientific microservice responsible for:
- Format-aware loading of PDS4 (`.xml` + `.img`), PDS3 (`.lbl` + `.img`), GeoTIFF, and standard image formats.
- Mission metadata extraction for Chandrayaan-2 payloads (TMC-2, OHRC, IIRS).
- Radiometric CLAHE preprocessing and invalid pixel cleaning.
- Deep learned local feature correspondence (**LoFTR**) via Kornia.
- Classical deterministic fallback matching (**SIFT** and **ORB**).
- Spatial grid distribution limit and normalized Shannon spatial entropy filtering ($8 \times 8$ grid).
- Coarse and fine RANSAC homography estimation with degenerate model rejection.
- Sub-pixel corner coordinate refinement via `cv2.cornerSubPix`.
- Independent validation RMSE calculation (held-out test split).
- Explainable confidence classification (`HIGH`, `MEDIUM`, `LOW`).
- Difference residual image generation (`|I_ref - I_registered|`).

---

## 2. API Endpoints

### `POST /register`
Accepts `source`, `reference`, optional detached labels (`source_label`, `reference_label`), `band`, `mode` (`auto` / `accuracy` / `fast`), `source_sensor`, `reference_sensor`.

Returns JSON payload:
```json
{
  "status": "completed",
  "method": "LoFTR",
  "registered_image_path": "/data/.../registered.png",
  "difference_image_path": "/data/.../difference.png",
  "match_points_path": "/data/.../matches.csv",
  "metrics": {
    "rmse_px": 0.482,
    "fit_rmse_px": 0.412,
    "test_rmse_px": 0.482,
    "inlier_count": 142,
    "total_matches": 178,
    "match_count": 178,
    "inlier_ratio": 0.798,
    "uniformity_score": 0.825,
    "spatial_entropy": 3.44,
    "runtime_sec": 1.24,
    "confidence_score": 0.884,
    "confidence_level": "HIGH",
    "method": "LoFTR"
  },
  "input_metadata": { ... }
}
```

Or on correspondence failure:
```json
{
  "status": "failed",
  "reason": "INSUFFICIENT_CORRESPONDENCES",
  "error": "Could not establish at least 4 reliable correspondences across lunar image pair.",
  "metrics": { ... }
}
```
