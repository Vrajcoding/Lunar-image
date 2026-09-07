# AGENT.md

# LunarMatch AI — SIH 2026 PS 26166 Upgrade Specification

## 0. Mission

You are the implementation agent responsible for upgrading the existing `Lunar-image` repository into a strong SIH 2026 solution for:

**PS 26166 — Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS).**

Do NOT rebuild the application from scratch.

The existing project already contains:

- React frontend
- FastAPI backend
- Separate ML service
- Docker Compose
- PDS/PDS4-oriented image loading
- preprocessing
- SIFT/ORB feature extraction
- feature matching
- spatial/grid filtering
- RANSAC homography
- sub-pixel refinement
- registration
- RMSE/inlier metrics
- automated tests

The objective is to evolve the existing implementation into a scientifically defensible hybrid correspondence and registration system.

---

# 1. Critical Rules

## Rule 1 — Do not destroy working functionality

Before modifying anything:

1. Inspect the complete repository.
2. Run the existing backend tests.
3. Run the existing ML-service tests.
4. Run the frontend.
5. Record the current behavior.
6. Preserve backward compatibility wherever practical.

Do not remove SIFT/ORB.

SIFT must remain available as a robust classical fallback.

## Rule 2 — Do not fabricate scientific results

NEVER invent:

- RMSE
- inlier ratio
- number of matches
- uniformity
- runtime
- accuracy
- benchmark results
- dataset size
- sensor performance

Every result shown in the UI, README, PPT, benchmark report, or API must come from an actual experiment.

If an example value is needed in documentation, explicitly label it:

`EXAMPLE — NOT A MEASURED RESULT`

## Rule 3 — Do not claim real Chandrayaan-2 validation until real data is used

The current repository contains sample/test images under:

`ml-service/tests/sample_images/`

These are testing assets, NOT the official Chandrayaan-2 benchmark dataset.

Do not call these OHRC, TMC, IIRS, or Chandrayaan-2 data unless their provenance has actually been verified.

---

# 2. Target Architecture

Implement:

```text
                 INPUT IMAGES
                      |
                      v
              PDS4 / Image Loader
                      |
                      v
             Metadata Extraction
                      |
                      v
            Radiometric / Intensity
               Preprocessing
                      |
          +-----------+-----------+
          |                       |
          v                       v
     LoFTR Matcher           SIFT Fallback
          |                       |
          +-----------+-----------+
                      |
                      v
              Match Confidence
                  Filtering
                      |
                      v
             Spatial Distribution
                 / Grid Filter
                      |
                      v
               RANSAC / USAC
              Geometric Model
                      |
                      v
             Sub-pixel Refinement
                      |
                      v
              Final Transform
                      |
                      v
                Image Warp
                      |
                      v
                 Evaluation
                      |
        +-------------+-------------+
        |             |             |
        v             v             v
       RMSE       Inlier Ratio   Uniformity
        |             |             |
        +-------------+-------------+
                      |
                      v
              Confidence Score
                      |
                      v
             Registered Product
```

---

# 3. Main Technical Strategy

The final system must be hybrid:

```text
Primary:
LoFTR

Fallback:
SIFT

Secondary fallback:
ORB
```

Decision flow:

```text
Image pair
    |
    v
Preprocessing
    |
    v
LoFTR
    |
    +---- enough reliable matches? ---- YES ---> geometry
    |
    NO
    |
    v
SIFT
    |
    +---- enough reliable matches? ---- YES ---> geometry
    |
    NO
    |
    v
ORB
    |
    +---- enough reliable matches? ---- YES ---> geometry
    |
    NO
    |
    v
REJECT REGISTRATION
```

Never silently return the reference image as a successful registration when registration failed.

---

# 4. Files To Modify

Inspect the exact repository structure before changing files.

Likely existing files:

```text
ml-service/app/main.py
ml-service/app/schemas.py
ml-service/app/config.py

ml-service/app/pipeline/loader.py
ml-service/app/pipeline/preprocess.py
ml-service/app/pipeline/features.py
ml-service/app/pipeline/matching.py
ml-service/app/pipeline/geometry.py
ml-service/app/pipeline/refine.py
ml-service/app/pipeline/evaluate.py

ml-service/requirements.txt
ml-service/Dockerfile
docker-compose.yml
frontend/src/
backend/
```

Do not assume filenames that are not present. Adapt to the actual repository.

---

# 5. New Files

Create:

```text
ml-service/app/models/
    __init__.py
    loftr_matcher.py
    model_loader.py

ml-service/app/pipeline/
    confidence.py
    registration.py

ml-service/app/evaluation/
    __init__.py
    dataset.py
    benchmark.py
    report.py

datasets/
    README.md

experiments/
    README.md
    results/

docs/
    benchmark.md
    architecture.md
```

---

# 6. LoFTR Integration

Use a pretrained LoFTR implementation first.

Preferred implementation:

**Kornia LoFTR**

Do NOT train LoFTR from scratch.

Do NOT immediately fine-tune LoFTR.

First establish:

```text
pretrained LoFTR
        |
        v
Chandrayaan-2 benchmark
        |
        v
failure analysis
        |
        v
optional fine-tuning
```

Verify the actual API against the installed/version-compatible Kornia implementation.

The implementation must support CPU and CUDA when available.

---

# 7. New File: models/loftr_matcher.py

Implement a clean wrapper:

```python
class LoFTRMatcher:

    def __init__(self, device=None):
        ...

    def match(self, image0, image1):
        ...
```

Requirements:

- grayscale input support
- float normalization
- CPU support
- CUDA support when available
- `torch.no_grad()`
- model loaded once
- model not recreated for every request
- return keypoints0, keypoints1, confidence

Suggested return:

```python
{
    "points0": np.ndarray,
    "points1": np.ndarray,
    "confidence": np.ndarray,
    "method": "loftr"
}
```

---

# 8. Device Configuration

Add configuration:

```text
USE_GPU=true/false
DEVICE=auto/cpu/cuda
LOFTR_ENABLED=true
SIFT_FALLBACK_ENABLED=true
ORB_FALLBACK_ENABLED=true
```

Default:

```text
DEVICE=auto
```

Behavior:

```text
CUDA available -> CUDA
otherwise -> CPU
```

The application must continue working on CPU.

---

# 9. Preprocessing

Modify:

`ml-service/app/pipeline/preprocess.py`

Maintain the existing preprocessing pipeline but make it explicitly modular:

```text
load
 |
normalize dtype
 |
remove invalid pixels
 |
contrast normalization
 |
CLAHE
 |
optional denoise
 |
LoFTR-ready grayscale image
```

Do not aggressively preprocess images in a way that destroys lunar texture.

Every preprocessing operation should be configurable.

---

# 10. PDS4 / Chandrayaan-2 Loader

Modify:

`ml-service/app/pipeline/loader.py`

Preserve existing support.

Add metadata extraction where available.

Potential metadata fields:

```text
instrument
sensor
product_id
acquisition_time
pixel_scale
image_width
image_height
band
resolution
sun_azimuth
sun_elevation
spacecraft_altitude
```

Do not invent metadata.

If metadata is unavailable, return `null` instead of guessing.

---

# 11. Dataset

## Primary real dataset

Use official Chandrayaan-2 data from:

**ISRO / ISSDC PRADAN**

Target payloads:

```text
OHRC
TMC / TMC-2
IIRS
```

Official archive:

`https://pradan.issdc.gov.in/ch2/`

The benchmark must document:

- source
- product ID
- sensor
- acquisition information
- file format
- access/licensing conditions where applicable

---

# 12. Dataset Organization

Create:

```text
datasets/
└── chandrayaan2/
    ├── raw/
    │   ├── ohrc/
    │   ├── tmc/
    │   └── iirs/
    │
    ├── processed/
    │
    ├── pairs/
    │   ├── same_sensor/
    │   ├── cross_sensor/
    │   ├── illumination/
    │   ├── scale/
    │   └── viewpoint/
    │
    └── metadata/
```

DO NOT commit large raw mission files into Git.

Add raw-data paths to `.gitignore`.

Commit only:

- metadata manifests
- sample images if legally permitted
- benchmark configuration
- scripts
- documentation

---

# 13. Initial Dataset Size

First target:

```text
50–100 real image pairs
```

Preferred final benchmark:

```text
150–300 pairs
```

Suggested target distribution:

```text
TMC-TMC              30
OHRC-OHRC            30
IIRS-IIRS            20
TMC-OHRC             30
TMC-IIRS             20
OHRC-IIRS            20
Illumination         20
Scale                20
Viewpoint            20
```

These are target counts, not claims that the data already exists.

If a category cannot be constructed reliably, document the limitation instead of fabricating pairs.

---

# 14. Dataset Benchmark Levels

Implement three levels.

## Level 1 — Synthetic Ground Truth

Start from real lunar imagery.

Apply known transformations:

```text
translation
rotation
scale
perspective
brightness
contrast
gamma
blur
noise
```

Store the exact transformation matrix:

```text
H_ground_truth
```

This provides quantitative ground truth.

## Level 2 — Real Mission Pairs

Use overlapping Chandrayaan-2 imagery.

Establish reliable reference correspondences through:

- mission metadata
- overlap information
- manual expert verification
- or another defensible reference method

Document exactly how reference points were produced.

## Level 3 — Qualitative Real Data

Show:

```text
source
reference
correspondence
registered
overlay
difference
```

Do not assign fake pixel-level ground truth to these pairs.

---

# 15. Matching

Modify:

`ml-service/app/pipeline/matching.py`

Pipeline:

```text
LoFTR confidence filtering
        |
        v
spatial grid filtering
        |
        v
uniform correspondence selection
```

Do not simply select the highest-confidence matches globally.

High-confidence matches can still be spatially clustered.

---

# 16. Spatial Uniformity

Retain the existing 8 × 8 grid approach.

Add normalized spatial entropy.

For every grid cell:

```text
p_i = matches_in_cell / total_matches
```

Calculate:

```text
entropy = -sum(p_i * log(p_i))
```

Normalize:

```text
uniformity =
entropy / log(number_of_cells)
```

Range:

```text
0 -> highly clustered
1 -> highly uniform
```

Expose:

```text
uniformity_score
spatial_entropy
covered_cells
```

---

# 17. Geometry

Modify:

`ml-service/app/pipeline/geometry.py`

Pipeline:

```text
correspondences
    |
    v
quality filtering
    |
    v
RANSAC
    |
    v
homography
```

If the installed OpenCV environment supports robust USAC variants, implement them as an optional benchmark.

Do NOT claim USAC is better until benchmark evidence proves it.

Return:

```python
{
    "transform": H,
    "inlier_mask": mask,
    "inlier_count": ...,
    "inlier_ratio": ...
}
```

Reject degenerate transformations.

---

# 18. Sub-pixel Refinement

Modify:

`ml-service/app/pipeline/refine.py`

Retain `cv2.cornerSubPix()` if appropriate.

Only refine points that have enough local image information.

Pipeline:

```text
match
 |
RANSAC inlier
 |
local patch validation
 |
cornerSubPix
 |
refined points
 |
re-estimate transform
```

The final transform must be estimated using refined coordinates.

---

# 19. Independent RMSE

This is mandatory.

Do NOT use only the same points used to estimate the transform.

For synthetic/ground-truth data:

```text
80% points -> transformation estimation
20% points -> independent evaluation
```

Calculate:

```text
test_rmse_px
```

Also retain:

```text
fit_rmse_px
```

The UI/PPT should emphasize:

**Independent Test RMSE**

rather than only fitting error.

---

# 20. Evaluation Metrics

Every successful registration should report:

```text
match_count
inlier_count
inlier_ratio
fit_rmse_px
test_rmse_px
uniformity_score
spatial_entropy
runtime_sec
confidence_score
method
```

Example schema:

```json
{
  "method": "LoFTR",
  "match_count": 0,
  "inlier_count": 0,
  "inlier_ratio": 0.0,
  "fit_rmse_px": 0.0,
  "test_rmse_px": 0.0,
  "uniformity_score": 0.0,
  "spatial_entropy": 0.0,
  "runtime_sec": 0.0,
  "confidence_score": 0.0
}
```

All numeric values must be measured.

---

# 21. Confidence System

Create:

`ml-service/app/pipeline/confidence.py`

Calculate a confidence score using:

```text
RMSE
inlier ratio
uniformity
match count
```

Suggested classification:

```text
HIGH
MEDIUM
LOW
```

Initial engineering thresholds can be configurable.

Do not present thresholds as scientifically validated until benchmarked.

---

# 22. Registration Failure

Mandatory behavior.

The system must NEVER silently create an identity/reference registration when correspondence fails.

Return:

```json
{
  "status": "failed",
  "reason": "INSUFFICIENT_CORRESPONDENCES"
}
```

Possible failure reasons:

```text
INVALID_IMAGE
INSUFFICIENT_MATCHES
LOW_CONFIDENCE
LOW_INLIER_RATIO
LOW_SPATIAL_COVERAGE
DEGENERATE_TRANSFORM
HIGH_RMSE
PROCESSING_ERROR
```

Frontend must display failure clearly.

---

# 23. New File: pipeline/registration.py

This is the central orchestration layer.

Expected flow:

```python
def register_images(source, reference):

    source = load_image(source)
    reference = load_image(reference)

    metadata = extract_metadata(...)

    source_p = preprocess(source)
    reference_p = preprocess(reference)

    loftr_result = loftr.match(
        source_p,
        reference_p
    )

    if reliable(loftr_result):
        method = "LoFTR"
        points0, points1 = filter_loftr(...)
    else:
        sift_result = classical_sift(...)
        method = "SIFT"

        if not reliable(sift_result):
            orb_result = classical_orb(...)
            method = "ORB"

            if not reliable(orb_result):
                return failed_result(...)

    if not reliable(...):
        return failed_result(...)

    points0, points1 = enforce_uniformity(...)

    geometry = estimate_geometry(...)

    refined_points = refine(...)

    final_transform = reestimate_geometry(...)

    registered = warp(...)

    metrics = evaluate(...)

    confidence = calculate_confidence(...)

    return final_result(...)
```

Keep orchestration separate from the FastAPI route.

---

# 24. main.py

Reduce business logic in:

`ml-service/app/main.py`

The API should call:

```python
registration_engine.register(...)
```

Do not keep the entire computer vision pipeline inside the HTTP endpoint.

---

# 25. API Response

Successful response:

```json
{
  "status": "success",
  "job_id": "...",
  "method": "LoFTR",
  "registered_image": "...",
  "matches": "...",
  "metrics": {},
  "metadata": {}
}
```

Failed response:

```json
{
  "status": "failed",
  "reason": "...",
  "metrics": {}
}
```

Maintain backward compatibility where possible.

---

# 26. Frontend Changes

Do not redesign the whole frontend.

Add sensor selection:

```text
Source Sensor
- TMC-2
- OHRC
- IIRS
- Unknown

Reference Sensor
- TMC-2
- OHRC
- IIRS
- External reference
```

Add processing mode:

```text
Auto
Maximum Accuracy
Fast
```

---

# 27. Progress UI

Use scientifically meaningful stages:

```text
Loading image...
Reading PDS4 metadata...
Normalizing image...
Running LoFTR correspondence...
Filtering correspondence confidence...
Enforcing spatial distribution...
Estimating geometric transformation...
Sub-pixel refinement...
Generating registered image...
Calculating validation metrics...
```

---

# 28. Metrics UI

Display:

```text
RMSE
Inlier Count
Inlier Ratio
Uniformity
Confidence
Runtime
Method
```

Example structure:

```text
Method: LoFTR

Test RMSE:       X.XX px
Inliers:         XXX
Inlier Ratio:    XX.X %
Uniformity:      0.XX
Confidence:      HIGH
Runtime:         X.XX sec
```

Do not show placeholder/fake values in the final demo.

---

# 29. Result Visualization

Add:

```text
Registration
Correspondences
Difference
Metadata
```

Correspondence visualization:

```text
Green = accepted/inlier
Red = rejected/outlier
```

Difference image:

```text
abs(reference - registered)
```

Metadata panel:

```text
Sensor
Product ID
Acquisition time
Resolution
Available illumination metadata
```

---

# 30. Benchmark Framework

Create:

`ml-service/app/evaluation/benchmark.py`

Compare:

```text
SIFT
LoFTR
Full Pipeline
```

Recommended ablation:

```text
A = SIFT

B = LoFTR

C = LoFTR + spatial uniformity

D = LoFTR + spatial uniformity + subpixel refinement

E = Full system
```

Measure:

```text
RMSE
Inlier Ratio
Uniformity
Failure Rate
Runtime
```

---

# 31. Benchmark Matrix

Produce:

```text
| Scenario       | SIFT | LoFTR | Full |
|----------------|------|-------|------|
| Same Sensor    |      |       |      |
| Scale          |      |       |      |
| Illumination   |      |       |      |
| Viewpoint      |      |       |      |
| TMC-OHRC       |      |       |      |
| TMC-IIRS       |      |       |      |
| OHRC-IIRS      |      |       |      |
```

Populate only with actual measurements.

---

# 32. Performance Targets

These are engineering TARGETS, not guaranteed results.

Aim for:

```text
Independent RMSE < 1.0 px
Inlier ratio > 70%
Uniformity > 0.70
Reliable matches > 100 where image content permits
```

A strong result would be:

```text
RMSE < 0.75 px
Inlier ratio > 80%
Uniformity > 0.80
```

Do not sacrifice correctness merely to hit these numbers.

---

# 33. Runtime

Measure:

```text
preprocessing time
LoFTR time
fallback time
geometry time
refinement time
total time
```

GPU and CPU must be reported separately if both are tested.

Example:

```text
CPU:
Total = X sec

GPU:
Total = X sec
```

Use actual measurements.

---

# 34. Tests

Preserve all existing tests.

Add tests for:

```text
LoFTR loading
LoFTR matching
SIFT fallback
ORB fallback
confidence filtering
uniformity
spatial entropy
RANSAC
subpixel refinement
independent RMSE
failure conditions
metadata parsing
API success
API failure
```

Test cases:

```text
same image
translated image
scaled image
rotated image
brightness changed image
blurred image
low-texture image
insufficient-match image
```

---

# 35. Regression Requirement

Before declaring the upgrade complete:

```text
all existing tests pass
new tests pass
frontend builds
backend starts
ML service starts
Docker Compose starts
CPU mode works
GPU mode works if hardware is available
```

Do not break original functionality.

---

# 36. Security / Production Hygiene

Do not commit:

```text
.env
large raw datasets
model credentials
private access tokens
```

Ensure:

```text
.env.example
```

contains safe placeholders.

---

# 37. Documentation

Update:

```text
README.md
backend.md
ml-service/README.md
```

Document:

1. Problem statement
2. Architecture
3. Installation
4. Dataset
5. LoFTR
6. SIFT fallback
7. PDS4 support
8. Metrics
9. Benchmark methodology
10. Limitations
11. Reproduction steps

---

# 38. SIH PPT Requirements

Create material for approximately 10 slides.

## Slide 1

```text
LunarMatch AI

Multi-modal, Sun-angle & Scale-Invariant
Lunar Image Correspondence & Registration

SIH 2026 — PS 26166
```

## Slide 2

Problem:

```text
Different sensors
Different scale
Different illumination
Different viewpoint
```

## Slide 3

Solution architecture.

## Slide 4

Why hybrid:

```text
LoFTR
+
SIFT fallback
+
spatial uniformity
+
robust geometry
+
sub-pixel refinement
```

## Slide 5

Chandrayaan-2:

```text
OHRC
TMC/TMC-2
IIRS
```

## Slide 6

Real demonstration.

## Slide 7

Benchmark:

```text
SIFT vs LoFTR vs Full System
```

## Slide 8

Sub-pixel validation.

Show:

```text
fit RMSE
vs
independent test RMSE
```

## Slide 9

Engineering architecture:

```text
React
FastAPI
ML service
PyTorch/OpenCV
PDS4
Docker
```

## Slide 10

Impact:

```text
Automated lunar co-registration
        ↓
Multi-temporal analysis
        ↓
Mosaicking
        ↓
Terrain/DEM analysis
        ↓
Future lunar mission processing
```

---

# 39. Live Demo

The complete live demo should fit within approximately 3 minutes.

Sequence:

```text
1. Explain problem
2. Upload actual Chandrayaan-2 pair
3. Show metadata
4. Start registration
5. Show LoFTR correspondence
6. Show inliers/outliers
7. Show registered image
8. Show overlay
9. Show difference image
10. Show measured metrics
11. Compare against SIFT
```

Do NOT spend most of the demo showing source code.

---

# 40. Judge Questions To Prepare

## Why LoFTR?

Because learned detector-free correspondence can use global image context and is designed for challenging image matching.

## Why SIFT?

It provides a deterministic, interpretable and computationally simpler fallback.

## Why not only deep learning?

Because the system needs robustness, fallback behavior, explainability and reliable geometric verification.

## How is sub-pixel accuracy validated?

By refining coordinates and evaluating transformation accuracy on independent/held-out correspondences or known synthetic ground truth.

## What happens when registration fails?

The system explicitly rejects the result.

## How do you handle different sensors?

Sensor metadata + intensity normalization + learned correspondence + separate cross-sensor evaluation.

## How do you prove your method is better?

Ablation and benchmark:

```text
SIFT
vs
LoFTR
vs
Full System
```

---

# 41. Scientific Honesty

Do not write:

- "Our system is 100% accurate."
- "Guaranteed sub-pixel accuracy."
- "Works on all lunar images."

Instead write:

> "The system evaluates registration quality using independent RMSE, inlier ratio, spatial uniformity and confidence metrics."

Only state measured performance after running the benchmark.

---

# 42. Development Order

Implement in this exact order.

## Phase 1 — Baseline

```text
Run current tests
Run current application
Record baseline
```

## Phase 2 — LoFTR

```text
Install dependencies
Create LoFTR wrapper
Test on existing sample images
```

## Phase 3 — Hybrid Pipeline

```text
LoFTR
 ↓
SIFT
 ↓
ORB
 ↓
failure
```

## Phase 4 — Scientific Evaluation

```text
independent RMSE
spatial entropy
confidence
failure detection
```

## Phase 5 — Real Dataset

```text
download authorized Chandrayaan-2 data
build metadata manifest
construct benchmark pairs
```

## Phase 6 — Benchmark

```text
SIFT
LoFTR
Full System
```

## Phase 7 — Frontend

```text
sensor metadata
method
metrics
confidence
difference visualization
```

## Phase 8 — Docker/GPU

```text
CPU
CUDA
GPU benchmark
```

## Phase 9 — Final Validation

```text
tests
benchmark
demo
documentation
PPT
```

---

# 43. Definition of Done

The project is SIH-upgrade complete only when:

- [ ] Existing tests pass
- [ ] LoFTR works
- [ ] SIFT fallback works
- [ ] ORB fallback works
- [ ] Registration failure is explicit
- [ ] Spatial uniformity is measured
- [ ] Independent RMSE is implemented
- [ ] Sub-pixel refinement is implemented
- [ ] PDS4 metadata is preserved/extracted where available
- [ ] Official Chandrayaan-2 data has been tested
- [ ] TMC/OHRC/IIRS evaluation is documented where data permits
- [ ] SIFT vs LoFTR benchmark exists
- [ ] Full-system ablation exists
- [ ] Real measured metrics are stored
- [ ] Frontend displays scientific metrics
- [ ] Difference visualization works
- [ ] CPU mode works
- [ ] GPU mode works if hardware is available
- [ ] Docker works
- [ ] README is updated
- [ ] SIH demo flow works
- [ ] No fabricated results exist

---

# 44. Final Engineering Principle

Do not optimize for the appearance of an AI project.

Optimize for:

```text
REAL DATA
    +
ROBUST CORRESPONDENCE
    +
GEOMETRIC VALIDATION
    +
SUB-PIXEL REFINEMENT
    +
MEASURABLE RESULTS
    +
FAILURE DETECTION
```

The final product should demonstrate that it can take real lunar imagery and produce a quantitatively validated registered product.

The existing SIFT system is the foundation.

LoFTR is the learned correspondence upgrade.

The benchmark is the proof.

The frontend is the demonstration.

The metrics are the scientific evidence.

The goal is not merely to say:

> "We use AI."

The goal is to demonstrate:

> "Our system reliably establishes and validates image correspondence under the difficult illumination, scale, viewpoint and cross-sensor conditions described by PS 26166."
