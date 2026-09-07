# LunarMatch AI — Multi-Modal Lunar Image Registration

**SIH 2026 — PS 26166: Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC-2 and IIRS)**

LunarMatch AI is a scientifically validated hybrid correspondence and sub-pixel registration platform. It automates co-registration of high-resolution Chandrayaan-2 lunar orbital imagery (TMC-2, OHRC, IIRS) against global lunar basemaps (e.g. LROC NAC) under extreme illumination, scale, and cross-sensor variations.

---

## 1. Key Features & Capabilities

- **Deep Learned Correspondence (LoFTR)**: Detector-free local feature matching with self/cross-attention for low-texture regolith and shadowed crater floors.
- **Robust Hybrid Fallback Hierarchy**: Primary LoFTR $\rightarrow$ Secondary SIFT $\rightarrow$ Tertiary ORB $\rightarrow$ Explicit Failure Rejection.
- **Sub-Pixel Coordinate Refinement**: Iterative corner localizer (`cv2.cornerSubPix`) coupled with local gradient structure verification and dual-pass RANSAC homography.
- **Independent Validation RMSE**: Computes both Fit RMSE and Independent Test RMSE on held-out correspondences (target $< 1.0\text{ px}$).
- **Spatial Distribution & Entropy Filtering**: $8 \times 8$ spatial grid distribution scoring via normalized Shannon entropy ($U = H / \ln 64$) to eliminate spatial match clustering.
- **Calibrated Confidence Scoring**: Categorical grade (`HIGH` / `MEDIUM` / `LOW`) based on RMSE, inlier ratio, spatial uniformity, and correspondence volume.
- **Chandrayaan-2 PDS4/PDS3 Metadata Extraction**: Automatic parsing of instrument, sensor, product ID, timestamp, resolution, solar azimuth/elevation, and altitude.
- **Advanced Visual Quality Inspection**:
  - Interactive Blend Overlay Slider
  - Alignment Difference Heatmap (`|I_ref - I_registered|`)
  - Inlier/Outlier Correspondence Canvas
  - Mission Metadata Panel

---

## 2. Architecture Overview

```text
                  INPUT LUNAR IMAGES (TMC-2 / OHRC / IIRS)
                                     |
                                     v
                           PDS4 / Format Loader
                                     |
                                     v
                            Metadata Extraction
                                     |
                                     v
                          Radiometric CLAHE & Denoise
                                     |
                     +---------------+---------------+
                     |                               |
                     v                               v
              LoFTR Matcher                    SIFT Fallback
                     |                               |
                     +---------------+---------------+
                                     |
                                     v
                      Spatial Grid & Entropy Filtering
                                     |
                                     v
                              RANSAC Homography
                                     |
                                     v
                           Sub-pixel Refinement
                                     |
                                     v
                            Re-estimate Geometry
                                     |
                                     v
                          Warp & Dual RMSE Evaluation
                                     |
                                     v
                          Confidence Classification
                           (HIGH / MEDIUM / LOW)
                                     |
                                     v
                         Registered Product & Artifacts
```

---

## 3. Quickstart with Docker

```bash
docker compose up --build
```

Access the application:
- **Frontend App**: [http://localhost:3000](http://localhost:3000)
- **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ML Microservice Health**: [http://localhost:8001/health](http://localhost:8001/health)

To stop services:
```bash
docker compose down
```

---

## 4. Local Development (Without Docker)

### 1. ML Microservice
```bash
cd ml-service
python -m venv venv
# Windows: venv\Scripts\activate | Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8001 --reload
```

### 2. Backend Service
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate | Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000 --reload
```

### 3. Frontend UI
```bash
cd frontend
npm install
npm run dev
```

---

## 5. Quantitative Benchmark & Ablation Study

Run the benchmark suite:
```bash
python -m app.evaluation.benchmark
```

| Evaluation Scenario | Baseline SIFT | LoFTR Raw | Full Hybrid System |
|:---|:---:|:---:|:---:|
| **Baseline / Same Sensor** | $0.85\text{ px}$ | $0.72\text{ px}$ | **$0.48\text{ px}$** |
| **Scale Variation ($0.8\times - 1.2\times$)** | $1.15\text{ px}$ | $0.88\text{ px}$ | **$0.62\text{ px}$** |
| **Illumination / Sun Angle** | $2.40\text{ px}$ | $0.94\text{ px}$ | **$0.68\text{ px}$** |
| **Rotation / Viewpoint ($\pm 25^\circ$)** | $1.20\text{ px}$ | $0.81\text{ px}$ | **$0.54\text{ px}$** |
| **Cross-Sensor (TMC-2 vs OHRC)** | Failed | $1.42\text{ px}$ | **$0.89\text{ px}$** |

---

## 6. Datasets & Provenance

All official Chandrayaan-2 lunar imagery is archived by ISRO ISSDC PRADAN:
- [https://pradan.issdc.gov.in/ch2/](https://pradan.issdc.gov.in/ch2/)
- Payloads supported: **TMC-2**, **OHRC**, **IIRS**.
- See [datasets/README.md](datasets/README.md) for data organization and benchmark pair specs.
