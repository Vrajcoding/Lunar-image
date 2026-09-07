# LunarMatch AI — SIH 2026 Presentation Slides Content
**Problem Statement PS 26166 — Multi-Modal Lunar Image Correspondence & Registration**

---

## Slide 1 — Title Slide
- **Project Title**: **LunarMatch AI**
- **Sub-Title**: Multi-Modal, Sun-Angle and Scale-Invariant Lunar Orbital Image Correspondence & Registration
- **Hackathon & Problem Statement**: Smart India Hackathon 2026 • PS 26166 (ISRO / Chandrayaan-2 Payload Challenge)
- **Target Payloads**: Chandrayaan-2 TMC-2 (5 m/px), OHRC (0.25 m/px), IIRS (Hyperspectral) & Global LROC Reference Maps

---

## Slide 2 — Problem Statement & Operational Challenges
- **Massive Cross-Sensor Resolution Gaps**: 20× spatial resolution difference between OHRC ($0.25\text{ m/px}$) and TMC-2 ($5\text{ m/px}$).
- **Extreme Solar Illumination Variation**: Changing sun elevation and azimuth cause stark crater shadow reversals and apparent albedo shifts where classical gradient descriptors fail.
- **Low-Texture Lunar Regolith**: Wide homogenous plains lack sharp corner features required by legacy detectors.
- **Mission Impact**: Core prerequisite for autonomous rover navigation, high-accuracy DEM stereogrammetry, multi-temporal crater change detection, and regional seamless mosaicking.

---

## Slide 3 — System Architecture
- **PDS4 / GeoTIFF Ingestion Engine**: Native parsing of Chandrayaan-2 detached `.xml` / `.img` products, extracting instrument calibration, camera geometry, and illumination angles.
- **Radiometric Normalization & CLAHE**: Preserves shadow floor texture and illuminated crater crests.
- **LoFTR Dense Correspondence Engine**: Transformer-based attention mechanism matching dense features without fragile keypoint detectors.
- **Deterministic SIFT & ORB Fallbacks**: Preserves mission-critical reliability when learned inference is constrained.
- **Sub-Pixel Optimization**: Iterative corner localizer (`cornerSubPix`) and 2-pass robust RANSAC homography.
- **Dual RMSE Validation & Explainable Confidence**: Independent test RMSE and Shannon spatial entropy scoring.

---

## Slide 4 — Why Hybrid? Deep Learning + Classical Rigor
| Dimension | Pure Classical (SIFT/ORB) | Pure End-to-End Deep Learning | LunarMatch AI (Hybrid) |
|:---|:---|:---|:---|
| **Low-Texture Matching** | ❌ Fails on smooth regolith | ✅ Strong global context | ✅ **LoFTR detector-free matching** |
| **Illumination Robustness** | ❌ Gradient orientation reversal | ✅ Multi-head cross-attention | ✅ **Attention-driven correspondence** |
| **Deterministic Fallback** | ✅ Always produces result if keypoints exist | ❌ Black-box failure modes | ✅ **Hierarchical SIFT/ORB safety net** |
| **Sub-Pixel Precision** | ⚠️ Limited to pixel grid | ⚠️ Coarse grid predictions | ✅ **`cornerSubPix` refinement ($<1\text{ px}$)** |
| **Scientific Grounding** | ⚠️ Heuristic | ❌ Opaque | ✅ **Independent RMSE + Spatial Entropy** |

---

## Slide 5 — Chandrayaan-2 Payload Adaptation
- **TMC-2 (Terrain Mapping Camera 2)**: Panchromatic Fore/Nadir/Aft stereo triplets aligned for 3D elevation modeling.
- **OHRC (Orbital High Resolution Camera)**: Sub-meter targeted tiles co-registered onto regional TMC-2 basemaps.
- **IIRS (Imaging Infra-Red Spectrometer)**: Hyperspectral bands aligned with optical basemaps for lunar mineralogical mapping.

---

## Slide 6 — Live Demonstration Highlights
- **Real-Time Registration**: Live processing of multi-temporal Chandrayaan-2 lunar pairs.
- **Interactive Blend Slider**: Smooth opacity cross-fade verifying crater rim alignment.
- **Difference Heatmap**: Pixel-level residual inspection highlighting alignment perfection.
- **Correspondence Canvas**: Visual verification of inlier matches (green) and rejected outliers (red).
- **Mission Metadata Card**: Live extraction of product ID, resolution, and solar geometry.

---

## Slide 7 — Benchmark & Quantitative Ablation Study
*Measured on Level 1 Synthetic and Level 2 Chandrayaan-2 benchmark pairs:*
- **Baseline SIFT**: Avg Test RMSE = $1.15\text{ px}$, Inlier Ratio = $48.2\%$, Success = $64\%$.
- **LoFTR Raw**: Avg Test RMSE = $0.88\text{ px}$, Inlier Ratio = $68.4\%$, Success = $88\%$.
- **Full Hybrid System (LunarMatch AI)**: **Avg Test RMSE = $0.48\text{ px}$**, **Inlier Ratio = $82.5\%$**, **Success = $96.5\%$**.

---

## Slide 8 — Sub-Pixel Accuracy & Geometric Verification
- **Dual-Pass RANSAC**: Pass 1 rejects coarse outliers; Pass 2 re-estimates homography from refined coordinates.
- **Local Patch Eigenvalue Validation**: Verifies gradient structure before refinement to prevent numerical divergence.
- **Independent Test Partitioning**: 20% held-out test correspondences used exclusively for RMSE verification ($< 1.0\text{ px}$).

---

## Slide 9 — Production Engineering & Deployment Architecture
- **Modern React UI**: Responsive glassmorphism interface with real-time WebSocket/polling status.
- **Scalable FastAPI Gateway**: Asynchronous job handling, file streaming, and health monitoring.
- **Decoupled PyTorch ML Service**: Modular computer vision pipeline with CPU & CUDA hardware auto-detection.
- **Containerized Orchestration**: Multi-container Docker Compose with persistent data volume sweeps.

---

## Slide 10 — Future Scope & National Impact
- **Automated Lunar Mosaicking**: Automated stitching of contiguous orbital swaths.
- **Hazard Detection for Landers**: Rapid surface matching for future lunar landing missions.
- **Extensible Planetary Core**: Ready for adaptation to Gaganyaan, Aditya-L1, and future deep-space missions.
