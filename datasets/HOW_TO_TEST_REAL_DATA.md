# Comprehensive Guide: Testing LunarMatch AI on Real Datasets

This guide details all available methods to test **LunarMatch AI** on real lunar satellite imagery (ISRO Chandrayaan-2 TMC-2, OHRC, IIRS, and NASA LROC).

---

## 🚀 Quick Options Summary

| Method | Best For | How to Run |
|:---|:---|:---|
| **Option 1: Web Interface** | Interactive visual testing, overlay sliders, heatmap inspection | Open UI at `http://localhost:5173` and upload images + XML |
| **Option 2: CLI Tool** | Quick single-command testing & artifact generation | `python scripts/test_real_dataset.py --source <src> --reference <ref>` |
| **Option 3: Pre-structured Pairs** | Immediate multi-sensor testing (sun angles, cross-resolution) | `python scripts/prepare_sample_real_data.py` |
| **Option 4: ISRO ISSDC PRADAN** | Downloading official Chandrayaan-2 mission products | Download from `https://pradan.issdc.gov.in/ch2/` |

---

## 1. Option 1: Testing via Web Application (Interactive)

1. **Start the ML Service**:
   ```bash
   cd ml-service
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

2. **Start the Backend**:
   ```bash
   cd backend
   uvicorn app.main:app --host 0.0.0.0 --port 5000
   ```

3. **Start the Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

4. **Upload & Test**:
   - Navigate to `http://localhost:5173` in your browser.
   - Drag & drop your **Source Image** (e.g., OHRC or morning TMC-2).
   - Drag & drop your **Reference Image** (e.g., TMC-2 or afternoon TMC-2).
   - *(Optional)* Attach the corresponding ISRO **PDS4 XML labels** to inspect solar angles and payload telemetry.
   - Click **Start Registration** to view:
     - 🎚️ **Interactive Overlay Slider & Side-by-Side Comparison**
     - 🌡️ **Residual Difference Heatmap**
     - 📍 **Dense Match Point Vectors Canvas**
     - 📊 **Scientific Metrics** (Dual RMSE, Spatial Entropy, Inlier Ratio, Confidence badge)

---

## 2. Option 2: Testing via Command-Line Tool

Use the CLI script [`scripts/test_real_dataset.py`](file:///d:/Sih-hackthon-project/scripts/test_real_dataset.py):

### A. Basic Registration on Any Two Images (PNG, TIFF, JPG)
```bash
python scripts/test_real_dataset.py --source path/to/source.png --reference path/to/reference.png
```

### B. Deep Learning (LoFTR) Mode
```bash
python scripts/test_real_dataset.py --source path/to/source.png --reference path/to/reference.png --detector loftr
```

### C. With Chandrayaan-2 PDS4 XML Metadata
```bash
python scripts/test_real_dataset.py \
  --source path/to/ch2_ohrc_image.tif \
  --reference path/to/ch2_tmc_image.tif \
  --source-xml path/to/ch2_ohrc_label.xml \
  --reference-xml path/to/ch2_tmc_label.xml
```

**Outputs generated in `experiments/results/real_tests/<pair_name>/`**:
- `registered.png` — High-precision warped and aligned image
- `difference.png` — Color-mapped residual error heatmap
- `matches.csv` — Full coordinate correspondence list with inlier flags
- `evaluation_summary.json` — Detailed JSON metrics (RMSE, entropy, runtime, confidence)

---

## 3. Option 3: Instant Test with Pre-built Real-Format Pairs

We have prepared structured multi-sensor Chandrayaan-2 sample pairs inside `datasets/chandrayaan2/pairs/`:

```bash
# 1. Generate sample pairs (if not already present):
python scripts/prepare_sample_real_data.py

# 2. Test TMC-2 Sun-Angle Illumination Pair (Morning vs Afternoon):
python scripts/test_real_dataset.py \
  --source datasets/chandrayaan2/pairs/illumination_tmc/tmc_source.png \
  --reference datasets/chandrayaan2/pairs/illumination_tmc/tmc_reference.png \
  --source-xml datasets/chandrayaan2/pairs/illumination_tmc/tmc_source.xml \
  --reference-xml datasets/chandrayaan2/pairs/illumination_tmc/tmc_reference.xml

# 3. Test Cross-Sensor Multi-Resolution Pair (OHRC 0.25m vs TMC-2 5.0m):
python scripts/test_real_dataset.py \
  --source datasets/chandrayaan2/pairs/cross_sensor_tmc_ohrc/ohrc_source.png \
  --reference datasets/chandrayaan2/pairs/cross_sensor_tmc_ohrc/tmc_reference.png \
  --source-xml datasets/chandrayaan2/pairs/cross_sensor_tmc_ohrc/ohrc_source.xml \
  --reference-xml datasets/chandrayaan2/pairs/cross_sensor_tmc_ohrc/tmc_reference.xml \
  --detector loftr
```

---

## 4. Option 4: Downloading Official Mission Data from ISRO ISSDC PRADAN

To test with full-scale mission archives:

### Step 1: Register on ISRO ISSDC PRADAN
1. Go to [https://pradan.issdc.gov.in/ch2/](https://pradan.issdc.gov.in/ch2/).
2. Create a free user account and log in.

### Step 2: Search for Overlapping Footprints
1. In the **Map / Search Interface**, select an interesting lunar feature:
   - **Apollo 11 / 17 Landing Sites** (Latitude 0.67° N, Longitude 23.47° E)
   - **Tycho Crater** (Latitude 43.3° S, Longitude 11.2° W)
   - **Shackleton / South Pole** (Latitude 89.9° S, Longitude 0.0° E)
2. Filter by payloads:
   - **TMC-2**: Search for `CALIBRATED` or `DERIVED` products (Nadir, Fore, or Aft).
   - **OHRC**: Search for targeted high-resolution observations overlapping the same coordinates.
   - **IIRS**: Search for hyperspectral radiance/reflectance cubes.

### Step 3: Download & Place Products
Download the archive zip file for both observations. Each product contains:
- Image raster (`.img` or `.tif`)
- Detached PDS4 XML label (`.xml`)

Place them into:
```text
datasets/chandrayaan2/raw/
├── ohrc/
│   ├── ch2_ohr_ncp_20230812T110000_d_img_d18.tif
│   └── ch2_ohr_ncp_20230812T110000_d_img_d18.xml
└── tmc/
    ├── ch2_tmc_ncn_20230812T110500_d_img_d18.tif
    └── ch2_tmc_ncn_20230812T110500_d_img_d18.xml
```

### Step 4: Run Registration
Run our format-aware loader directly:
```bash
python scripts/test_real_dataset.py \
  --source datasets/chandrayaan2/raw/ohrc/ch2_ohr_ncp_20230812T110000_d_img_d18.tif \
  --reference datasets/chandrayaan2/raw/tmc/ch2_tmc_ncn_20230812T110500_d_img_d18.tif \
  --source-xml datasets/chandrayaan2/raw/ohrc/ch2_ohr_ncp_20230812T110000_d_img_d18.xml \
  --reference-xml datasets/chandrayaan2/raw/tmc/ch2_tmc_ncn_20230812T110500_d_img_d18.xml
```
