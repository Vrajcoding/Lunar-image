# Chandrayaan-2 Lunar Dataset Management

This directory documents the official dataset sources, structure, and pair definitions for **PS 26166 — Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC-2, and IIRS)**.

---

## 1. Official Data Sources

All official mission products are obtained from the ISRO ISSDC PRADAN portal:
- **Archive URL**: [https://pradan.issdc.gov.in/ch2/](https://pradan.issdc.gov.in/ch2/)
- **Authorized Payloads**:
  1. **OHRC (Orbital High Resolution Camera)**: Sub-meter (~0.25 m/pixel) panchromatic high-resolution targeted lunar surface imagery.
  2. **TMC-2 (Terrain Mapping Camera 2)**: 5 m/pixel panchromatic stereo triplets (Fore, Nadir, Aft) for lunar 3D digital elevation and geomorphology.
  3. **IIRS (Imaging Infra-Red Spectrometer)**: Hyperspectral / multi-band infrared imagery (0.8–5.0 µm) for lunar mineralogical mapping.

---

## 2. Directory Hierarchy

```text
datasets/
└── chandrayaan2/
    ├── raw/
    │   ├── ohrc/        # Large raw mission files (.img, .xml) - Ignored in Git
    │   ├── tmc/         # Large raw mission files (.img, .xml) - Ignored in Git
    │   └── iirs/        # Large raw mission files (.img, .xml) - Ignored in Git
    ├── processed/       # Calibrated, radiometrically normalized tiles
    ├── pairs/           # Standardized benchmark evaluation pairs
    │   ├── same_sensor/ # TMC-TMC, OHRC-OHRC, IIRS-IIRS
    │   ├── cross_sensor/# TMC-OHRC, TMC-IIRS, OHRC-IIRS
    │   ├── illumination/# Multi-temporal varying sun-angle / solar elevation
    │   ├── scale/       # Cross-resolution multi-scale pairs
    │   └── viewpoint/   # Fore / Nadir / Aft stereo angle variations
    └── metadata/        # JSON/CSV manifests of dataset pairs and provenance
```

> [!NOTE]
> Large raw mission raster data files must never be checked into Git. Use `.gitignore` to track only sample manifests and benchmark configurations.
