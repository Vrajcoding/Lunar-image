# LunarMatch AI Experiments & Benchmark Registry

This directory contains quantitative benchmark artifacts, measured registration metrics, and reproducible ablation experiments for SIH 2026 PS 26166.

---

## 1. Running Benchmark Experiments

To run the automated multi-method ablation suite across synthetic and sample lunar imagery:

```bash
# Run from repository root
python -m ml-service.app.evaluation.benchmark
```

Results are saved to `experiments/results/benchmark_summary.json` and `experiments/results/benchmark_report.md`.

---

## 2. Benchmark Structure

- **Level 1 — Quantitative Ground Truth**: Parameterized geometric warp ($H_{GT}$) + illumination / scale / blur / noise changes.
- **Level 2 — Real Mission Pairs**: Overlapping Chandrayaan-2 and LROC stereo pairs.
- **Level 3 — Cross-sensor Alignment**: TMC-2 vs OHRC vs IIRS qualitative inspection and alignment overlays.
