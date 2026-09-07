"""
benchmark.py — Systematic evaluation and ablation benchmark suite.
Executes quantitative comparisons across SIFT, LoFTR, and the Full Hybrid System (Ablations A-E)
over synthetic and real lunar image datasets.
"""
import json
import logging
import os
import time
import cv2
import numpy as np

from app.pipeline.features import detect_and_describe
from app.pipeline.matching import (
    match_descriptors,
    enforce_uniform_distribution,
    filter_dense_correspondences,
    compute_spatial_entropy,
)
from app.pipeline.geometry import estimate_homography, warp_image, is_valid_homography
from app.pipeline.refine import subpixel_refine_points
from app.pipeline.evaluate import compute_rmse, compute_train_test_rmse, build_metrics
from app.pipeline.preprocess import normalize_illumination, denoise
from app.models.model_loader import get_loftr_matcher
from app.evaluation.dataset import SyntheticDatasetGenerator, compute_ground_truth_corner_error

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates multi-method benchmark and ablation experiments."""

    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.generator = SyntheticDatasetGenerator(random_seed=123)

    def run_method(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        method_name: str,
        H_gt: np.ndarray | None = None,
    ) -> dict:
        """Run a specific method variant: 'SIFT', 'LoFTR_Raw', 'LoFTR_Uniform', 'LoFTR_Refined', 'Full_Hybrid'."""
        t0 = time.time()
        h, w = reference.shape[:2]

        src_p = denoise(normalize_illumination(source))
        ref_p = denoise(normalize_illumination(reference))

        src_pts = np.zeros((0, 2), dtype=np.float32)
        dst_pts = np.zeros((0, 2), dtype=np.float32)
        uniformity_score = 0.0
        spatial_entropy = 0.0
        actual_method = method_name

        # ── Step 1: Matching based on method variant ──────────────────────────
        if method_name == "SIFT":
            kp_a, desc_a = detect_and_describe(src_p, 5000)
            kp_b, desc_b = detect_and_describe(ref_p, 5000)
            raw = match_descriptors(desc_a, desc_b, 0.75)
            filtered, uniformity_score = enforce_uniform_distribution(raw, kp_a, src_p.shape, 8, 15)
            if len(filtered) >= 4:
                src_pts = np.float32([kp_a[m.queryIdx].pt for m in filtered])
                dst_pts = np.float32([kp_b[m.trainIdx].pt for m in filtered])
                _, spatial_entropy, _ = compute_spatial_entropy(src_pts, src_p.shape, 8)

        elif "LoFTR" in method_name or method_name == "Full_Hybrid":
            loftr = get_loftr_matcher()
            if loftr is not None and loftr.is_available:
                l_res = loftr.match(src_p, ref_p, confidence_threshold=0.20)
                l_pts0 = l_res.get("points0", np.zeros((0, 2), dtype=np.float32))
                l_pts1 = l_res.get("points1", np.zeros((0, 2), dtype=np.float32))
                l_conf = l_res.get("confidence", np.zeros((0,), dtype=np.float32))

                if len(l_pts0) >= 8:
                    if method_name == "LoFTR_Raw":
                        src_pts, dst_pts = l_pts0, l_pts1
                        uniformity_score, spatial_entropy, _ = compute_spatial_entropy(src_pts, src_p.shape, 8)
                    else:  # LoFTR_Uniform, LoFTR_Refined, Full_Hybrid
                        f_pts0, f_pts1, _, u_score, s_ent = filter_dense_correspondences(
                            l_pts0, l_pts1, l_conf, src_p.shape, 8, 15
                        )
                        src_pts, dst_pts = f_pts0, f_pts1
                        uniformity_score, spatial_entropy = u_score, s_ent

            if len(src_pts) < 4 and method_name == "Full_Hybrid":
                # Fallback to SIFT
                actual_method = "Full_Hybrid (SIFT Fallback)"
                kp_a, desc_a = detect_and_describe(src_p, 5000)
                kp_b, desc_b = detect_and_describe(ref_p, 5000)
                raw = match_descriptors(desc_a, desc_b, 0.75)
                filtered, uniformity_score = enforce_uniform_distribution(raw, kp_a, src_p.shape, 8, 15)
                if len(filtered) >= 4:
                    src_pts = np.float32([kp_a[m.queryIdx].pt for m in filtered])
                    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in filtered])
                    _, spatial_entropy, _ = compute_spatial_entropy(src_pts, src_p.shape, 8)

        # ── Step 2: Geometric Estimation & Refinement ─────────────────────────
        if len(src_pts) < 4:
            return {
                "success": False,
                "method": actual_method,
                "match_count": 0,
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "fit_rmse_px": None,
                "test_rmse_px": None,
                "corner_error_px": None,
                "uniformity_score": 0.0,
                "spatial_entropy": 0.0,
                "runtime_sec": round(time.time() - t0, 3),
            }

        H, mask = estimate_homography(src_pts, dst_pts, 3.0)
        inlier_mask = mask.ravel().astype(bool)
        inlier_src = src_pts[inlier_mask]
        inlier_dst = dst_pts[inlier_mask]
        inlier_count = len(inlier_src)
        inlier_ratio = inlier_count / max(len(src_pts), 1)

        if inlier_count < 4 or not is_valid_homography(H):
            return {
                "success": False,
                "method": actual_method,
                "match_count": len(src_pts),
                "inlier_count": inlier_count,
                "inlier_ratio": round(inlier_ratio, 3),
                "fit_rmse_px": None,
                "test_rmse_px": None,
                "corner_error_px": None,
                "uniformity_score": uniformity_score,
                "spatial_entropy": spatial_entropy,
                "runtime_sec": round(time.time() - t0, 3),
            }

        # Subpixel refinement for refined variants
        if method_name in ("LoFTR_Refined", "Full_Hybrid", "SIFT"):
            ref_src = subpixel_refine_points(src_p, inlier_src)
            ref_dst = subpixel_refine_points(ref_p, inlier_dst)
            H_ref, mask2 = estimate_homography(ref_src, ref_dst, 1.5)
            if len(ref_src[mask2.ravel().astype(bool)]) >= 4 and is_valid_homography(H_ref):
                H = H_ref
                inlier_src = ref_src[mask2.ravel().astype(bool)]
                inlier_dst = ref_dst[mask2.ravel().astype(bool)]

        fit_rmse, test_rmse = compute_train_test_rmse(inlier_src, inlier_dst, H)

        corner_err = None
        if H_gt is not None:
            corner_err = compute_ground_truth_corner_error(H, H_gt, reference.shape)

        return {
            "success": True,
            "method": actual_method,
            "match_count": len(src_pts),
            "inlier_count": len(inlier_src),
            "inlier_ratio": round(len(inlier_src) / max(len(src_pts), 1), 4),
            "fit_rmse_px": round(fit_rmse, 4),
            "test_rmse_px": round(test_rmse, 4),
            "corner_error_px": round(corner_err, 4) if corner_err is not None else None,
            "uniformity_score": round(uniformity_score, 4),
            "spatial_entropy": round(spatial_entropy, 4),
            "runtime_sec": round(time.time() - t0, 3),
        }

    def run_synthetic_benchmark(self, base_image_paths: list[str], scenarios: list[str] | None = None) -> dict:
        """Run full benchmark suite over synthetic test image pairs."""
        if scenarios is None:
            scenarios = ["translation", "rotation", "scale", "perspective", "illumination", "noise", "general"]

        methods = ["SIFT", "LoFTR_Raw", "LoFTR_Uniform", "LoFTR_Refined", "Full_Hybrid"]
        results_by_scenario = {s: {m: [] for m in methods} for s in scenarios}

        logger.info("Running Synthetic Benchmark on %d base images across %d scenarios...", len(base_image_paths), len(scenarios))

        for img_path in base_image_paths:
            base_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if base_img is None:
                continue

            for scenario in scenarios:
                source, ref, H_gt, meta = self.generator.generate_pair(base_img, scenario=scenario)

                for method in methods:
                    res = self.run_method(source, ref, method, H_gt=H_gt)
                    results_by_scenario[scenario][method].append(res)

        # Aggregate summary statistics
        summary = {}
        for scenario in scenarios:
            summary[scenario] = {}
            for method in methods:
                runs = results_by_scenario[scenario][method]
                if not runs:
                    continue
                successful_runs = [r for r in runs if r["success"]]
                success_rate = len(successful_runs) / max(len(runs), 1)

                avg_test_rmse = float(np.mean([r["test_rmse_px"] for r in successful_runs])) if successful_runs else None
                avg_corner_err = float(np.mean([r["corner_error_px"] for r in successful_runs if r["corner_error_px"] is not None])) if successful_runs else None
                avg_inlier_ratio = float(np.mean([r["inlier_ratio"] for r in successful_runs])) if successful_runs else 0.0
                avg_uniformity = float(np.mean([r["uniformity_score"] for r in successful_runs])) if successful_runs else 0.0
                avg_runtime = float(np.mean([r["runtime_sec"] for r in runs]))

                summary[scenario][method] = {
                    "total_pairs": len(runs),
                    "success_rate": round(success_rate * 100.0, 1),
                    "avg_test_rmse_px": round(avg_test_rmse, 3) if avg_test_rmse is not None else "N/A",
                    "avg_corner_error_px": round(avg_corner_err, 3) if avg_corner_err is not None else "N/A",
                    "avg_inlier_ratio": round(avg_inlier_ratio * 100.0, 1),
                    "avg_uniformity_score": round(avg_uniformity, 3),
                    "avg_runtime_sec": round(avg_runtime, 3),
                }

        output_path = os.path.join(self.output_dir, "benchmark_summary.json")
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary


def run_full_benchmark(sample_dir: str = "ml-service/tests/sample_images") -> dict:
    """Convenience function to run benchmark on local sample imagery."""
    runner = BenchmarkRunner()
    sample_files = []
    if os.path.isdir(sample_dir):
        for fname in os.listdir(sample_dir):
            if fname.lower().endswith((".png", ".jpg", ".tif")):
                sample_files.append(os.path.join(sample_dir, fname))

    if not sample_files:
        # Create a synthetic base image for standalone execution
        dummy_base = np.zeros((512, 512), dtype=np.uint8)
        cv2.circle(dummy_base, (256, 256), 80, 200, -1)
        cv2.rectangle(dummy_base, (100, 100), (200, 200), 150, -1)
        os.makedirs("experiments/results", exist_ok=True)
        dummy_path = "experiments/results/dummy_base.png"
        cv2.imwrite(dummy_path, dummy_base)
        sample_files = [dummy_path]

    return runner.run_synthetic_benchmark(sample_files[:3])
