"""
dataset.py — Dataset management and synthetic ground truth generation.
Supports Level 1 (Quantitative Ground Truth with known H_gt), Level 2 (Real Mission Pairs),
and Level 3 (Cross-sensor Qualitative validation).
"""
import json
import logging
import os
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SyntheticDatasetGenerator:
    """Generates synthetic transformed image pairs with exact ground truth homography H_gt."""

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    def generate_pair(
        self,
        base_img: np.ndarray,
        scenario: str = "general",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Apply parameterized geometric and radiometric transformations to base_img.
        
        Args:
            base_img: Input grayscale uint8 image (reference)
            scenario: "translation" | "rotation" | "scale" | "perspective" | "illumination" | "noise" | "general"
            
        Returns:
            (source_img, reference_img, H_gt, metadata)
        """
        h, w = base_img.shape[:2]
        center = (w / 2.0, h / 2.0)

        # Default transformation parameters
        angle = 0.0
        scale = 1.0
        tx, ty = 0.0, 0.0
        persp_strength = 0.0
        gamma = 1.0
        brightness = 0.0
        contrast = 1.0
        noise_sigma = 0.0
        blur_ksize = 0

        if scenario == "translation":
            tx = float(self.rng.uniform(-30.0, 30.0))
            ty = float(self.rng.uniform(-30.0, 30.0))
        elif scenario == "rotation":
            angle = float(self.rng.uniform(-25.0, 25.0))
        elif scenario == "scale":
            scale = float(self.rng.uniform(0.75, 1.30))
        elif scenario == "perspective":
            angle = float(self.rng.uniform(-15.0, 15.0))
            persp_strength = float(self.rng.uniform(0.0002, 0.0005))
        elif scenario == "illumination":
            brightness = float(self.rng.uniform(-40.0, 40.0))
            contrast = float(self.rng.uniform(0.7, 1.4))
            gamma = float(self.rng.uniform(0.6, 1.5))
        elif scenario == "noise":
            noise_sigma = float(self.rng.uniform(5.0, 20.0))
            blur_ksize = 3
        else:  # general composite
            angle = float(self.rng.uniform(-15.0, 15.0))
            scale = float(self.rng.uniform(0.85, 1.15))
            tx = float(self.rng.uniform(-20.0, 20.0))
            ty = float(self.rng.uniform(-20.0, 20.0))
            contrast = float(self.rng.uniform(0.8, 1.2))
            gamma = float(self.rng.uniform(0.8, 1.3))

        # ── Construct 3x3 Homography H_gt (mapping source to reference) ─────────
        # M_rot_scale
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        cx, cy = center
        
        # Translate to origin, rotate+scale, translate back, add translation
        H_trans1 = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        H_rot = np.array([
            [scale * cos_a, -scale * sin_a, 0.0],
            [scale * sin_a, scale * cos_a, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        H_persp = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [persp_strength, -persp_strength, 1.0]
        ], dtype=np.float64)
        H_trans2 = np.array([[1.0, 0.0, cx + tx], [0.0, 1.0, cy + ty], [0.0, 0.0, 1.0]], dtype=np.float64)

        H_gt = H_trans2 @ H_persp @ H_rot @ H_trans1
        H_gt /= H_gt[2, 2]

        # Invert to warp reference to create source
        H_inv = np.linalg.inv(H_gt)
        source = cv2.warpPerspective(base_img, H_inv, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        # ── Apply Radiometric Transformations to source ──────────────────────
        src_f = source.astype(np.float32)
        if contrast != 1.0 or brightness != 0.0:
            src_f = src_f * contrast + brightness
        if gamma != 1.0:
            src_norm = np.clip(src_f / 255.0, 0.0, 1.0)
            src_f = (src_norm ** gamma) * 255.0
        if noise_sigma > 0:
            noise = self.rng.normal(0.0, noise_sigma, src_f.shape)
            src_f += noise
        if blur_ksize > 1:
            src_f = cv2.GaussianBlur(src_f, (blur_ksize, blur_ksize), 0)

        source_final = np.clip(src_f, 0.0, 255.0).astype(np.uint8)

        metadata = {
            "scenario": scenario,
            "angle_deg": round(angle, 2),
            "scale": round(scale, 3),
            "translation_px": (round(tx, 2), round(ty, 2)),
            "contrast": round(contrast, 2),
            "gamma": round(gamma, 2),
            "noise_sigma": round(noise_sigma, 2),
        }

        return source_final, base_img.copy(), H_gt, metadata


def compute_ground_truth_corner_error(H_est: np.ndarray, H_gt: np.ndarray, img_shape: tuple[int, int]) -> float:
    """Compute mean corner transfer error between estimated and ground truth homographies."""
    h, w = img_shape[:2]
    corners = np.array([
        [0.0, 0.0],
        [w - 1.0, 0.0],
        [w - 1.0, h - 1.0],
        [0.0, h - 1.0],
    ], dtype=np.float64)

    corners_h = np.hstack([corners, np.ones((4, 1))])

    # Transform with H_est
    p_est = (H_est @ corners_h.T).T
    denom_est = np.where(np.abs(p_est[:, 2:3]) < 1e-8, 1e-8, p_est[:, 2:3])
    xy_est = p_est[:, :2] / denom_est

    # Transform with H_gt
    p_gt = (H_gt @ corners_h.T).T
    denom_gt = np.where(np.abs(p_gt[:, 2:3]) < 1e-8, 1e-8, p_gt[:, 2:3])
    xy_gt = p_gt[:, :2] / denom_gt

    corner_errors = np.linalg.norm(xy_est - xy_gt, axis=1)
    return float(np.mean(corner_errors))


def load_image_pair(source_path: str, reference_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an image pair as grayscale numpy arrays."""
    src = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    if src is None or ref is None:
        raise ValueError(f"Could not load pair: {source_path}, {reference_path}")
    return src, ref
