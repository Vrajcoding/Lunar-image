"""
preprocess.py — Modular image preprocessing pipeline for lunar orbital imagery.
Provides radiometric normalization, adaptive histogram equalization (CLAHE),
invalid pixel cleaning, and multi-scale Gaussian pyramids.
"""
import cv2
import numpy as np
from app.pipeline.loader import load_image


def load_grayscale(path: str) -> np.ndarray:
    """Load grayscale image through format-aware loader."""
    img, _metadata = load_image(path)
    return img


def clean_invalid_pixels(img: np.ndarray, nodata_val: float | None = None) -> np.ndarray:
    """Replace NaNs, Infinities, and optional nodata values with scene median."""
    cleaned = img.copy()
    invalid_mask = np.isnan(cleaned) | np.isinf(cleaned)
    if nodata_val is not None:
        invalid_mask = invalid_mask | (cleaned == nodata_val)
    
    if np.any(invalid_mask):
        valid_pixels = cleaned[~invalid_mask]
        fill_val = float(np.median(valid_pixels)) if len(valid_pixels) > 0 else 0.0
        cleaned[invalid_mask] = fill_val
    return cleaned


def normalize_contrast(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Percentile-based linear contrast normalization to [0, 255] uint8."""
    if img.dtype == np.uint8 and p_low <= 0.0 and p_high >= 100.0:
        return img
    
    float_img = img.astype(np.float32)
    v_min, v_max = float(np.percentile(float_img, p_low)), float(np.percentile(float_img, p_high))
    if v_max <= v_min:
        v_min, v_max = float(float_img.min()), float(float_img.max())
        if v_max <= v_min:
            v_max = v_min + 1.0
            
    scaled = np.clip((float_img - v_min) / (v_max - v_min) * 255.0, 0.0, 255.0)
    return scaled.astype(np.uint8)


def normalize_illumination(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) for sun-angle invariance."""
    if img.dtype != np.uint8:
        img = normalize_contrast(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def denoise(img: np.ndarray, h: int = 7) -> np.ndarray:
    """Fast Non-Local Means Denoising tailored for grayscale lunar scenes."""
    if img.dtype != np.uint8:
        img = normalize_contrast(img)
    return cv2.fastNlMeansDenoising(img, h=h)


def build_pyramid(img: np.ndarray, levels: int = 4) -> list[np.ndarray]:
    """Generate multi-scale Gaussian downsampling pyramid."""
    pyramid = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(pyramid[-1])
        pyramid.append(img)
    return pyramid


def preprocess(path: str, levels: int = 4, apply_denoise: bool = True) -> list[np.ndarray]:
    """Full preprocessing pipeline returning pyramid of enhanced grayscale images."""
    img = load_grayscale(path)
    img = clean_invalid_pixels(img)
    img_norm = normalize_illumination(img)
    if apply_denoise:
        img_denoised = denoise(img_norm)
    else:
        img_denoised = img_norm
    return build_pyramid(img_denoised, levels)
