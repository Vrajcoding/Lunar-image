"""
geometry.py — Robust geometric transformation estimation and planar homography warping.
Implements RANSAC / USAC homography fitting and degenerate model rejection.
"""
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def is_valid_homography(H: np.ndarray, max_cond: float = 1e6) -> bool:
    """Check if homography is non-degenerate and invertible."""
    if H is None or H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return False
    
    det = np.linalg.det(H)
    if abs(det) < 1e-8:
        return False

    try:
        cond = np.linalg.cond(H)
        if cond > max_cond or np.isnan(cond):
            return False
    except Exception:
        return False

    return True


def estimate_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    reproj_thresh: float = 3.0,
    confidence: float = 0.995,
    max_iters: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate homography matrix using RANSAC or USAC_MAGSAC (if available).
    
    Returns:
        H (3x3 float64 array): Homography matrix
        mask (N, 1 uint8 array): Inlier mask (1 for inlier, 0 for outlier)
    """
    if len(src_pts) < 4 or len(dst_pts) < 4:
        return np.eye(3, dtype=np.float64), np.zeros((len(src_pts), 1), dtype=np.uint8)

    src_f = src_pts.reshape(-1, 2).astype(np.float64)
    dst_f = dst_pts.reshape(-1, 2).astype(np.float64)

    # Prefer USAC_MAGSAC if supported by OpenCV build, fallback to standard RANSAC
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC

    try:
        H, mask = cv2.findHomography(
            src_f, dst_f, method, reproj_thresh, maxIters=max_iters, confidence=confidence
        )
    except Exception as e:
        logger.warning("Homography estimation failed: %s, falling back to cv2.RANSAC", e)
        H, mask = cv2.findHomography(src_f, dst_f, cv2.RANSAC, reproj_thresh)

    if H is None or not is_valid_homography(H):
        H = np.eye(3, dtype=np.float64)
        mask = np.zeros((len(src_pts), 1), dtype=np.uint8)

    if mask is None:
        mask = np.zeros((len(src_pts), 1), dtype=np.uint8)

    return H.astype(np.float64), mask.astype(np.uint8)


def warp_image(img: np.ndarray, H: np.ndarray, out_shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
    """Warp source image into reference coordinate frame using estimated homography."""
    h, w = out_shape[:2]
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
