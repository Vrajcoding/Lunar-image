import cv2
import numpy as np

def estimate_homography(src_pts: np.ndarray, dst_pts: np.ndarray, reproj_thresh: float = 3.0):
    if len(src_pts) < 4:
        # Fallback identity matrix
        return np.eye(3, dtype=np.float32), np.ones((len(src_pts), 1), dtype=np.uint8)
        
    H, mask = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, reproj_thresh, maxIters=5000, confidence=0.995
    )
    if H is None:
        H = np.eye(3, dtype=np.float32)
        mask = np.ones((len(src_pts), 1), dtype=np.uint8)
    return H, mask

def warp_image(img: np.ndarray, H: np.ndarray, out_shape):
    h, w = out_shape[:2]
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC)
