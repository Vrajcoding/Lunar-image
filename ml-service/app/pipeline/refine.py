import cv2
import numpy as np

def subpixel_refine_points(img: np.ndarray, points: np.ndarray, win_size=(5, 5)):
    if len(points) == 0:
        return points
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    try:
        refined = cv2.cornerSubPix(img, pts, win_size, (-1, -1), criteria)
        return refined.reshape(-1, 2)
    except Exception:
        return points
