import cv2
import numpy as np

def detect_and_describe(img: np.ndarray, n_features: int = 5000):
    try:
        sift = cv2.SIFT_create(nfeatures=n_features)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if keypoints and descriptors is not None:
            return keypoints, descriptors
    except Exception:
        pass
    
    # Fallback to ORB if SIFT fails or finds 0 points
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors
