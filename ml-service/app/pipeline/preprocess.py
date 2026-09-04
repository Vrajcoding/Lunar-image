import cv2
import numpy as np

from app.pipeline.loader import load_image

def load_grayscale(path: str) -> np.ndarray:
    img, _metadata = load_image(path)
    return img

def normalize_illumination(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(img, h=7)

def build_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(pyramid[-1])
        pyramid.append(img)
    return pyramid

def preprocess(path: str, levels: int = 4) -> list[np.ndarray]:
    img = load_grayscale(path)
    img_norm = normalize_illumination(img)
    img_denoised = denoise(img_norm)
    return build_pyramid(img_denoised, levels)
