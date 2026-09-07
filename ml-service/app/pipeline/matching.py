"""
matching.py — Feature matching and spatial distribution filtering.
Implements Lowe's ratio test, 8x8 spatial grid bucketing, and normalized spatial entropy
to avoid spatial correspondence clustering.
"""
import math
from collections import defaultdict
import cv2
import numpy as np


def match_descriptors(desc_a, desc_b, ratio_thresh: float = 0.75):
    """KNN match with Lowe's ratio test for classical descriptors."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return []

    if desc_a.dtype == np.uint8:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2)

    raw_matches = bf.knnMatch(desc_a, desc_b, k=2)
    good = []
    for m_tuple in raw_matches:
        if len(m_tuple) == 2:
            m, n = m_tuple
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
    return good


def compute_spatial_entropy(points: np.ndarray, img_shape: tuple[int, int], grid_size: int = 8) -> tuple[float, float, int]:
    """Compute normalized spatial entropy and cell coverage for point coordinates.
    
    Returns:
        uniformity_score (float in [0, 1]): H / ln(grid_size^2)
        entropy (float): Shannon entropy in nats
        covered_cells (int): Number of non-empty grid cells
    """
    total_points = len(points)
    total_cells = grid_size * grid_size
    if total_points == 0 or total_cells <= 1:
        return 0.0, 0.0, 0

    h, w = img_shape[:2]
    cell_h, cell_w = max(h / grid_size, 1e-6), max(w / grid_size, 1e-6)
    cell_counts = defaultdict(int)

    for pt in points:
        x, y = float(pt[0]), float(pt[1])
        r = min(int(y // cell_h), grid_size - 1)
        c = min(int(x // cell_w), grid_size - 1)
        cell_counts[(r, c)] += 1

    covered_cells = len(cell_counts)
    entropy = 0.0
    for count in cell_counts.values():
        p = count / float(total_points)
        if p > 0:
            entropy -= p * math.log(p)

    max_entropy = math.log(total_cells)
    uniformity_score = float(np.clip(entropy / max_entropy, 0.0, 1.0))
    return round(uniformity_score, 4), round(entropy, 4), covered_cells


def enforce_uniform_distribution(
    matches,
    kp_a,
    img_shape,
    grid_size: int = 8,
    max_per_cell: int = 10,
) -> tuple[list, float]:
    """Enforce spatial grid limit on cv2.DMatch list and calculate uniformity score."""
    if not matches:
        return [], 0.0

    h, w = img_shape[:2]
    cell_h, cell_w = max(h / grid_size, 1.0), max(w / grid_size, 1.0)
    cells = defaultdict(list)

    for m in matches:
        x, y = kp_a[m.queryIdx].pt
        cell_id = (min(int(y // cell_h), grid_size - 1), min(int(x // cell_w), grid_size - 1))
        cells[cell_id].append(m)

    filtered = []
    for cell_matches in cells.values():
        cell_matches.sort(key=lambda m: m.distance)
        filtered.extend(cell_matches[:max_per_cell])

    # Extract coordinates of filtered points to compute entropy-based uniformity
    filtered_pts = np.float32([kp_a[m.queryIdx].pt for m in filtered])
    uniformity_score, _, _ = compute_spatial_entropy(filtered_pts, img_shape, grid_size)
    return filtered, uniformity_score


def filter_dense_correspondences(
    pts0: np.ndarray,
    pts1: np.ndarray,
    confidence: np.ndarray,
    img_shape: tuple[int, int],
    grid_size: int = 8,
    max_per_cell: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Uniform grid filtering on arbitrary dense point arrays (e.g., from LoFTR).
    
    Returns:
        (filtered_pts0, filtered_pts1, filtered_confidence, uniformity_score, spatial_entropy)
    """
    if len(pts0) == 0:
        return pts0, pts1, confidence, 0.0, 0.0

    h, w = img_shape[:2]
    cell_h, cell_w = max(h / grid_size, 1.0), max(w / grid_size, 1.0)
    cells = defaultdict(list)

    for i in range(len(pts0)):
        x, y = pts0[i]
        cell_id = (min(int(y // cell_h), grid_size - 1), min(int(x // cell_w), grid_size - 1))
        cells[cell_id].append((confidence[i] if i < len(confidence) else 1.0, i))

    selected_indices = []
    for cell_entries in cells.values():
        # Sort by confidence descending
        cell_entries.sort(key=lambda e: e[0], reverse=True)
        selected_indices.extend([e[1] for e in cell_entries[:max_per_cell]])

    selected_indices = np.array(selected_indices, dtype=int)
    f_pts0 = pts0[selected_indices]
    f_pts1 = pts1[selected_indices]
    f_conf = confidence[selected_indices] if len(confidence) > 0 else np.ones(len(f_pts0), dtype=np.float32)

    uniformity_score, entropy, _ = compute_spatial_entropy(f_pts0, img_shape, grid_size)
    return f_pts0, f_pts1, f_conf, uniformity_score, entropy
