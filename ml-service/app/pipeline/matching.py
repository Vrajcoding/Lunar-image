import cv2
import numpy as np
from collections import defaultdict

def match_descriptors(desc_a, desc_b, ratio_thresh: float = 0.75):
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return []
    
    # Check descriptor type
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

def enforce_uniform_distribution(matches, kp_a, img_shape, grid_size: int = 8, max_per_cell: int = 10):
    if not matches:
        return [], 0.0
        
    h, w = img_shape[:2]
    cell_h, cell_w = max(h / grid_size, 1.0), max(w / grid_size, 1.0)
    cells = defaultdict(list)
    
    for m in matches:
        x, y = kp_a[m.queryIdx].pt
        cell_id = (int(y // cell_h), int(x // cell_w))
        cells[cell_id].append(m)

    filtered = []
    for cell_matches in cells.values():
        cell_matches.sort(key=lambda m: m.distance)
        filtered.extend(cell_matches[:max_per_cell])

    covered_cells = len(cells)
    total_cells = grid_size * grid_size
    uniformity_score = covered_cells / max(total_cells, 1)
    return filtered, uniformity_score
