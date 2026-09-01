import numpy as np

def compute_rmse(src_pts: np.ndarray, dst_pts: np.ndarray, H: np.ndarray) -> float:
    if len(src_pts) == 0:
        return 0.0
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    projected = (H @ src_h.T).T
    denom = projected[:, 2:3]
    denom[denom == 0] = 1e-6
    projected = projected[:, :2] / denom
    errors = np.linalg.norm(projected - dst_pts, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))

def build_metrics(inlier_count: int, total_matches: int, rmse: float, uniformity_score: float, runtime_sec: float):
    return {
        "rmse_px": round(rmse, 4),
        "inlier_count": int(inlier_count),
        "inlier_ratio": round(inlier_count / max(total_matches, 1), 4),
        "uniformity_score": round(uniformity_score, 4),
        "runtime_sec": round(runtime_sec, 3),
    }
