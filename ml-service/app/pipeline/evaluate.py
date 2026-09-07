"""
evaluate.py — Registration evaluation metrics including fit RMSE, independent test RMSE,
inlier ratio, spatial entropy, and runtime metrics.
"""
import numpy as np


def compute_rmse(src_pts: np.ndarray, dst_pts: np.ndarray, H: np.ndarray) -> float:
    """Compute Root Mean Square Error (RMSE) in pixels between transformed src and dst points."""
    if len(src_pts) == 0 or len(dst_pts) == 0:
        return 0.0

    src_pts = np.asarray(src_pts, dtype=np.float64).reshape(-1, 2)
    dst_pts = np.asarray(dst_pts, dtype=np.float64).reshape(-1, 2)

    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1), dtype=np.float64)])
    projected = (H @ src_h.T).T
    denom = projected[:, 2:3]
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    projected_xy = projected[:, :2] / denom
    
    errors = np.linalg.norm(projected_xy - dst_pts, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    return float(np.clip(rmse, 0.0, 1000.0))


def compute_train_test_rmse(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    H: np.ndarray,
    test_split: float = 0.20,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Compute both Fit RMSE (on 80% train partition) and Independent Test RMSE (on 20% held-out test partition)."""
    n = len(src_pts)
    if n < 4:
        fit = compute_rmse(src_pts, dst_pts, H)
        return fit, fit

    if n < 8:
        fit = compute_rmse(src_pts, dst_pts, H)
        return fit, fit

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(n)
    n_test = max(int(round(n * test_split)), 1)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    fit_rmse = compute_rmse(src_pts[train_idx], dst_pts[train_idx], H)
    test_rmse = compute_rmse(src_pts[test_idx], dst_pts[test_idx], H)

    return round(fit_rmse, 4), round(test_rmse, 4)


def build_metrics(
    inlier_count: int,
    total_matches: int,
    rmse: float,
    uniformity_score: float,
    runtime_sec: float,
    test_rmse: float | None = None,
    spatial_entropy: float | None = None,
    confidence_score: float | None = None,
    confidence_level: str | None = None,
    method: str | None = None,
) -> dict:
    """Build registration metrics dictionary. Preserves legacy 5-key format when called without optional extras."""
    inlier_ratio = float(inlier_count / max(total_matches, 1))
    fit_rmse = float(rmse)

    res = {
        "rmse_px": round(fit_rmse, 4),
        "inlier_count": int(inlier_count),
        "inlier_ratio": round(inlier_ratio, 4),
        "uniformity_score": round(float(uniformity_score), 4),
        "runtime_sec": round(float(runtime_sec), 3),
    }

    if test_rmse is not None:
        res["test_rmse_px"] = round(float(test_rmse), 4)
        res["fit_rmse_px"] = round(fit_rmse, 4)
        res["total_matches"] = int(total_matches)
        res["match_count"] = int(total_matches)

    if spatial_entropy is not None:
        res["spatial_entropy"] = round(float(spatial_entropy), 4)

    if confidence_score is not None:
        res["confidence_score"] = round(float(confidence_score), 4)

    if confidence_level is not None:
        res["confidence_level"] = str(confidence_level)

    if method is not None:
        res["method"] = str(method)

    return res
