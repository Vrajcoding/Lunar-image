"""
confidence.py — Scientifically explainable registration confidence estimation.
Combines independent RMSE, inlier ratio, spatial uniformity, and correspondence volume
into a calibrated quality score and categorical grade (HIGH / MEDIUM / LOW).
"""
import numpy as np


def compute_confidence_score(
    test_rmse: float,
    inlier_ratio: float,
    uniformity_score: float,
    inlier_count: int,
) -> tuple[float, str]:
    """Compute registration confidence metric in [0.0, 1.0] and categorical grade.
    
    Scoring components:
        1. Accuracy factor (40%): exponential penalty on RMSE (ideal < 1.0 px)
        2. Inlier purity (30%): RANSAC inlier ratio (ideal > 0.70)
        3. Spatial coverage (20%): Grid uniformity score (ideal > 0.70)
        4. Support volume (10%): Log-saturating correspondence count (ideal > 100)
        
    Returns:
        (confidence_score, confidence_level: "HIGH" | "MEDIUM" | "LOW")
    """
    if inlier_count < 4 or test_rmse > 50.0:
        return 0.0, "LOW"

    # Accuracy factor: 1.0 at 0.0 px RMSE, ~0.6 at 1.0 px, ~0.35 at 2.0 px
    rmse_factor = float(np.exp(-0.5 * max(test_rmse, 0.0)))
    
    # Inlier ratio factor: linear in [0, 1]
    ratio_factor = float(np.clip(inlier_ratio, 0.0, 1.0))
    
    # Uniformity factor: linear in [0, 1]
    uniformity_factor = float(np.clip(uniformity_score, 0.0, 1.0))
    
    # Volume factor: saturates around 100 inliers
    volume_factor = float(np.clip(inlier_count / 100.0, 0.0, 1.0))

    # Weighted composite score
    score = (
        0.40 * rmse_factor +
        0.30 * ratio_factor +
        0.20 * uniformity_factor +
        0.10 * volume_factor
    )
    score = float(np.clip(score, 0.0, 1.0))

    # Determine classification grade
    if score >= 0.65 and test_rmse <= 1.5 and inlier_count >= 15:
        level = "HIGH"
    elif score >= 0.30 and inlier_count >= 6 and test_rmse <= 6.0:
        level = "MEDIUM"
    else:
        level = "LOW"

    return round(score, 4), level
