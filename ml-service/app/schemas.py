"""
schemas.py — Pydantic models for ML service endpoints.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel


class RegistrationMetrics(BaseModel):
    rmse_px: float
    fit_rmse_px: float
    test_rmse_px: float
    inlier_count: int
    total_matches: int
    match_count: int
    inlier_ratio: float
    uniformity_score: float
    spatial_entropy: float
    runtime_sec: float
    confidence_score: float
    confidence_level: str
    method: str


class RegistrationResponse(BaseModel):
    status: str
    job_id: str
    method: Optional[str] = None
    registered_image_path: Optional[str] = None
    difference_image_path: Optional[str] = None
    match_points_path: Optional[str] = None
    metrics: RegistrationMetrics
    input_metadata: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    error: Optional[str] = None
