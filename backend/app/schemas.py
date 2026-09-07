"""
schemas.py — Pydantic schemas for Lunar Registration backend API.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel


class RegisterAccepted(BaseModel):
    job_id: str
    status: str


class Metrics(BaseModel):
    rmse_px: float
    fit_rmse_px: Optional[float] = None
    test_rmse_px: Optional[float] = None
    inlier_count: int
    total_matches: Optional[int] = None
    match_count: Optional[int] = None
    inlier_ratio: float
    uniformity_score: float
    spatial_entropy: Optional[float] = None
    runtime_sec: float
    confidence_score: Optional[float] = None
    confidence_level: Optional[str] = None
    method: Optional[str] = None


class ResultResponse(BaseModel):
    job_id: str
    status: str
    method: Optional[str] = None
    registered_image_url: Optional[str] = None
    difference_image_url: Optional[str] = None
    match_points_url: Optional[str] = None
    metrics: Optional[Metrics] = None
    input_metadata: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    error: Optional[str] = None
