from pydantic import BaseModel
from typing import Optional

class RegisterAccepted(BaseModel):
    job_id: str
    status: str

class Metrics(BaseModel):
    rmse_px: float
    inlier_count: int
    inlier_ratio: float
    uniformity_score: float
    runtime_sec: float

class ResultResponse(BaseModel):
    job_id: str
    status: str
    registered_image_url: Optional[str] = None
    match_points_url: Optional[str] = None
    metrics: Optional[Metrics] = None
    error: Optional[str] = None
