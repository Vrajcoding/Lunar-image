from pydantic import BaseModel

class RegistrationMetrics(BaseModel):
    rmse_px: float
    inlier_count: int
    inlier_ratio: float
    uniformity_score: float
    runtime_sec: float

class RegistrationResponse(BaseModel):
    status: str
    job_id: str
    registered_image_path: str
    match_points_path: str
    metrics: RegistrationMetrics
