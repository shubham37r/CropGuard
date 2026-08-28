from pydantic import BaseModel
from typing import List, Optional

class HotspotPoint(BaseModel):
    report_id: int
    crop: str
    condition_name: str
    risk_level: str
    latitude: float
    longitude: float
    address: Optional[str] = None
    district: str
    status: str

class HotspotCluster(BaseModel):
    cluster_id: str
    center_latitude: float
    center_longitude: float
    radius_km: float
    high_risk_count: int
    total_reports_count: int
    dominant_crop: str
    dominant_condition: str
    district: str
    title: str
    description: str
    report_ids: List[int]

class HotspotResponse(BaseModel):
    points: List[HotspotPoint]
    clusters: List[HotspotCluster]
    methodology_note: str = "Simple radius-based prototype spatial clustering of high-risk report clusters."
