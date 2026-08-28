from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime
from app.schemas.verification import VerificationOut

class LocationSchema(BaseModel):
    latitude: float
    longitude: float
    district: str = "Nagpur"
    address: Optional[str] = None
    region: Optional[str] = None

class ConditionItem(BaseModel):
    name: str
    type: str  # DISEASE or PEST
    confidence: Optional[float] = None

class AlternativeCondition(BaseModel):
    name: str
    type: str
    confidence: float

class AnalysisOut(BaseModel):
    id: Optional[int] = None
    condition: ConditionItem
    confidence: float
    alternatives: List[AlternativeCondition]
    is_mock: str = "PROTOTYPE_MOCK"

class IPMAdvisoryOut(BaseModel):
    priority: str
    actions: List[str]
    monitoring: List[str]
    expert_referral: bool
    safety_note: str

class CropReportCreate(BaseModel):
    farmer_id: int
    crop: str
    variety: Optional[str] = None
    growth_stage: str
    symptoms_description: Optional[str] = None
    image_url: str
    location: LocationSchema

class CropReportOut(BaseModel):
    id: int
    farmer_id: int
    farmer_name: Optional[str] = None
    crop: str
    variety: Optional[str] = None
    growth_stage: str
    symptoms_description: Optional[str] = None
    image_url: str
    status: str
    created_at: datetime
    location: LocationSchema
    analysis: Optional[AnalysisOut] = None
    risk_assessment: Optional[Any] = None
    verification: Optional[VerificationOut] = None
    advisory: Optional[IPMAdvisoryOut] = None

    class Config:
        from_attributes = True
