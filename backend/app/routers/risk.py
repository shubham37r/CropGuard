from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.services.risk_engine import risk_engine
from app.schemas.risk import RiskAssessmentOut

router = APIRouter(prefix="/risk", tags=["risk"])

class RiskCalculationRequest(BaseModel):
    disease_confidence: float
    weather_risk: float
    crop_stage: str
    outbreak_signal: Optional[float] = 50.0
    condition_name: Optional[str] = "Suspected Disease"

@router.post("/calculate", response_model=RiskAssessmentOut)
def calculate_risk_endpoint(req: RiskCalculationRequest):
    res = risk_engine.calculate_risk(
        disease_confidence=req.disease_confidence,
        weather_risk=req.weather_risk,
        crop_stage=req.crop_stage,
        outbreak_signal=req.outbreak_signal,
        condition_name=req.condition_name
    )
    return res
