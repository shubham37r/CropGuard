from pydantic import BaseModel
from typing import List, Dict, Optional

class ComponentScores(BaseModel):
    disease_confidence: float
    weather_risk: float
    stage_risk: float
    outbreak_signal: float

class RiskAssessmentOut(BaseModel):
    id: Optional[int] = None
    score: float
    risk_level: str  # LOW, MEDIUM, HIGH
    component_scores: ComponentScores
    contributing_factors: List[str]
    methodology_note: str

    class Config:
        from_attributes = True
