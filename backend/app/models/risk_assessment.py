from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from datetime import datetime
from app.database import Base

class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, index=True)
    score = Column(Float, nullable=False)             # 0 - 100
    risk_level = Column(String, nullable=False)        # LOW, MEDIUM, HIGH
    component_scores = Column(JSON, nullable=False)   # {"disease_confidence", "weather_risk", "stage_risk", "outbreak_signal"}
    contributing_factors = Column(JSON, nullable=False) # List of strings
    methodology_note = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
