from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from datetime import datetime
from app.database import Base

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    condition_name = Column(String, nullable=False)
    condition_type = Column(String, nullable=False)  # DISEASE or PEST
    confidence = Column(Float, nullable=False)        # 0 - 100 %
    alternatives = Column(JSON, nullable=True)        # List of {"name", "type", "confidence"}
    is_mock = Column(String, default="PROTOTYPE_MOCK")
    created_at = Column(DateTime, default=datetime.utcnow)
