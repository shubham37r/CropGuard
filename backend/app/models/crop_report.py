import enum
from sqlalchemy import Column, Integer, String, Enum, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class GrowthStage(str, enum.Enum):
    SEEDLING = "Seedling"
    VEGETATIVE = "Vegetative"
    FLOWERING = "Flowering"
    FRUITING = "Fruiting"
    MATURITY = "Maturity"

class ReportStatus(str, enum.Enum):
    SUBMITTED = "SUBMITTED"
    ANALYZED = "ANALYZED"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    NEEDS_MORE_INFO = "NEEDS_MORE_INFO"

class CropReport(Base):
    __tablename__ = "crop_reports"

    id = Column(Integer, primary_key=True, index=True)
    farmer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    crop = Column(String, nullable=False)              # Tomato, Cotton, Soybean
    variety = Column(String, nullable=True)
    growth_stage = Column(Enum(GrowthStage), nullable=False)
    symptoms_description = Column(Text, nullable=True)
    image_url = Column(String, nullable=False)

    status = Column(Enum(ReportStatus), default=ReportStatus.SUBMITTED, nullable=False)
    
    location_id = Column(Integer, ForeignKey("location_reports.id"), nullable=False)
    analysis_id = Column(Integer, ForeignKey("analysis_results.id"), nullable=True)
    risk_id = Column(Integer, ForeignKey("risk_assessments.id"), nullable=True)
    verification_id = Column(Integer, ForeignKey("verifications.id"), nullable=True)

    advisory = Column(JSON, nullable=True)             # IPM Guidance structured output
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    farmer = relationship("User", foreign_keys=[farmer_id])
    location = relationship("LocationReport", foreign_keys=[location_id], cascade="all, delete")
    analysis = relationship("AnalysisResult", foreign_keys=[analysis_id], cascade="all, delete")
    risk_assessment = relationship("RiskAssessment", foreign_keys=[risk_id], cascade="all, delete")
    verification = relationship("Verification", foreign_keys=[verification_id], cascade="all, delete")
