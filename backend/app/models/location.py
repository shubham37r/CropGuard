from sqlalchemy import Column, Integer, String, Float
from app.database import Base

class LocationReport(Base):
    __tablename__ = "location_reports"

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    district = Column(String, nullable=False, default="Nagpur")
    address = Column(String, nullable=True)
    region = Column(String, nullable=True)
