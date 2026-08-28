from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.crop_report import CropReport
from app.services.hotspot_service import hotspot_service
from app.schemas.hotspot import HotspotResponse

router = APIRouter(prefix="/hotspots", tags=["hotspots"])

@router.get("/", response_model=HotspotResponse)
def get_geospatial_hotspots(db: Session = Depends(get_db)):
    reports = db.query(CropReport).all()
    return hotspot_service.compute_hotspots(reports)
