import os
import shutil
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from app.database import get_db
from app.config import settings
from app.models.crop_report import CropReport, ReportStatus, GrowthStage
from app.models.location import LocationReport
from app.models.analysis import AnalysisResult
from app.models.risk_assessment import RiskAssessment
from app.models.verification import Verification, VerificationStatus
from app.models.user import User
from app.schemas.report import CropReportCreate, CropReportOut, LocationSchema
from app.services.classifier_factory import get_classifier_service
from app.services.mock_weather import mock_weather
from app.services.risk_engine import risk_engine
from app.services.advisory_service import advisory_service

router = APIRouter(prefix="/reports", tags=["reports"])

def format_report_response(report: CropReport) -> dict:
    loc = report.location
    ana = report.analysis
    risk = report.risk_assessment
    verif = report.verification
    
    cond_dict = None
    if ana:
        cond_dict = {
            "name": ana.condition_name,
            "type": ana.condition_type
        }

    verif_dict = None
    if verif:
        verif_dict = {
            "id": verif.id,
            "status": verif.status.value if hasattr(verif.status, "value") else str(verif.status),
            "officer_id": verif.officer_id,
            "officer_name": verif.officer.name if verif.officer else None,
            "officer_notes": verif.officer_notes,
            "verified_at": verif.verified_at
        }

    return {
        "id": report.id,
        "farmer_id": report.farmer_id,
        "farmer_name": report.farmer.name if report.farmer else "Farmer",
        "crop": report.crop,
        "variety": report.variety,
        "growth_stage": report.growth_stage.value if hasattr(report.growth_stage, "value") else str(report.growth_stage),
        "symptoms_description": report.symptoms_description,
        "image_url": report.image_url,
        "status": report.status.value if hasattr(report.status, "value") else str(report.status),
        "created_at": report.created_at,
        "location": {
            "latitude": loc.latitude if loc else 21.1458,
            "longitude": loc.longitude if loc else 79.0882,
            "district": loc.district if loc else "Nagpur",
            "address": loc.address if loc else None,
            "region": loc.region if loc else None
        },
        "analysis": {
            "id": ana.id if ana else None,
            "condition": cond_dict,
            "confidence": ana.confidence if ana else 85.0,
            "alternatives": ana.alternatives if ana else [],
            "is_mock": ana.is_mock if ana else "PROTOTYPE_MOCK"
        } if ana else None,
        "risk_assessment": {
            "id": risk.id if risk else None,
            "score": risk.score if risk else 50.0,
            "risk_level": risk.risk_level if risk else "MEDIUM",
            "component_scores": risk.component_scores if risk else {},
            "contributing_factors": risk.contributing_factors if risk else [],
            "methodology_note": risk.methodology_note if risk else settings.METHODOLOGY_NOTE
        } if risk else None,
        "verification": verif_dict,
        "advisory": report.advisory
    }

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Return served URL path
    return {"image_url": f"/uploads/{filename}"}

@router.post("/", response_model=CropReportOut)
def create_report(payload: CropReportCreate, db: Session = Depends(get_db)):
    # 1. Create Location
    loc = LocationReport(
        latitude=payload.location.latitude,
        longitude=payload.location.longitude,
        district=payload.location.district or "Nagpur",
        address=payload.location.address or "Nagpur District",
        region=payload.location.region or "Maharashtra"
    )
    db.add(loc)
    db.flush()

    # 2. Trigger Active Classifier Service (Mock or Real based on CLASSIFIER_MODE)
    classifier = get_classifier_service()

    # Resolve uploaded image bytes for real CV inference.
    # image_url is stored as "/uploads/<filename>"; resolve to absolute filesystem path.
    image_bytes: Optional[bytes] = None
    if payload.image_url:
        img_filename = os.path.basename(payload.image_url)
        img_path = os.path.join(settings.UPLOAD_DIR, img_filename)

        if os.path.exists(img_path):
            try:
                with open(img_path, "rb") as img_f:
                    image_bytes = img_f.read()
            except Exception:
                image_bytes = None  # Classifier will handle gracefully

    # classify_crop_image only uses symptoms for mock; real CV needs image_bytes
    # Call predict() directly so image_bytes can be forwarded
    clf_result = classifier.predict(
        image_bytes=image_bytes,
        crop=payload.crop,
        symptoms=payload.symptoms_description
    )
    clf_res = clf_result.to_legacy_dict()
    cond_name = clf_res["condition"]["name"]
    cond_type = clf_res["condition"]["type"]
    confidence = clf_res["confidence"]
    alternatives = clf_res["alternatives"]
    is_mock = clf_res.get("is_mock", "PROTOTYPE_MOCK")


    ana = AnalysisResult(
        condition_name=cond_name,
        condition_type=cond_type,
        confidence=confidence,
        alternatives=alternatives,
        is_mock=is_mock
    )
    db.add(ana)
    db.flush()

    # 3. Trigger Environmental / Weather Service
    weather_res = mock_weather.get_environmental_risk(loc.latitude, loc.longitude, payload.crop)
    weather_score = weather_res["environmental_risk_score"]

    # 4. Trigger Risk Engine Service
    risk_res = risk_engine.calculate_risk(
        disease_confidence=confidence,
        weather_risk=weather_score,
        crop_stage=payload.growth_stage,
        outbreak_signal=65.0,
        condition_name=cond_name
    )

    risk_obj = RiskAssessment(
        score=risk_res["score"],
        risk_level=risk_res["risk_level"],
        component_scores=risk_res["component_scores"],
        contributing_factors=risk_res["contributing_factors"],
        methodology_note=risk_res["methodology_note"]
    )
    db.add(risk_obj)
    db.flush()

    # 5. Trigger Dedicated Advisory Service
    advisory_res = advisory_service.generate_advisory(
        crop=payload.crop,
        condition=cond_name,
        condition_type=cond_type,
        risk_level=risk_res["risk_level"],
        crop_stage=payload.growth_stage,
        confidence=confidence
    )

    # 6. Set initial Case Lifecycle Status
    initial_status = ReportStatus.ANALYZED
    verif = Verification(status=VerificationStatus.PENDING_VERIFICATION)
    db.add(verif)
    db.flush()

    if confidence < settings.CONFIDENCE_VERIFICATION_THRESHOLD:
        initial_status = ReportStatus.PENDING_VERIFICATION

    # 7. Create CropReport
    report = CropReport(
        farmer_id=payload.farmer_id,
        crop=payload.crop,
        variety=payload.variety,
        growth_stage=payload.growth_stage,
        symptoms_description=payload.symptoms_description,
        image_url=payload.image_url,
        status=initial_status,
        location_id=loc.id,
        analysis_id=ana.id,
        risk_id=risk_obj.id,
        verification_id=verif.id,
        advisory=advisory_res
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    return format_report_response(report)

@router.get("/", response_model=List[CropReportOut])
def get_reports(
    farmer_id: Optional[int] = None,
    status: Optional[str] = None,
    crop: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(CropReport)
    if farmer_id:
        query = query.filter(CropReport.farmer_id == farmer_id)
    if status:
        query = query.filter(CropReport.status == status)
    if crop:
        query = query.filter(CropReport.crop == crop)
    
    reports = query.order_by(CropReport.created_at.desc()).all()
    return [format_report_response(r) for r in reports]

@router.get("/{report_id}", response_model=CropReportOut)
def get_report_detail(report_id: int, db: Session = Depends(get_db)):
    report = db.query(CropReport).filter(CropReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Crop report not found")
    return format_report_response(report)

@router.post("/{report_id}/refer-expert", response_model=CropReportOut)
def refer_to_expert(report_id: int, db: Session = Depends(get_db)):
    report = db.query(CropReport).filter(CropReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Crop report not found")
    
    report.status = ReportStatus.PENDING_VERIFICATION
    if report.verification:
        report.verification.status = VerificationStatus.PENDING_VERIFICATION
    
    db.commit()
    db.refresh(report)
    return format_report_response(report)
