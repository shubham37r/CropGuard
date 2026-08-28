from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.crop_report import CropReport, ReportStatus
from app.models.verification import Verification, VerificationStatus
from app.schemas.verification import VerificationUpdate
from app.schemas.report import CropReportOut
from app.routers.reports import format_report_response

router = APIRouter(prefix="/verification", tags=["verification"])

@router.post("/{report_id}", response_model=CropReportOut)
def verify_report(
    report_id: int,
    officer_id: int,
    payload: VerificationUpdate,
    db: Session = Depends(get_db)
):
    report = db.query(CropReport).filter(CropReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Crop report not found")

    new_status = payload.status.upper()
    if new_status not in [VerificationStatus.CONFIRMED, VerificationStatus.REJECTED, VerificationStatus.NEEDS_MORE_INFO]:
        raise HTTPException(status_code=400, detail="Invalid verification status")

    verif = report.verification
    if not verif:
        verif = Verification(report_id=report.id)
        db.add(verif)
        db.flush()

    verif.status = VerificationStatus(new_status)
    verif.officer_id = officer_id
    verif.officer_notes = payload.officer_notes
    verif.verified_at = datetime.utcnow()

    # Sync report status with lifecycle
    report.status = ReportStatus(new_status)

    db.commit()
    db.refresh(report)

    return format_report_response(report)
