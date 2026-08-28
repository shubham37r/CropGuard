from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User, UserRole
from app.schemas.user import UserOut
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

class LoginRequest(BaseModel):
    email: str

@router.post("/login", response_model=UserOut)
def mock_login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email.strip().lower()).first()
    if not user:
        # Default fallback creation if missing
        if "officer" in req.email.lower():
            user = User(email=req.email.lower(), name="Dr. Anish Sharma", role=UserRole.EXTENSION_OFFICER, region="Nagpur Zone")
        else:
            user = User(email=req.email.lower(), name="Rajesh Patel", role=UserRole.FARMER, region="Nagpur District")
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

@router.get("/users", response_model=list[UserOut])
def get_demo_users(db: Session = Depends(get_db)):
    return db.query(User).all()
