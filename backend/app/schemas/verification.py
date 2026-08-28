from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class VerificationUpdate(BaseModel):
    status: str  # CONFIRMED, REJECTED, NEEDS_MORE_INFO
    officer_notes: Optional[str] = None

class VerificationOut(BaseModel):
    id: int
    status: str
    officer_id: Optional[int] = None
    officer_name: Optional[str] = None
    officer_notes: Optional[str] = None
    verified_at: Optional[datetime] = None

    class Config:
        from_attributes = True
