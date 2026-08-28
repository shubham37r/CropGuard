import enum
from sqlalchemy import Column, Integer, String, Enum, DateTime
from datetime import datetime
from app.database import Base

class UserRole(str, enum.Enum):
    FARMER = "FARMER"
    EXTENSION_OFFICER = "EXTENSION_OFFICER"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.FARMER)
    phone = Column(String, nullable=True)
    region = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
