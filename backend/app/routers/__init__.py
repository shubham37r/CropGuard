from app.routers.auth import router as auth_router
from app.routers.reports import router as reports_router
from app.routers.risk import router as risk_router
from app.routers.verification import router as verification_router
from app.routers.hotspots import router as hotspots_router

__all__ = [
    "auth_router",
    "reports_router",
    "risk_router",
    "verification_router",
    "hotspots_router",
]
