import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.database import engine, Base, SessionLocal
from app.seed_data import seed_database
from app.routers import (
    auth_router,
    reports_router,
    risk_router,
    verification_router,
    hotspots_router
)

# 1. Create DB tables
Base.metadata.create_all(bind=engine)

# 2. Seed database with demo data on startup
db = SessionLocal()
try:
    seed_database(db)
finally:
    db.close()

# 3. Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Localized Crop Health Early Warning System and Decision-Support Platform MVP for SIH.",
    version="1.0.0"
)

# 4. Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Static file serving for image uploads
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# 6. Include API Routers
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(reports_router, prefix=settings.API_V1_STR)
app.include_router(risk_router, prefix=settings.API_V1_STR)
app.include_router(verification_router, prefix=settings.API_V1_STR)
app.include_router(hotspots_router, prefix=settings.API_V1_STR)

@app.get("/")
def root_status():
    return {
        "status": "online",
        "app": settings.PROJECT_NAME,
        "version": "1.0.0",
        "disclaimer": settings.METHODOLOGY_NOTE
    }
