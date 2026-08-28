# CropGuard — Localized Crop Health Early Warning System (SIH MVP)

**CropGuard** is a localized Crop Health Early Warning System and decision-support platform designed for early detection and management of crop diseases and pest infestations.

It combines crop image analysis, crop type, growth stage vulnerability, location, environmental weather suitability, and local spatial outbreak signals to evaluate crop health risk and deliver validated Integrated Pest Management (IPM) guidance.

---

## ⚠️ PROTOTYPE & PHASE 2A STATUS DISCLAIMER

> **Phase 2A Status**: Real classifier integration architecture prepared; model training/integration pending.
> **Important**: The current default classifier remains the Phase 1 mock classifier (`CLASSIFIER_MODE=mock`). No production disease-detection accuracy is claimed. Prototype risk scores and mock analysis results are not scientifically validated agricultural predictions and should not be used for real-world treatment decisions.

---

## 🏛️ System Architecture

```
CropGuard/
├── backend/
│   ├── app/
│   │   ├── models/           # Modular SQLAlchemy models (user, crop_report, analysis, risk_assessment, verification, location)
│   │   ├── schemas/          # Pydantic v2 validation schemas
│   │   ├── services/         # Business logic & ML boundary layer
│   │   │   ├── classifier_interface.py  # BaseClassifierService & ClassifierResult contract
│   │   │   ├── mock_classifier.py       # Mock classifier implementation
│   │   │   ├── real_classifier.py       # Real CV model interface (lazy loading, GPU/CPU detection, crop filtering)
│   │   │   ├── classifier_factory.py    # Factory supporting CLASSIFIER_MODE=mock|real
│   │   │   ├── image_preprocessor.py    # Dedicated 224x224 RGB image preprocessing & validation
│   │   │   ├── risk_engine.py           # Weighted Prototype Contextual Risk Score Engine (0-100)
│   │   │   ├── mock_weather.py          # Environmental humidity & temperature suitability score
│   │   │   ├── advisory_service.py      # IPM guidance (no arbitrary chemical pesticide recommendations)
│   │   │   └── hotspot_service.py       # 15km radius-based spatial clustering
│   │   ├── routers/          # REST API Endpoints (/api/auth, /api/reports, /api/risk, /api/verification, /api/hotspots)
│   │   ├── config.py         # Configurable weights, thresholds, CLASSIFIER_MODE, and MODEL_PATH settings
│   │   ├── database.py       # SQLite connection layer
│   │   ├── seed_data.py      # Pre-populated Nagpur region demo data
│   │   └── main.py           # FastAPI application entrypoint
│   ├── requirements.txt
│   ├── run.py
│   └── tests/                # Pytest unit & integration test suite (20 tests passing)
└── frontend/
    ├── src/
    │   ├── api/client.ts     # Axios REST client
    │   ├── components/
    │   │   ├── common/       # NoticeBanner, RiskBadge, StatusBadge
    │   │   ├── farmer/       # FarmerDashboard, CheckCropWizard, MyReports, ReportDetailModal
    │   │   ├── officer/      # OfficerDashboard, VerificationTable, VerificationModal, HotspotsMapView
    │   │   └── maps/         # LeafletMap (Picker & Hotspots cluster visualizer)
    │   ├── context/          # AuthContext for role switching
    │   ├── types/            # TypeScript interfaces
    │   ├── App.tsx
    │   └── main.tsx
    └── tailwind.config.js
```

---

## 🔑 Demo User Accounts

Switch roles seamlessly using the top-right header role toggle or login with these credentials:

| Role | Demo Email | Target User | Focus Area |
|---|---|---|---|
| **Farmer** | `farmer@example.com` | Rajesh Patel | Check Crop, Submit Images, View IPM Advice, Request Verification |
| **Extension Officer** | `officer@example.com` | Dr. Anish Sharma | Case Verification Queue, Confirm/Reject Diagnoses, Hotspot Map |

---

## 🚀 How to Run the Application

### 1. Run the Backend (Python FastAPI)

```bash
# Navigate to backend folder
cd backend

# Install dependencies
pip install -r requirements.txt

# Start FastAPI dev server (runs on http://localhost:8000)
python run.py
```
*Note: SQLite database (`cropguard.db`) is automatically initialized and seeded with Nagpur demo data on first startup.*

### 2. Run the Frontend (React + Vite + TypeScript)

```bash
# Navigate to frontend folder
cd frontend

# Install npm packages
npm install

# Start Vite dev server (runs on http://localhost:5173)
npm run dev
```

---

## 🔬 Phase 2A Computer-Vision Classifier Architecture

CropGuard features a decoupled classifier boundary that allows switching between `mock` and `real` ML models without altering API routes, risk engine calculations, or frontend components.

- **Configuration Settings** (`config.py` / `.env`):
  - `CLASSIFIER_MODE`: `"mock"` (default) or `"real"`
  - `MODEL_PATH`: `models/cropguard_disease_model.keras`
  - `MODEL_NAME`: `"CropGuard-EfficientNet"`
  - `MODEL_VERSION`: `"0.1.0"`

- **Pre-processing Specifications** (`image_preprocessor.py`):
  - Expected Image Size: `224 x 224` pixels (RGB)
  - Supported Formats: `JPEG`, `PNG`, `WEBP`
  - Validation: Strict magic bytes checking for image integrity.

- **Crop Validation**:
  - Model prediction is validated against the farmer's selected crop to prevent crop mismatch errors (e.g. predicting a Tomato disease on Cotton).

---

## 🔮 Future Integration Boundaries

1. **Model Training & Weights**: Train EfficientNet / MobileNetV3 model on agricultural disease datasets and output `cropguard_disease_model.keras` into `backend/models/`.
2. **Real Weather & Microclimate API**: Integration with OpenWeatherMap / Indian Meteorological Department (IMD) APIs via `BaseWeatherService`.
3. **Validated Agricultural IPM Knowledge Base**: Integration with ICAR / State Agriculture University advisory databases.
4. **Gemini AI**: Dedicated for natural language explanation, query handling, and multilingual farmer communication.
5. **IoT Sensors & Pest Trap Data**: Automated telemetry ingestion for pheromone trap counts.
6. **Satellite & Remote Sensing Data**: Sentinel/Landsat NDVI vegetation health imagery.
7. **Database Migration**: Seamless SQLAlchemy migration from SQLite to PostgreSQL / Supabase.
8. **Production Authentication**: OAuth2 / JWT authentication integration.
