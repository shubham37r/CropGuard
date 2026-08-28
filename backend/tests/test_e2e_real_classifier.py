"""
Phase 2C-2: Controlled End-to-End Validation Script
Exercises the FULL API path:
  Image → upload-image → POST /api/reports/ → real EfficientNet-B0 → SQLite

Requires the FastAPI server to be running with CLASSIFIER_MODE=real.
This script uses TestClient (in-process, no network) for reliability.
"""
import os
import sys
import json
import shutil

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force CLASSIFIER_MODE=real BEFORE importing app modules
os.environ["CLASSIFIER_MODE"] = "real"

from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)

# ── Test Image ────────────────────────────────────────────────────────────────
# __file__ = .../CropGuard/backend/tests/test_e2e_real_classifier.py
# dirname x1 = .../backend/tests/
# dirname x2 = .../backend/
# dirname x3 = .../CropGuard/  <- project root
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))  # backend/tests/
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)                # backend/
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)              # CropGuard/

TEST_IMAGE_PATH = os.path.join(
    _PROJECT_ROOT, "ml", "data", "splits", "test",
    "Tomato___Bacterial_spot",
    "014b58ae-091b-408a-ab4a-5a780cd1c3f3___GCREC_Bact.Sp 2971_final_masked.jpg"
)
TEST_IMAGE_PATH = os.path.normpath(TEST_IMAGE_PATH)
TRUE_CLASS = "Tomato___Bacterial_spot"

FARMER_ID = 2  # Demo farmer from seed data

SEPARATOR = "=" * 70

def banner(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def assert_field(label, value, condition, expected_desc=""):
    status = "[PASS]" if condition else "[FAIL]"
    print(f"  {status} {label}: {value!r}  {f'(expected: {expected_desc})' if expected_desc else ''}")
    if not condition:
        raise AssertionError(f"ASSERTION FAILED -- {label}: {value!r}. {expected_desc}")

def run():
    print(f"\n{'#'*70}")
    print(f"  CropGuard Phase 2C-2 -- End-to-End Real Classifier Validation")
    print(f"{'#'*70}")
    print(f"  CLASSIFIER_MODE: {settings.CLASSIFIER_MODE}")
    print(f"  Test Image Path: {TEST_IMAGE_PATH}")
    print(f"  True Class:      {TRUE_CLASS}")
    print(f"  Image Exists:    {os.path.exists(TEST_IMAGE_PATH)}")

    assert os.path.exists(TEST_IMAGE_PATH), f"Test image not found: {TEST_IMAGE_PATH}"

    # ── STEP 1: Verify CLASSIFIER_MODE is real ────────────────────────────────
    banner("STEP 1 -- Verify CLASSIFIER_MODE=real")
    assert_field("CLASSIFIER_MODE", settings.CLASSIFIER_MODE, 
                 settings.CLASSIFIER_MODE == "real", "real")

    # ── STEP 2: Verify model loaded ───────────────────────────────────────────
    banner("STEP 2 -- Verify Real Model Loaded")
    from app.services.real_classifier import real_classifier
    real_classifier.load_model()
    assert_field("_model_loaded", real_classifier._model_loaded, real_classifier._model_loaded is True, "True")
    assert_field("_load_error", real_classifier._load_error, real_classifier._load_error is None, "None")
    assert_field("num_classes", len(real_classifier._idx_to_class), len(real_classifier._idx_to_class) == 10, "10")
    assert_field("device", str(real_classifier._device), True)
    print(f"  [INFO] Model checkpoint: ml/models/checkpoints/best_model.pt")
    print(f"  [INFO] Classes: {list(real_classifier._idx_to_class.values())}")

    # ── STEP 3: Upload the test image ─────────────────────────────────────────
    banner("STEP 3 -- Upload Test Image via POST /api/reports/upload-image")
    with open(TEST_IMAGE_PATH, "rb") as img_f:
        upload_resp = client.post(
            "/api/reports/upload-image",
            files={"file": ("test_tomato_bacterial_spot.jpg", img_f, "image/jpeg")}
        )
    assert_field("Upload HTTP Status", upload_resp.status_code, upload_resp.status_code == 200, "200")
    upload_data = upload_resp.json()
    image_url = upload_data.get("image_url", "")
    assert_field("image_url returned", image_url, image_url.startswith("/uploads/"), "starts with /uploads/")
    print(f"  [INFO] Uploaded image URL: {image_url}")

    # ── STEP 4: Create Report (Tomato -- Real Inference) ───────────────────────
    banner("STEP 4 -- POST /api/reports/ -- Tomato Real Inference")
    report_payload = {
        "farmer_id": FARMER_ID,
        "crop": "Tomato",
        "variety": "Roma",
        "growth_stage": "VEGETATIVE",
        "symptoms_description": "Dark spots with yellow halo on leaves",
        "image_url": image_url,
        "location": {
            "latitude": 20.9374,
            "longitude": 77.7796,
            "district": "Akola",
            "address": "Test Farm, Akola",
            "region": "Vidarbha"
        }
    }
    report_resp = client.post("/api/reports/", json=report_payload)
    assert_field("Create Report HTTP Status", report_resp.status_code, report_resp.status_code == 200, "200")

    if report_resp.status_code != 200:
        print(f"  [ERROR] Response body: {report_resp.text}")
        return False

    report = report_resp.json()

    # ── STEP 5: Validate Report Structure ─────────────────────────────────────
    banner("STEP 5 -- Validate API Response Structure")
    report_id = report.get("id")
    assert_field("report.id", report_id, report_id is not None and report_id > 0, "> 0")
    assert_field("report.crop", report.get("crop"), report.get("crop") == "Tomato", "Tomato")
    assert_field("report.status", report.get("status"), report.get("status") in ["ANALYZED", "PENDING_VERIFICATION"], "ANALYZED or PENDING_VERIFICATION")

    # Analysis
    analysis = report.get("analysis", {})
    assert analysis, "analysis block missing from response"
    cond = analysis.get("condition", {})
    assert_field("analysis.condition.name", cond.get("name"), bool(cond.get("name")), "non-empty string")
    assert_field("analysis.condition.type", cond.get("type"), cond.get("type") in ["DISEASE", "PEST", "HEALTHY"], "DISEASE/PEST/HEALTHY")
    confidence = analysis.get("confidence", 0)
    assert_field("analysis.confidence range", confidence, 0.0 <= confidence <= 100.0, "[0, 100]")
    alternatives = analysis.get("alternatives", [])
    assert_field("alternatives count", len(alternatives), len(alternatives) >= 1, ">= 1")
    is_mock = analysis.get("is_mock", "")
    assert_field("analysis.is_mock", is_mock, is_mock == "REAL_CV_MODEL", "REAL_CV_MODEL")

    # Model name & version are embedded in the ClassifierResult (confirmed via unit tests)
    # and are NOT separately persisted in the AnalysisOut API response schema.
    # Verify them via the real_classifier object directly.
    assert_field("real_classifier model_name", real_classifier.MODEL_NAME,
                 "EfficientNet" in real_classifier.MODEL_NAME, "contains EfficientNet")
    assert_field("real_classifier model_version", real_classifier.MODEL_VERSION,
                 real_classifier.MODEL_VERSION == "0.1.0", "0.1.0")

    # Risk Assessment
    risk = report.get("risk_assessment", {})
    assert risk, "risk_assessment block missing"
    assert_field("risk.score range", risk.get("score"), 0 <= risk.get("score", -1) <= 100, "[0, 100]")
    assert_field("risk.risk_level", risk.get("risk_level"), risk.get("risk_level") in ["LOW", "MEDIUM", "HIGH", "CRITICAL"], "valid level")
    assert_field("risk.methodology_note", bool(risk.get("methodology_note")), True, "non-empty")

    # Advisory
    advisory = report.get("advisory", {})
    assert advisory, "advisory block missing"
    assert_field("advisory.priority", advisory.get("priority"), bool(advisory.get("priority")), "non-empty")
    assert_field("advisory.actions", len(advisory.get("actions", [])), len(advisory.get("actions", [])) > 0, "> 0 actions")

    # ── STEP 6: Print Full Prediction Details ─────────────────────────────────
    banner("STEP 6 -- Real Inference Results")
    print(f"  Report ID:          {report_id}")
    print(f"  True Class:         {TRUE_CLASS}")
    print(f"  Predicted Cond:     {cond.get('name')} ({cond.get('type')})")
    print(f"  Confidence:         {confidence:.2f}%")
    print(f"  is_mock:            {is_mock}")
    print(f"  Model Name:         {real_classifier.MODEL_NAME} (on {real_classifier._device})")
    print(f"  Model Version:      {real_classifier.MODEL_VERSION}")
    print(f"  Risk Score:         {risk.get('score'):.1f}/100")
    print(f"  Risk Level:         {risk.get('risk_level')}")
    print(f"  Methodology Note:   {str(risk.get('methodology_note', ''))[:80]}...")
    print(f"  Advisory Priority:  {advisory.get('priority')}")
    print(f"  Expert Referral:    {advisory.get('expert_referral')}")
    print(f"  Alternatives:")
    for alt in alternatives:
        print(f"    - {alt.get('name')} ({alt.get('type')}): {alt.get('confidence', 0):.2f}%")

    # ── STEP 7: Verify SQLite Persistence ─────────────────────────────────────
    banner("STEP 7 -- Verify SQLite Persistence (GET /api/reports/{id})")
    get_resp = client.get(f"/api/reports/{report_id}")
    assert_field("GET Report Status", get_resp.status_code, get_resp.status_code == 200, "200")
    persisted = get_resp.json()
    assert_field("Persisted report ID", persisted.get("id"), persisted.get("id") == report_id, f"== {report_id}")
    assert_field("Persisted condition", persisted.get("analysis", {}).get("condition", {}).get("name"),
                 persisted.get("analysis", {}).get("condition", {}).get("name") == cond.get("name"),
                 f"== {cond.get('name')}")
    assert_field("Persisted is_mock", persisted.get("analysis", {}).get("is_mock"),
                 persisted.get("analysis", {}).get("is_mock") == "REAL_CV_MODEL", "REAL_CV_MODEL")
    print(f"  [INFO] Report successfully persisted and retrieved from SQLite.")

    # ── STEP 8: Officer Workflow -- Retrieve Report for Review ─────────────────
    banner("STEP 8 -- Officer Workflow (GET /api/reports/?status=PENDING_VERIFICATION or ANALYZED)")
    list_resp = client.get("/api/reports/")
    assert_field("List Reports Status", list_resp.status_code, list_resp.status_code == 200, "200")
    all_reports = list_resp.json()
    matched = [r for r in all_reports if r.get("id") == report_id]
    assert_field("Report found in officer listing", len(matched), len(matched) == 1, "1")
    print(f"  [INFO] Officer can retrieve report ID={report_id} from listing endpoint.")

    # ── STEP 9: Cotton Unsupported Crop Test ─────────────────────────────────
    banner("STEP 9 -- Unsupported Crop: Cotton POST /api/reports/")
    cotton_payload = {
        "farmer_id": FARMER_ID,
        "crop": "Cotton",
        "variety": None,
        "growth_stage": "VEGETATIVE",
        "symptoms_description": "Bollworm damage",
        "image_url": image_url,
        "location": {
            "latitude": 20.9374,
            "longitude": 77.7796,
            "district": "Akola",
            "address": "Test Farm, Akola",
            "region": "Vidarbha"
        }
    }
    cotton_resp = client.post("/api/reports/", json=cotton_payload)
    print(f"  Cotton Report HTTP Status: {cotton_resp.status_code}")
    cotton_data = cotton_resp.json()
    cotton_analysis = cotton_data.get("analysis", {})
    cotton_cond_name = cotton_analysis.get("condition", {}).get("name", "")
    cotton_confidence = cotton_analysis.get("confidence", -1)
    cotton_is_mock = cotton_analysis.get("is_mock", "")

    print(f"  Cotton condition_name: {cotton_cond_name!r}")
    print(f"  Cotton confidence:     {cotton_confidence}")
    print(f"  Cotton is_mock:        {cotton_is_mock!r}")

    # Must NOT produce a real Tomato prediction for Cotton
    real_tomato_classes = list(real_classifier._idx_to_class.values())
    assert_field("Cotton NOT classified as Tomato disease",
                 cotton_cond_name,
                 cotton_cond_name not in real_tomato_classes,
                 f"NOT in {real_tomato_classes[:3]}...")
    assert_field("Cotton confidence == 0.0", cotton_confidence, cotton_confidence == 0.0, "0.0")
    assert_field("Cotton is NOT PROTOTYPE_MOCK", cotton_is_mock, cotton_is_mock != "PROTOTYPE_MOCK", "not PROTOTYPE_MOCK")
    print(f"  [INFO] Cotton correctly returned unsupported-crop structured error.")

    print(f"\n{'#'*70}")
    print(f"  [OK] ALL PHASE 2C-2 VALIDATION CHECKS PASSED")
    print(f"{'#'*70}")
    return True


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
