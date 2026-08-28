import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_root_status():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "disclaimer" in data

def test_create_report_high_confidence_flow():
    payload = {
        "farmer_id": 1,
        "crop": "Tomato",
        "variety": "Abhinav",
        "growth_stage": "Flowering",
        "symptoms_description": "Early Blight spots on lower leaves",
        "image_url": "https://images.unsplash.com/photo-1592841200221-a6898f307baa?w=600",
        "location": {
            "latitude": 21.1458,
            "longitude": 79.0882,
            "district": "Nagpur",
            "address": "Nagpur Central Block"
        }
    }
    res = client.post("/api/reports/", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["crop"] == "Tomato"
    assert data["growth_stage"] == "Flowering"
    assert data["status"] in ["ANALYZED", "PENDING_VERIFICATION"]
    assert "analysis" in data
    assert data["analysis"]["confidence"] >= 70.0
    assert "risk_assessment" in data
    assert data["risk_assessment"]["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "advisory" in data

def test_create_report_low_confidence_flow():
    # Symptoms engineered to trigger low confidence scenario in mock classifier (e.g. Tomato Yellow Leaf Curl -> 64% conf)
    payload = {
        "farmer_id": 1,
        "crop": "Tomato",
        "variety": "Local",
        "growth_stage": "Vegetative",
        "symptoms_description": "Tomato Yellow Leaf Curl Virus symptoms yellowing",
        "image_url": "https://images.unsplash.com/photo-1592841200221-a6898f307baa?w=600",
        "location": {
            "latitude": 21.1500,
            "longitude": 79.0900,
            "district": "Nagpur",
            "address": "Nagpur West Block"
        }
    }
    res = client.post("/api/reports/", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["analysis"]["confidence"] < 70.0
    # Mandatory requirement: Low confidence (<70%) triggers PENDING_VERIFICATION automatically
    assert data["status"] == "PENDING_VERIFICATION"

def test_officer_verification_lifecycle_transitions():
    # Create report first
    payload = {
        "farmer_id": 1,
        "crop": "Cotton",
        "growth_stage": "Flowering",
        "symptoms_description": "Pink Bollworm infestation",
        "image_url": "https://images.unsplash.com/photo-1605000797499-95a51c5269ae?w=600",
        "location": {
            "latitude": 21.2825,
            "longitude": 78.5840,
            "district": "Nagpur",
            "address": "Katol Block"
        }
    }
    create_res = client.post("/api/reports/", json=payload)
    report_id = create_res.json()["id"]

    # 1. PENDING_VERIFICATION -> CONFIRMED
    verif_payload = {
        "status": "CONFIRMED",
        "officer_notes": "Field visit confirmed Pink Bollworm larvae in bolls."
    }
    res_confirm = client.post(f"/api/verification/{report_id}?officer_id=4", json=verif_payload)
    assert res_confirm.status_code == 200
    assert res_confirm.json()["status"] == "CONFIRMED"
    assert res_confirm.json()["verification"]["officer_notes"] == verif_payload["officer_notes"]

    # 2. Transition -> REJECTED
    verif_reject = {
        "status": "REJECTED",
        "officer_notes": "Rejected. Symptoms are due to heat stress."
    }
    res_reject = client.post(f"/api/verification/{report_id}?officer_id=4", json=verif_reject)
    assert res_reject.status_code == 200
    assert res_reject.json()["status"] == "REJECTED"

    # 3. Transition -> NEEDS_MORE_INFO
    verif_info = {
        "status": "NEEDS_MORE_INFO",
        "officer_notes": "Please upload additional photo of fruit cross section."
    }
    res_info = client.post(f"/api/verification/{report_id}?officer_id=4", json=verif_info)
    assert res_info.status_code == 200
    assert res_info.json()["status"] == "NEEDS_MORE_INFO"

def test_hotspots_endpoint():
    res = client.get("/api/hotspots/")
    assert res.status_code == 200
    data = res.json()
    assert "points" in data
    assert "clusters" in data
    assert isinstance(data["points"], list)
    assert isinstance(data["clusters"], list)
    assert len(data["points"]) > 0
