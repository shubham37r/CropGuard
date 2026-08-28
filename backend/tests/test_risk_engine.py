import pytest
from app.services.risk_engine import risk_engine
from app.config import settings

def test_risk_engine_weights_and_normalization():
    # Min score test: stage_risk for Maturity is 40.0 -> score = 0.15 * 40 = 6.0
    res = risk_engine.calculate_risk(disease_confidence=0, weather_risk=0, crop_stage="Maturity", outbreak_signal=0)
    assert res["score"] == 6.0
    assert res["risk_level"] == "LOW"
    assert "methodology_note" in res
    assert res["methodology_note"] == settings.METHODOLOGY_NOTE

    # Max score test: all 100 -> score = 40 + 25 + 15*0.85 + 20 = 97.7 -> HIGH
    res_high = risk_engine.calculate_risk(disease_confidence=100, weather_risk=100, crop_stage="Flowering", outbreak_signal=100)
    assert res_high["score"] >= 70.0
    assert res_high["risk_level"] == "HIGH"

def test_risk_boundaries_exact():
    # Test boundary score mapping:
    # 0 - 39 -> LOW
    # 40 - 69 -> MEDIUM
    # 70 - 100 -> HIGH
    
    # 1. Exact Score = 39.0 (LOW)
    # 0.40 * 82.5 (33.0) + 0.15 * 40.0 (6.0) = 39.0
    res_39 = risk_engine.calculate_risk(disease_confidence=82.5, weather_risk=0, crop_stage="Maturity", outbreak_signal=0)
    assert res_39["score"] == 39.0
    assert res_39["risk_level"] == "LOW"

    # 2. Exact Score = 40.0 (MEDIUM)
    # 0.40 * 85.0 (34.0) + 0.15 * 40.0 (6.0) = 40.0
    res_40 = risk_engine.calculate_risk(disease_confidence=85.0, weather_risk=0, crop_stage="Maturity", outbreak_signal=0)
    assert res_40["score"] == 40.0
    assert res_40["risk_level"] == "MEDIUM"

    # 3. Exact Score = 69.0 (MEDIUM)
    # 0.40 * 100 (40.0) + 0.25 * 92 (23.0) + 0.15 * 40 (6.0) = 69.0
    res_69 = risk_engine.calculate_risk(disease_confidence=100, weather_risk=92, crop_stage="Maturity", outbreak_signal=0)
    assert res_69["score"] == 69.0
    assert res_69["risk_level"] == "MEDIUM"

    # 4. Exact Score = 70.0 (HIGH)
    # 0.40 * 100 (40.0) + 0.25 * 96 (24.0) + 0.15 * 40 (6.0) = 70.0
    res_70 = risk_engine.calculate_risk(disease_confidence=100, weather_risk=96, crop_stage="Maturity", outbreak_signal=0)
    assert res_70["score"] == 70.0
    assert res_70["risk_level"] == "HIGH"

def test_risk_component_scores_structure():
    res = risk_engine.calculate_risk(disease_confidence=85.0, weather_risk=70.0, crop_stage="Fruiting", outbreak_signal=50.0)
    assert "component_scores" in res
    comps = res["component_scores"]
    assert comps["disease_confidence"] == 85.0
    assert comps["weather_risk"] == 70.0
    assert comps["stage_risk"] == 80.0
    assert comps["outbreak_signal"] == 50.0
    assert isinstance(res["contributing_factors"], list)
    assert len(res["contributing_factors"]) > 0
