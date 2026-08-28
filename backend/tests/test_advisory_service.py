import pytest
from app.services.advisory_service import advisory_service

def test_advisory_structure_and_safety():
    conditions = ["Early Blight", "Pink Bollworm", "Asian Soybean Rust", "Tomato Yellow Leaf Curl Virus", "Cotton Leaf Curl Virus", "Unknown Condition"]
    crops = ["Tomato", "Cotton", "Soybean"]

    for cond in conditions:
        for crop in crops:
            adv = advisory_service.generate_advisory(
                crop=crop,
                condition=cond,
                condition_type="DISEASE",
                risk_level="HIGH",
                crop_stage="Flowering",
                confidence=85.0
            )

            assert "priority" in adv
            assert "actions" in adv
            assert "monitoring" in adv
            assert "safety_note" in adv
            assert "expert_referral" in adv

            assert isinstance(adv["actions"], list)
            assert len(adv["actions"]) > 0
            assert isinstance(adv["monitoring"], list)

            # Safety Audit: Verify NO chemical dosage or chemical spray volume prescriptions exist
            full_text = " ".join(adv["actions"] + adv["monitoring"] + [adv["safety_note"]]).lower()
            assert "ml/l" not in full_text
            assert "kg/acre" not in full_text
            assert "g/l" not in full_text
            assert "spray 500ml" not in full_text

def test_expert_referral_trigger_on_low_confidence():
    adv_low = advisory_service.generate_advisory(
        crop="Tomato",
        condition="Early Blight",
        condition_type="DISEASE",
        risk_level="MEDIUM",
        crop_stage="Vegetative",
        confidence=62.0  # < 70% threshold
    )
    assert adv_low["expert_referral"] is True

def test_expert_referral_trigger_on_high_risk():
    adv_high = advisory_service.generate_advisory(
        crop="Cotton",
        condition="Pink Bollworm",
        condition_type="PEST",
        risk_level="HIGH",
        crop_stage="Fruiting",
        confidence=89.0
    )
    assert adv_high["expert_referral"] is True
