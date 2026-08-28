import pytest
from app.config import settings
from app.services.classifier_interface import BaseClassifierService, ClassifierResult, AlternativeCondition
from app.services.mock_classifier import mock_classifier
from app.services.real_classifier import real_classifier
from app.services.classifier_factory import get_classifier_service
from app.services.image_preprocessor import image_preprocessor, ImagePreprocessingError

def test_classifier_result_contract():
    res = mock_classifier.predict(crop="Tomato", symptoms="Early Blight")
    assert isinstance(res, ClassifierResult)
    assert res.condition_name == "Early Blight"
    assert res.condition_type == "DISEASE"
    assert 0.0 <= res.confidence <= 100.0
    assert res.is_mock == "PROTOTYPE_MOCK"
    assert isinstance(res.alternatives, list)

    legacy_dict = res.to_legacy_dict()
    assert legacy_dict["condition"]["name"] == "Early Blight"
    assert legacy_dict["condition"]["type"] == "DISEASE"
    assert "alternatives" in legacy_dict

def test_disease_and_pest_conditions():
    # Test Disease
    res_dis = mock_classifier.predict(crop="Tomato", symptoms="Early Blight")
    assert res_dis.condition_type == "DISEASE"

    # Test Pest
    res_pest = mock_classifier.predict(crop="Cotton", symptoms="Pink Bollworm")
    assert res_pest.condition_type == "PEST"

def test_real_classifier_unsupported_crop_boundary():
    """Real classifier is Tomato-only — Cotton must be rejected cleanly."""
    res = real_classifier.predict(crop="Cotton", symptoms="Pink Bollworm")
    assert isinstance(res, ClassifierResult)
    assert res.crop_matched is False
    assert res.confidence == 0.0
    assert res.error is not None
    assert "not supported" in res.error.lower() or "tomato" in res.error.lower()
    assert "REAL_CV_MODEL" in res.is_mock
    assert "CropGuard-EfficientNet-B0" in res.model_name

def test_image_preprocessor_validation():
    # Valid JPEG magic bytes header
    valid_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 30
    prep_res = image_preprocessor.validate_and_preprocess(valid_jpeg)
    assert prep_res["valid"] is True
    assert prep_res["format"] == "JPEG"
    assert prep_res["target_size"] == (224, 224)

    # Empty payload -> Exception
    with pytest.raises(ImagePreprocessingError):
        image_preprocessor.validate_and_preprocess(b"")

    # Corrupt / non-image header -> Exception
    with pytest.raises(ImagePreprocessingError):
        image_preprocessor.validate_and_preprocess(b"CORRUPT_NOT_AN_IMAGE_PAYLOAD_12345")

def test_real_classifier_invalid_image_handling():
    corrupt_bytes = b"NOT_AN_IMAGE"
    res = real_classifier.predict(image_bytes=corrupt_bytes, crop="Tomato")
    assert res.confidence == 0.0
    assert res.crop_matched is False
    assert res.error is not None
    assert "corrupt" in res.error.lower() or "invalid" in res.error.lower()

def test_real_classifier_crop_mismatch_handling():
    res_unsupported = real_classifier.predict(crop="UnsupportedCropName123")
    assert res_unsupported.confidence == 0.0
    assert res_unsupported.crop_matched is False
    assert res_unsupported.error is not None
    assert "not supported" in res_unsupported.error.lower()

def test_classifier_factory_mode_switch(monkeypatch):
    # Test Mock Mode Default
    monkeypatch.setattr(settings, "CLASSIFIER_MODE", "mock")
    srv_mock = get_classifier_service()
    assert srv_mock is mock_classifier

    # Test Real Mode Switch
    monkeypatch.setattr(settings, "CLASSIFIER_MODE", "real")
    srv_real = get_classifier_service()
    assert srv_real is real_classifier
