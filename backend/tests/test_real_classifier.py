"""
Tests for RealClassifierService (Phase 2C-1 Integration).

These tests run on CPU and do NOT require CUDA.
The real EfficientNet-B0 checkpoint is loaded from:
  ml/models/checkpoints/best_model.pt

CLASSIFIER_MODE remains "mock" — tests exercise the real service directly
without changing global settings.
"""
import os
import pytest
import torch

from app.config import settings
from app.services.classifier_interface import ClassifierResult, AlternativeCondition
from app.services.real_classifier import RealClassifierService, _BEST_CHECKPOINT, _CLASS_MAPPING
from app.services.mock_classifier import mock_classifier
from app.services.classifier_factory import get_classifier_service

# ── JPEG minimal valid bytes (SOI + APP0 marker) ──────────────────────────────
_VALID_JPEG_HEADER = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 30
_CORRUPT_BYTES = b"NOT_AN_IMAGE_PAYLOAD_1234567890"

# ── Helper: load the real service (one per test session via module-level fixture)
@pytest.fixture(scope="module")
def real_svc():
    svc = RealClassifierService()
    svc.load_model()
    return svc


# ── 1. Import / instantiation ─────────────────────────────────────────────────
def test_real_classifier_imports():
    from app.services.real_classifier import real_classifier
    assert real_classifier is not None


# ── 2. Checkpoint file is present ─────────────────────────────────────────────
def test_checkpoint_file_exists():
    assert os.path.exists(_BEST_CHECKPOINT), (
        f"best_model.pt not found at {_BEST_CHECKPOINT}"
    )


# ── 3. Class mapping file present and correct ─────────────────────────────────
def test_class_mapping_structure():
    assert os.path.exists(_CLASS_MAPPING)
    import json
    with open(_CLASS_MAPPING, encoding="utf-8") as f:
        data = json.load(f)
    assert "class_to_idx" in data
    assert "idx_to_class" in data
    assert len(data["class_to_idx"]) == 10
    assert len(data["idx_to_class"]) == 10


# ── 4. Model loads successfully ───────────────────────────────────────────────
def test_model_loads_successfully(real_svc):
    assert real_svc._model_loaded is True
    assert real_svc._model is not None
    assert real_svc._load_error is None


# ── 5. Class mapping loads with exactly 10 classes ───────────────────────────
def test_model_has_exactly_10_classes(real_svc):
    assert len(real_svc._idx_to_class) == 10
    expected = {
        0: "Tomato___Bacterial_spot",
        1: "Tomato___Early_blight",
        2: "Tomato___Late_blight",
        3: "Tomato___Leaf_Mold",
        4: "Tomato___Septoria_leaf_spot",
        5: "Tomato___Spider_mites Two-spotted_spider_mite",
        6: "Tomato___Target_Spot",
        7: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        8: "Tomato___Tomato_mosaic_virus",
        9: "Tomato___healthy",
    }
    assert real_svc._idx_to_class == expected


# ── 6. Forward pass produces 10-class output ─────────────────────────────────
def test_model_produces_10_class_output(real_svc):
    import numpy as np
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        logits = real_svc._model(dummy.to(real_svc._device))
    assert logits.shape == (1, 10)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    assert probs.shape == (10,)
    assert abs(float(np.sum(probs)) - 1.0) < 1e-4


# ── 7. Tomato accepted — real file from test split ───────────────────────────
def test_tomato_real_inference_with_test_image(real_svc):
    """Use an actual image from the PlantVillage test split."""
    test_dir = os.path.join(
        os.path.dirname(_BEST_CHECKPOINT), "..", "..", "data", "splits", "test"
    )
    test_dir = os.path.normpath(test_dir)
    test_img_bytes = None
    if os.path.exists(test_dir):
        for root, _, files in os.walk(test_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg")):
                    with open(os.path.join(root, f), "rb") as fh:
                        test_img_bytes = fh.read()
                    break
            if test_img_bytes:
                break

    if test_img_bytes is None:
        pytest.skip("No test image found in ml/data/splits/test — skipping.")

    res = real_svc.predict(image_bytes=test_img_bytes, crop="Tomato")
    assert isinstance(res, ClassifierResult)
    assert res.crop_matched is True
    assert res.is_mock == "REAL_CV_MODEL"
    assert res.condition_name in real_svc._idx_to_class.values()
    assert 0.0 <= res.confidence <= 100.0
    assert len(res.alternatives) == 2
    for alt in res.alternatives:
        assert alt.condition_name in real_svc._idx_to_class.values()
        assert 0.0 <= alt.confidence <= 100.0
    assert "CropGuard-EfficientNet-B0" in res.model_name
    assert res.model_version == "0.1.0"
    assert res.error is None


# ── 8. Confidence is strictly in [0, 100] ────────────────────────────────────
def test_confidence_bounds(real_svc):
    test_dir = os.path.join(
        os.path.dirname(_BEST_CHECKPOINT), "..", "..", "data", "splits", "test"
    )
    test_dir = os.path.normpath(test_dir)
    if not os.path.exists(test_dir):
        pytest.skip("Test split not found.")
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg")):
                with open(os.path.join(root, f), "rb") as fh:
                    img_bytes = fh.read()
                res = real_svc.predict(image_bytes=img_bytes, crop="Tomato")
                assert 0.0 <= res.confidence <= 100.0
                return
    pytest.skip("No image found.")


# ── 9. Cotton rejected with unsupported error ─────────────────────────────────
def test_cotton_rejected_by_real_classifier(real_svc):
    res = real_svc.predict(image_bytes=_VALID_JPEG_HEADER, crop="Cotton")
    assert isinstance(res, ClassifierResult)
    assert res.crop_matched is False
    assert res.confidence == 0.0
    assert res.error is not None
    assert "not supported" in res.error.lower() or "tomato-only" in res.error.lower()
    # Must NOT produce a Tomato disease prediction for Cotton
    assert res.condition_name not in [
        "Tomato___Bacterial_spot", "Tomato___Early_blight",
        "Tomato___Late_blight", "Tomato___Leaf_Mold"
    ]


# ── 10. Soybean rejected with unsupported error ───────────────────────────────
def test_soybean_rejected_by_real_classifier(real_svc):
    res = real_svc.predict(image_bytes=_VALID_JPEG_HEADER, crop="Soybean")
    assert res.crop_matched is False
    assert res.confidence == 0.0
    assert res.error is not None


# ── 11. Invalid / corrupt image handled safely ───────────────────────────────
def test_corrupt_image_handled_safely(real_svc):
    res = real_svc.predict(image_bytes=_CORRUPT_BYTES, crop="Tomato")
    assert isinstance(res, ClassifierResult)
    assert res.confidence == 0.0
    assert res.crop_matched is False
    assert res.error is not None


# ── 12. Missing model path returns structured error ──────────────────────────
def test_missing_model_returns_error():
    import sys
    # Use sys.modules to get the actual module, avoiding name collision with
    # the module-level imports at the top of this test file.
    rcmod = sys.modules["app.services.real_classifier"]
    orig_best = rcmod._BEST_CHECKPOINT
    orig_fallback = rcmod._FALLBACK_MODEL
    rcmod._BEST_CHECKPOINT = "/nonexistent/best_model.pt"
    rcmod._FALLBACK_MODEL  = "/nonexistent/fallback.pt"
    try:
        svc = RealClassifierService()
        svc.load_model()
        assert svc._model_loaded is False
        assert svc._load_error is not None
    finally:
        rcmod._BEST_CHECKPOINT = orig_best
        rcmod._FALLBACK_MODEL  = orig_fallback


# ── 13. Mock classifier still works ──────────────────────────────────────────
def test_mock_classifier_still_works():
    res = mock_classifier.predict(crop="Tomato", symptoms="Early Blight")
    assert isinstance(res, ClassifierResult)
    assert res.is_mock == "PROTOTYPE_MOCK"
    assert res.condition_name == "Early Blight"
    assert 0.0 <= res.confidence <= 100.0


def test_mock_supports_cotton_and_soybean():
    for crop in ["Cotton", "Soybean"]:
        res = mock_classifier.predict(crop=crop)
        assert isinstance(res, ClassifierResult)
        assert res.confidence > 0.0


# ── 14. Factory correctly switches mock / real ────────────────────────────────
def test_factory_mock_mode(monkeypatch):
    monkeypatch.setattr(settings, "CLASSIFIER_MODE", "mock")
    svc = get_classifier_service()
    assert svc is mock_classifier


def test_factory_real_mode(monkeypatch):
    from app.services.real_classifier import real_classifier
    monkeypatch.setattr(settings, "CLASSIFIER_MODE", "real")
    svc = get_classifier_service()
    assert svc is real_classifier


# ── 15. Real classifier does NOT silently fall back to mock ──────────────────
def test_real_classifier_does_not_return_mock_tag(real_svc):
    """When real mode is used, is_mock must never be PROTOTYPE_MOCK."""
    test_dir = os.path.join(
        os.path.dirname(_BEST_CHECKPOINT), "..", "..", "data", "splits", "test"
    )
    test_dir = os.path.normpath(test_dir)
    if not os.path.exists(test_dir):
        pytest.skip("Test split not found.")
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg")):
                with open(os.path.join(root, f), "rb") as fh:
                    img_bytes = fh.read()
                res = real_svc.predict(image_bytes=img_bytes, crop="Tomato")
                assert res.is_mock != "PROTOTYPE_MOCK"
                assert res.is_mock == "REAL_CV_MODEL"
                return
    pytest.skip("No image found.")
