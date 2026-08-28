import pytest
from app.services.mock_classifier import mock_classifier

def test_classifier_data_contract():
    crops = ["Tomato", "Cotton", "Soybean"]
    for c in crops:
        res = mock_classifier.classify_crop_image(crop=c)
        assert "condition" in res
        assert "confidence" in res
        assert "alternatives" in res
        assert "is_mock" in res

        cond = res["condition"]
        assert "name" in cond
        assert "type" in cond
        assert cond["type"] in ["DISEASE", "PEST"]

        alternatives = res["alternatives"]
        assert isinstance(alternatives, list)
        for alt in alternatives:
            assert "name" in alt
            assert "type" in alt
            assert "confidence" in alt

def test_classifier_distinguishes_disease_and_pest():
    res_tomato = mock_classifier.classify_crop_image(crop="Tomato", symptoms="Early Blight spots")
    assert res_tomato["condition"]["name"] == "Early Blight"
    assert res_tomato["condition"]["type"] == "DISEASE"

    res_cotton = mock_classifier.classify_crop_image(crop="Cotton", symptoms="Pink Bollworm larvae")
    assert res_cotton["condition"]["name"] == "Pink Bollworm"
    assert res_cotton["condition"]["type"] == "PEST"
