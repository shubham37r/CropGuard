from typing import Dict, Any, List, Optional
from app.services.classifier_interface import BaseClassifierService, ClassifierResult, AlternativeCondition

class MockClassifierService(BaseClassifierService):
    """
    Mock Image Classifier Service for Prototype.
    Returns structured conditions distinguishing DISEASE vs PEST with alternatives.
    Does NOT claim real ML inference performance.
    """

    CROP_CONDITIONS = {
        "Tomato": [
            {
                "condition": {"name": "Early Blight", "type": "DISEASE"},
                "confidence": 87.0,
                "alternatives": [
                    {"name": "Late Blight", "type": "DISEASE", "confidence": 8.0},
                    {"name": "Septoria Leaf Spot", "type": "DISEASE", "confidence": 5.0}
                ]
            },
            {
                "condition": {"name": "Tomato Yellow Leaf Curl Virus", "type": "DISEASE"},
                "confidence": 64.0,  # Low confidence scenario
                "alternatives": [
                    {"name": "Whitefly Infestation", "type": "PEST", "confidence": 25.0},
                    {"name": "Mosaic Virus", "type": "DISEASE", "confidence": 11.0}
                ]
            },
            {
                "condition": {"name": "Tomato Fruit Borer", "type": "PEST"},
                "confidence": 91.0,
                "alternatives": [
                    {"name": "Cutworm Damage", "type": "PEST", "confidence": 6.0},
                    {"name": "Sunscald", "type": "DISEASE", "confidence": 3.0}
                ]
            }
        ],
        "Cotton": [
            {
                "condition": {"name": "Pink Bollworm", "type": "PEST"},
                "confidence": 89.0,
                "alternatives": [
                    {"name": "American Bollworm", "type": "PEST", "confidence": 7.0},
                    {"name": "Spotted Bollworm", "type": "PEST", "confidence": 4.0}
                ]
            },
            {
                "condition": {"name": "Cotton Leaf Curl Virus", "type": "DISEASE"},
                "confidence": 62.0,  # Low confidence scenario
                "alternatives": [
                    {"name": "Aphid Infestation", "type": "PEST", "confidence": 28.0},
                    {"name": "Bacterial Blight", "type": "DISEASE", "confidence": 10.0}
                ]
            },
            {
                "condition": {"name": "Whitefly Infestation", "type": "PEST"},
                "confidence": 78.0,
                "alternatives": [
                    {"name": "Thrips Damage", "type": "PEST", "confidence": 15.0},
                    {"name": "Jassids Attack", "type": "PEST", "confidence": 7.0}
                ]
            }
        ],
        "Soybean": [
            {
                "condition": {"name": "Asian Soybean Rust", "type": "DISEASE"},
                "confidence": 85.0,
                "alternatives": [
                    {"name": "Cercospora Leaf Blight", "type": "DISEASE", "confidence": 10.0},
                    {"name": "Brown Spot", "type": "DISEASE", "confidence": 5.0}
                ]
            },
            {
                "condition": {"name": "Stem Fly Attack", "type": "PEST"},
                "confidence": 66.0,  # Low confidence scenario
                "alternatives": [
                    {"name": "Girdle Beetle", "type": "PEST", "confidence": 22.0},
                    {"name": "Charcoal Rot", "type": "DISEASE", "confidence": 12.0}
                ]
            },
            {
                "condition": {"name": "Tobacco Caterpillar", "type": "PEST"},
                "confidence": 82.0,
                "alternatives": [
                    {"name": "Green Semilooper", "type": "PEST", "confidence": 12.0},
                    {"name": "Bacterial Pustule", "type": "DISEASE", "confidence": 6.0}
                ]
            }
        ]
    }

    def predict(
        self,
        image_bytes: Optional[bytes] = None,
        crop: str = "Tomato",
        symptoms: Optional[str] = None
    ) -> ClassifierResult:
        crop_name = crop.strip().title() if crop else "Tomato"
        if crop_name not in self.CROP_CONDITIONS:
            crop_name = "Tomato"

        options = self.CROP_CONDITIONS[crop_name]
        selected = options[0]
        if symptoms:
            sym_lower = symptoms.lower()
            for opt in options:
                cond_name = opt["condition"]["name"].lower()
                if any(k in sym_lower for k in cond_name.split()):
                    selected = opt
                    break

        alt_objects = [
            AlternativeCondition(
                condition_name=alt["name"],
                condition_type=alt["type"],
                confidence=alt["confidence"]
            )
            for alt in selected["alternatives"]
        ]

        return ClassifierResult(
            condition_name=selected["condition"]["name"],
            condition_type=selected["condition"]["type"],
            confidence=selected["confidence"],
            alternatives=alt_objects,
            model_name="Mock-Rule-Classifier",
            model_version="1.0.0",
            is_mock="PROTOTYPE_MOCK",
            crop_matched=True
        )

    def classify_crop_image(self, crop: str, symptoms: Optional[str] = None) -> Dict[str, Any]:
        res = self.predict(image_bytes=None, crop=crop, symptoms=symptoms)
        return res.to_legacy_dict()

mock_classifier = MockClassifierService()
