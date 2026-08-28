from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class AlternativeCondition:
    condition_name: str
    condition_type: str  # DISEASE | PEST | PHYSIOLOGICAL
    confidence: float    # 0.0 - 100.0

@dataclass
class ClassifierResult:
    condition_name: str
    condition_type: str  # DISEASE | PEST | PHYSIOLOGICAL
    confidence: float    # 0.0 - 100.0
    alternatives: List[AlternativeCondition] = field(default_factory=list)
    model_name: str = "Mock-Rule-Classifier"
    model_version: str = "1.0.0"
    is_mock: str = "PROTOTYPE_MOCK"
    inference_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    crop_matched: bool = True
    error: Optional[str] = None

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Format to match existing Phase 1 API response structure."""
        return {
            "condition": {
                "name": self.condition_name,
                "type": self.condition_type
            },
            "confidence": self.confidence,
            "alternatives": [
                {
                    "name": alt.condition_name,
                    "type": alt.condition_type,
                    "confidence": alt.confidence
                }
                for alt in self.alternatives
            ],
            "is_mock": self.is_mock,
            "metadata": {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "inference_timestamp": self.inference_timestamp,
                "crop_matched": self.crop_matched
            }
        }

class BaseClassifierService:
    """
    Abstract Classifier Service Interface for CropGuard.
    Defines the contract for both mock and real ML disease/pest classifiers.
    """
    def predict(
        self,
        image_bytes: Optional[bytes] = None,
        crop: str = "Tomato",
        symptoms: Optional[str] = None
    ) -> ClassifierResult:
        raise NotImplementedError

    def classify_crop_image(self, crop: str, symptoms: Optional[str] = None) -> Dict[str, Any]:
        """Legacy helper matching Phase 1 API contract."""
        res = self.predict(image_bytes=None, crop=crop, symptoms=symptoms)
        return res.to_legacy_dict()
