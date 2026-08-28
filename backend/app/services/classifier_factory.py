from app.config import settings
from app.services.classifier_interface import BaseClassifierService
from app.services.mock_classifier import mock_classifier
from app.services.real_classifier import real_classifier

def get_classifier_service() -> BaseClassifierService:
    """
    Returns active classifier service based on CLASSIFIER_MODE settings.
    Default mode: 'mock' (Phase 1).
    """
    mode = settings.CLASSIFIER_MODE.lower()
    if mode == "real":
        return real_classifier
    return mock_classifier
