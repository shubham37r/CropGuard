import os

class Settings:
    PROJECT_NAME: str = "CropGuard Early Warning System"
    API_V1_STR: str = "/api"
    DATABASE_URL: str = "sqlite:///./cropguard.db"

    # Risk Engine Configuration (Configurable Weights & Thresholds)
    WEIGHT_DISEASE_CONFIDENCE: float = 0.40
    WEIGHT_WEATHER_RISK: float = 0.25
    WEIGHT_STAGE_RISK: float = 0.15
    WEIGHT_OUTBREAK_SIGNAL: float = 0.20

    CONFIDENCE_VERIFICATION_THRESHOLD: float = 70.0  # %

    METHODOLOGY_NOTE: str = (
        "This is a prototype contextual risk score based on model confidence and contextual signals. "
        "It is not a scientifically validated disease-risk probability."
    )

    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")

    # Phase 2B/2C/2F: Classifier Service Configuration
    # Real EfficientNet-B0 inference activated with accepted candidate model: best_unified_tomato_model.pt
    _DEFAULT_UNIFIED_PATH: str = os.path.join(os.path.dirname(BASE_DIR), "ml", "models", "unified_adaptation", "checkpoints", "best_unified_tomato_model.pt")
    _DEFAULT_BASELINE_PATH: str = os.path.join(os.path.dirname(BASE_DIR), "ml", "models", "checkpoints", "best_model.pt")

    CLASSIFIER_MODE: str = os.getenv("CLASSIFIER_MODE", "mock")  # "mock" or "real"

    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        _DEFAULT_UNIFIED_PATH if os.path.exists(_DEFAULT_UNIFIED_PATH) else _DEFAULT_BASELINE_PATH
    )
    MODEL_NAME: str = "CropGuard-EfficientNet-B0"
    MODEL_VERSION: str = "0.2.0-unified"


settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
