import io
import os
import json
from typing import Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

from app.config import settings
from app.services.classifier_interface import BaseClassifierService, ClassifierResult, AlternativeCondition
from app.services.image_preprocessor import image_preprocessor, ImagePreprocessingError

# ── path resolution ──────────────────────────────────────────────────────────
# __file__ = .../CropGuard/backend/app/services/real_classifier.py
# dirname x1 = .../backend/app/services/
# dirname x2 = .../backend/app/
# dirname x3 = .../backend/
# dirname x4 = .../CropGuard/  ← project root
_PROJECT_ROOT = os.path.dirname(  # CropGuard/
    os.path.dirname(              # backend/
        os.path.dirname(          # app/
            os.path.dirname(      # services/
                os.path.abspath(__file__)
            )
        )
    )
)

_BEST_CHECKPOINT = os.path.join(_PROJECT_ROOT, "ml", "models", "checkpoints", "best_model.pt")
_FALLBACK_MODEL  = os.path.join(_PROJECT_ROOT, "ml", "models", "cropguard_disease_model.pt")
_CLASS_MAPPING   = os.path.join(_PROJECT_ROOT, "ml", "reports", "class_mapping.json")

# Condition type derived from class name
def _condition_type(class_name: str) -> str:
    name_lower = class_name.lower()
    if "spider" in name_lower or "mite" in name_lower or "borer" in name_lower:
        return "PEST"
    if "healthy" in name_lower:
        return "HEALTHY"
    return "DISEASE"


class RealClassifierService(BaseClassifierService):
    """
    Real EfficientNet-B0 Classifier for CropGuard (PyTorch backend).

    Design:
    - Lazy-loads best_model.pt once; reuses for all requests.
    - Runs on CPU or CUDA, auto-detected.
    - Inference: torch.no_grad() + softmax.
    - Preprocessing exactly matches Phase 2B training:
        RGB → Resize(224,224) → ToTensor → ImageNet Normalize.
    - Tomato-only: Cotton / Soybean return a structured unsupported-crop error.
    - Does NOT fall back silently to mock behaviour.
    """

    MODEL_NAME = "CropGuard-EfficientNet-B0"
    MODEL_VERSION = "0.1.0"
    IS_MOCK_TAG = "REAL_CV_MODEL"
    SUPPORTED_CROPS = {"Tomato"}
    NUM_EXPECTED_CLASSES = 10

    # Preprocessing matching Phase 2B training/evaluation exactly
    _EVAL_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def __init__(self):
        self._model: Optional[torch.nn.Module] = None
        self._model_loaded: bool = False
        self._load_error: Optional[str] = None
        self._idx_to_class: dict = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Model & mapping loading (lazy, called once)
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Load EfficientNet-B0 weights and class mapping exactly once."""
        if self._model_loaded or self._load_error:
            return

        # 1. Load class mapping
        if not os.path.exists(_CLASS_MAPPING):
            self._load_error = f"Class mapping file not found: {_CLASS_MAPPING}"
            print(f"[RealClassifierService] ERROR: {self._load_error}")
            return

        try:
            with open(_CLASS_MAPPING, "r", encoding="utf-8") as f:
                mdata = json.load(f)
            self._idx_to_class = {int(k): v for k, v in mdata["idx_to_class"].items()}
        except Exception as exc:
            self._load_error = f"Failed to parse class_mapping.json: {exc}"
            print(f"[RealClassifierService] ERROR: {self._load_error}")
            return

        if len(self._idx_to_class) != self.NUM_EXPECTED_CLASSES:
            self._load_error = (
                f"Unexpected number of classes in mapping: "
                f"expected {self.NUM_EXPECTED_CLASSES}, got {len(self._idx_to_class)}"
            )
            print(f"[RealClassifierService] ERROR: {self._load_error}")
            return

        # 2. Resolve checkpoint path (prefer settings.MODEL_PATH, fallback to _BEST_CHECKPOINT / _FALLBACK_MODEL)
        model_path = getattr(settings, "MODEL_PATH", _BEST_CHECKPOINT)
        if not os.path.exists(model_path) or not os.path.exists(_BEST_CHECKPOINT):
            model_path = _BEST_CHECKPOINT
        if not os.path.exists(model_path):
            model_path = _FALLBACK_MODEL



        if not os.path.exists(model_path):
            self._load_error = (
                f"No model checkpoint found. Checked:\n"
                f"  {_BEST_CHECKPOINT}\n"
                f"  {_FALLBACK_MODEL}"
            )
            print(f"[RealClassifierService] ERROR: {self._load_error}")
            return

        # 3. Build architecture and load weights
        try:
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.NUM_EXPECTED_CLASSES)

            # weights_only=False required for PyTorch ≥2.6 with dict checkpoints
            raw = torch.load(model_path, map_location=self._device, weights_only=False)

            state_dict = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw

            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()

            self._model = model
            self._model_loaded = True
            epoch_info = f"epoch={raw.get('epoch', '?')}" if isinstance(raw, dict) else "raw state_dict"
            print(
                f"[RealClassifierService] Loaded {model_path} ({epoch_info}) "
                f"on {self._device} | classes={len(self._idx_to_class)}"
            )
        except Exception as exc:
            self._load_error = f"Checkpoint load failed: {exc}"
            print(f"[RealClassifierService] ERROR: {self._load_error}")

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        image_bytes: Optional[bytes] = None,
        crop: str = "Tomato",
        symptoms: Optional[str] = None,
    ) -> ClassifierResult:

        # ── 1. Image validation ──────────────────────────────────────────
        if image_bytes is not None:
            try:
                image_preprocessor.validate_and_preprocess(image_bytes)
            except ImagePreprocessingError as err:
                return self._error_result("Invalid Image", str(err))

        # ── 2. Crop validation (Tomato-only real model) ──────────────────
        crop_clean = crop.strip().title() if crop else "Tomato"
        if crop_clean not in self.SUPPORTED_CROPS:
            return ClassifierResult(
                condition_name="Unsupported Crop",
                condition_type="DISEASE",
                confidence=0.0,
                alternatives=[],
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                is_mock=self.IS_MOCK_TAG,
                crop_matched=False,
                error=(
                    f"The real EfficientNet-B0 model is a Tomato-only model. "
                    f"Crop '{crop}' is not supported by the current classifier. "
                    f"Use CLASSIFIER_MODE=mock for Cotton/Soybean."
                )
            )

        # ── 3. Lazy model load ──────────────────────────────────────────
        self.load_model()

        if not self._model_loaded or self._model is None:
            return self._error_result(
                "Model Unavailable",
                self._load_error or "Real model failed to load. Check logs."
            )

        # ── 4. Require image bytes for real inference ────────────────────
        if image_bytes is None:
            return self._error_result(
                "No Image Provided",
                "Real CV model requires image bytes to perform inference."
            )

        # ── 5. Inference ─────────────────────────────────────────────────
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self._EVAL_TRANSFORM(img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            sorted_indices = np.argsort(probs)[::-1]

            top_idx = int(sorted_indices[0])
            top_name = self._idx_to_class.get(top_idx, f"Unknown_{top_idx}")
            top_conf = round(float(probs[top_idx] * 100.0), 2)
            top_type = _condition_type(top_name)

            # Next-best 2 alternatives (not top-1)
            alternatives = []
            for idx in sorted_indices[1:3]:
                alt_name = self._idx_to_class.get(int(idx), f"Unknown_{idx}")
                alternatives.append(AlternativeCondition(
                    condition_name=alt_name,
                    condition_type=_condition_type(alt_name),
                    confidence=round(float(probs[idx] * 100.0), 2)
                ))

            return ClassifierResult(
                condition_name=top_name,
                condition_type=top_type,
                confidence=top_conf,
                alternatives=alternatives,
                model_name=f"{self.MODEL_NAME} ({self._device})",
                model_version=self.MODEL_VERSION,
                is_mock=self.IS_MOCK_TAG,
                crop_matched=True,
            )

        except Exception as exc:
            return self._error_result("Inference Error", str(exc))

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _error_result(self, condition_name: str, error_msg: str) -> ClassifierResult:
        return ClassifierResult(
            condition_name=condition_name,
            condition_type="DISEASE",
            confidence=0.0,
            alternatives=[],
            model_name=self.MODEL_NAME,
            model_version=self.MODEL_VERSION,
            is_mock=self.IS_MOCK_TAG,
            crop_matched=False,
            error=error_msg,
        )


real_classifier = RealClassifierService()
