"""
CropGuard Phase 2D — Temperature Scaling Confidence Calibration
================================================================
Fits a single scalar temperature T on the VALIDATION split logits,
then evaluates both RAW and CALIBRATED confidence on the held-out
TEST split.

IMPORTANT SAFETY GUARANTEES:
  - Model weights (best_model.pt) are NEVER modified.
  - Temperature T is fit ONLY on validation data.
  - Test data is used ONLY for final evaluation, never for fitting.
  - Production classifier (RealClassifierService) is NOT changed.
  - CLASSIFIER_MODE remains mock.

Usage:
    python -m ml.src.calibrate_model

Outputs (ml/reports/):
    calibration_config.json      - fitted temperature + metadata
    calibration_report.json      - full raw vs calibrated comparison
    calibration_report.md        - human-readable report
    reliability_diagram.png      - reliability diagram (raw vs calibrated)
"""

import os
import json
import hashlib
import numpy as np
from datetime import datetime, timezone
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_DIR     = os.path.join(BASE_DIR, "data", "splits")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
BEST_CKPT      = os.path.join(CHECKPOINTS_DIR, "best_model.pt")
MAPPING_PATH   = os.path.join(REPORTS_DIR, "class_mapping.json")

VAL_DIR  = os.path.join(SPLITS_DIR, "validation")
TEST_DIR = os.path.join(SPLITS_DIR, "test")

RANDOM_SEED = 42
BASELINE_SHA = "300a46ea8c2a3f9ff3f0e6fee0a9acdf19ecf5f8dc6b3e76ebe83c8a3b9d87d3"


# ── Class mapping ─────────────────────────────────────────────────────────────
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    _map = json.load(f)

class_to_idx  = _map["class_to_idx"]
idx_to_class  = {int(k): v for k, v in _map["idx_to_class"].items()}
num_classes   = len(class_to_idx)
sorted_classes = [idx_to_class[i] for i in range(num_classes)]

# ── Eval preprocessing (must match training) ──────────────────────────────────
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class SplitDataset(Dataset):
    """Loads images from a split directory using the class mapping."""

    def __init__(self, split_dir: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        for cls_name in sorted_classes:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            cls_idx = class_to_idx[cls_name]
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, label


# ── Model builder (identical to evaluate_model.py) ────────────────────────────
def build_model() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# ── Checkpoint SHA256 ─────────────────────────────────────────────────────────
def checkpoint_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Logit collection (frozen model, no grad) ──────────────────────────────────
@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device):
    """Returns (logits_np, labels_np) as numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


# ── Temperature Scaling ───────────────────────────────────────────────────────
class TemperatureScaler:
    """
    Fits a single scalar temperature T to minimise NLL on logits collected
    from the VALIDATION split.  Model weights are never touched.

    calibrated_probs = softmax(logits / T)
    """

    def __init__(self):
        self.temperature: float = 1.0  # raw (uncalibrated) baseline

    def fit(self, logits_np: np.ndarray, labels_np: np.ndarray,
            lr: float = 0.01, max_iter: int = 100) -> float:
        """
        Optimise NLL w.r.t. T using LBFGS on the validation logits.
        Returns the fitted temperature.
        """
        logits_t = torch.tensor(logits_np, dtype=torch.float32)
        labels_t = torch.tensor(labels_np, dtype=torch.long)

        # Learnable scalar, initialised to 1.0
        T = nn.Parameter(torch.ones(1))
        optimizer = optim.LBFGS([T], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits_t / T, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Clamp to a sensible range to avoid degenerate solutions
        fitted_T = float(T.item())
        fitted_T = max(0.05, min(10.0, fitted_T))
        self.temperature = fitted_T
        return fitted_T

    def scale_logits(self, logits_np: np.ndarray) -> np.ndarray:
        """Return calibrated logits = logits / T."""
        return logits_np / self.temperature

    def calibrated_probs(self, logits_np: np.ndarray) -> np.ndarray:
        scaled = self.scale_logits(logits_np)
        exp    = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


# ── Calibration Metrics ───────────────────────────────────────────────────────
def compute_ece(probs: np.ndarray, labels: np.ndarray,
                n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) — equal-width binning.
    ECE = sum_b (|B_b| / n) * |accuracy(B_b) - confidence(B_b)|
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)
    n = len(labels)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Multiclass Brier score = mean squared error of the probability vector
    against one-hot true labels.
    Lower is better; perfectly calibrated model on correct predictions → 0.
    """
    n, k = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def confidence_stats(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute the confidence analysis identical to evaluate_model.py."""
    preds         = probs.argmax(axis=1)
    confidences   = probs.max(axis=1) * 100.0
    correct_mask  = preds == labels
    incorrect_mask = ~correct_mask
    inc_confs     = confidences[incorrect_mask]

    return {
        "mean_confidence":   round(float(confidences.mean()), 2),
        "median_confidence": round(float(np.median(confidences)), 2),
        "correct_predictions": {
            "count":            int(correct_mask.sum()),
            "mean_confidence":  round(float(confidences[correct_mask].mean()), 2) if correct_mask.any() else None,
            "median_confidence":round(float(np.median(confidences[correct_mask])), 2) if correct_mask.any() else None,
        },
        "incorrect_predictions": {
            "count":            int(incorrect_mask.sum()),
            "mean_confidence":  round(float(inc_confs.mean()), 2) if incorrect_mask.any() else None,
            "median_confidence":round(float(np.median(inc_confs)), 2) if incorrect_mask.any() else None,
            "pct_incorrect_with_confidence_ge_70": (
                round(float((inc_confs >= 70).mean() * 100), 2)
                if incorrect_mask.any() else 0.0
            ),
        },
    }


def classification_stats(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds   = probs.argmax(axis=1)
    top1    = float(accuracy_score(labels, preds))
    top3    = float(top_k_accuracy_score(labels, probs, k=3,
                                         labels=list(range(num_classes))))
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "top1_accuracy": round(top1, 4),
        "top3_accuracy": round(top3, 4),
        "macro_precision": round(float(p), 4),
        "macro_recall":    round(float(r), 4),
        "macro_f1":        round(float(f1), 4),
    }


# ── Reliability Diagram ───────────────────────────────────────────────────────
def reliability_diagram(raw_probs: np.ndarray, cal_probs: np.ndarray,
                         labels: np.ndarray, save_path: str,
                         n_bins: int = 10):
    """
    Plots confidence vs fraction-of-correct (reliability) for raw and
    calibrated probabilities side-by-side.
    """
    def _bin_data(probs, labels, n_bins):
        confs   = probs.max(axis=1)
        preds   = probs.argmax(axis=1)
        correct = (preds == labels).astype(float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_mids, bin_acc, bin_conf, bin_count = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0:
                continue
            bin_mids.append((lo + hi) / 2)
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confs[mask].mean())
            bin_count.append(mask.sum())
        return np.array(bin_mids), np.array(bin_acc), np.array(bin_conf), np.array(bin_count)

    raw_mids, raw_acc, raw_conf, raw_cnt = _bin_data(raw_probs, labels, n_bins)
    cal_mids, cal_acc, cal_conf, cal_cnt = _bin_data(cal_probs, labels, n_bins)

    raw_ece = compute_ece(raw_probs, labels, n_bins=15)
    cal_ece = compute_ece(cal_probs, labels, n_bins=15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, mids, acc, conf, cnt, title, ece in [
        (axes[0], raw_mids, raw_acc, raw_conf, raw_cnt,
         f"Raw Softmax\n(ECE={raw_ece:.4f})", raw_ece),
        (axes[1], cal_mids, cal_acc, cal_conf, cal_cnt,
         f"Temperature Scaled\n(ECE={cal_ece:.4f})", cal_ece),
    ]:
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", lw=1.5)
        ax.bar(mids, acc, width=0.08, alpha=0.6, color="steelblue",
               label="Fraction correct", edgecolor="navy", linewidth=0.5)
        ax.plot(conf, acc, "r^-", ms=6, label="Mean confidence", lw=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

    fig.suptitle(
        "CropGuard EfficientNet-B0 — Reliability Diagram\n"
        "(PlantVillage Test Set, 2735 images)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Reliability diagram saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  CropGuard Phase 2D — Temperature Scaling Calibration")
    print("=" * 68)
    print()

    # ── 0. Checkpoint integrity ──────────────────────────────────────────────
    BASELINE_SHA = "300a46ea8c2a3f9ff3f0e6fee0a9acdf19ecf5f8dc6b3e76ebe83c8a3b9d87d3"
    pre_sha = checkpoint_sha256(BEST_CKPT)
    print(f"[INTEGRITY] best_model.pt SHA256 (pre):  {pre_sha}")
    assert pre_sha == BASELINE_SHA, \
        f"best_model.pt has been modified! SHA mismatch.\nExpected: {BASELINE_SHA}\nGot:      {pre_sha}"
    print(f"[INTEGRITY] Checkpoint integrity: VERIFIED")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── 1. Load model (eval, frozen) ─────────────────────────────────────────
    raw_ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    sd = raw_ckpt["model_state_dict"] if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt else raw_ckpt

    model = build_model()
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    # Hard freeze — no grad on any parameter
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[INFO] Model loaded: EfficientNet-B0, {num_classes} classes, frozen")
    print()

    # ── 2. Collect validation logits (for fitting T) ─────────────────────────
    print("[STEP 1] Collecting validation logits...")
    val_ds = SplitDataset(VAL_DIR, transform=eval_transform)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    val_logits, val_labels = collect_logits(model, val_loader, device)
    print(f"  Validation images: {len(val_labels)}")

    # ── 3. Fit temperature on validation logits only ─────────────────────────
    print("[STEP 2] Fitting temperature T on validation set...")
    scaler = TemperatureScaler()
    T = scaler.fit(val_logits, val_labels)
    print(f"  Fitted temperature T = {T:.6f}")
    print()

    assert T > 0, f"Temperature must be positive, got {T}"

    # ── 4. Collect TEST logits (held-out, never used in fitting) ─────────────
    print("[STEP 3] Collecting test logits (held-out, used only for evaluation)...")
    test_ds = SplitDataset(TEST_DIR, transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    test_logits, test_labels = collect_logits(model, test_loader, device)
    print(f"  Test images: {len(test_labels)}")
    print()

    # ── 5. Compute raw and calibrated probabilities ───────────────────────────
    # Raw
    exp_raw = np.exp(test_logits - test_logits.max(axis=1, keepdims=True))
    raw_probs = exp_raw / exp_raw.sum(axis=1, keepdims=True)

    # Calibrated
    cal_probs = scaler.calibrated_probs(test_logits)

    # Sanity assertions on calibrated probs
    assert (cal_probs >= 0).all() and (cal_probs <= 1).all(), \
        "Calibrated probabilities outside [0, 1]"
    assert np.allclose(cal_probs.sum(axis=1), 1.0, atol=1e-5), \
        "Calibrated probabilities do not sum to 1"

    # ── 6. Compute all metrics ────────────────────────────────────────────────
    print("[STEP 4] Computing raw vs calibrated metrics on test set...")

    raw_clf   = classification_stats(raw_probs, test_labels)
    cal_clf   = classification_stats(cal_probs, test_labels)
    raw_conf  = confidence_stats(raw_probs, test_labels)
    cal_conf  = confidence_stats(cal_probs, test_labels)
    raw_ece   = compute_ece(raw_probs, test_labels)
    cal_ece   = compute_ece(cal_probs, test_labels)
    raw_brier = compute_brier(raw_probs, test_labels)
    cal_brier = compute_brier(cal_probs, test_labels)

    # Validation ECE for reference
    exp_val = np.exp(val_logits - val_logits.max(axis=1, keepdims=True))
    val_raw_probs = exp_val / exp_val.sum(axis=1, keepdims=True)
    val_cal_probs = scaler.calibrated_probs(val_logits)
    val_raw_ece = compute_ece(val_raw_probs, val_labels)
    val_cal_ece = compute_ece(val_cal_probs, val_labels)

    # ── 7. Print comparison table ─────────────────────────────────────────────
    print()
    print("=" * 68)
    print("  TEST SET RESULTS — Raw vs Temperature-Scaled")
    print("=" * 68)
    print(f"  {'Metric':<45} {'RAW':>8}  {'CALIB':>8}  {'DELTA':>8}")
    print(f"  {'-'*45} {'-'*8}  {'-'*8}  {'-'*8}")

    rows = [
        ("Top-1 Accuracy",             raw_clf["top1_accuracy"],     cal_clf["top1_accuracy"]),
        ("Top-3 Accuracy",             raw_clf["top3_accuracy"],     cal_clf["top3_accuracy"]),
        ("Macro F1",                   raw_clf["macro_f1"],          cal_clf["macro_f1"]),
        ("Macro Precision",            raw_clf["macro_precision"],   cal_clf["macro_precision"]),
        ("Macro Recall",               raw_clf["macro_recall"],      cal_clf["macro_recall"]),
        ("ECE (15 bins)",              raw_ece,                      cal_ece),
        ("Brier Score",                raw_brier,                    cal_brier),
        ("Mean Confidence (%)",        raw_conf["mean_confidence"],  cal_conf["mean_confidence"]),
        ("Median Confidence (%)",      raw_conf["median_confidence"],cal_conf["median_confidence"]),
        ("Correct — Mean Conf (%)",    raw_conf["correct_predictions"]["mean_confidence"],
                                       cal_conf["correct_predictions"]["mean_confidence"]),
        ("Incorrect — Mean Conf (%)",  raw_conf["incorrect_predictions"]["mean_confidence"],
                                       cal_conf["incorrect_predictions"]["mean_confidence"]),
        ("Incorrect — % conf>=70",     raw_conf["incorrect_predictions"]["pct_incorrect_with_confidence_ge_70"],
                                       cal_conf["incorrect_predictions"]["pct_incorrect_with_confidence_ge_70"]),
    ]
    for label, raw_val, cal_val in rows:
        delta = cal_val - raw_val if raw_val is not None and cal_val is not None else float("nan")
        delta_s = f"{delta:+.4f}" if not (delta != delta) else "   N/A"
        print(f"  {label:<45} {raw_val:>8.4f}  {cal_val:>8.4f}  {delta_s:>8}")

    print()
    print(f"  Fitted temperature T = {T:.6f}")
    print(f"  Validation ECE (raw):  {val_raw_ece:.6f}")
    print(f"  Validation ECE (cal):  {val_cal_ece:.6f}")
    print()

    # ── 8. Generate reliability diagram ──────────────────────────────────────
    diagram_path = os.path.join(REPORTS_DIR, "reliability_diagram.png")
    reliability_diagram(raw_probs, cal_probs, test_labels, diagram_path)

    # ── 9. Save calibration config ────────────────────────────────────────────
    config = {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "calibration_method": "Temperature Scaling (single scalar T, NLL-minimised via LBFGS)",
        "fitted_temperature": round(T, 6),
        "temperature_positive": T > 0,
        "fitting_split":      "validation",
        "evaluation_split":   "test",
        "validation_images":  int(len(val_labels)),
        "test_images":        int(len(test_labels)),
        "random_seed":        RANDOM_SEED,
        "checkpoint":         BEST_CKPT,
        "checkpoint_sha256":  pre_sha,
        "weights_modified":   False,
        "production_changed": False,
        "classifier_mode":    "mock",
        "optimizer":          "LBFGS(lr=0.01, max_iter=100)",
        "validation_ece_raw": round(val_raw_ece, 6),
        "validation_ece_cal": round(val_cal_ece, 6),
    }
    config_path = os.path.join(REPORTS_DIR, "calibration_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ── 10. Save full calibration report (JSON) ───────────────────────────────
    high_conf_err_raw = raw_conf["incorrect_predictions"]["pct_incorrect_with_confidence_ge_70"]
    high_conf_err_cal = cal_conf["incorrect_predictions"]["pct_incorrect_with_confidence_ge_70"]
    high_conf_improved = high_conf_err_cal < high_conf_err_raw

    report = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "phase":           "Phase 2D — Confidence Calibration",
        "model":           "CropGuard EfficientNet-B0",
        "checkpoint":      BEST_CKPT,
        "checkpoint_sha256": pre_sha,
        "calibration": {
            "method":      "Temperature Scaling",
            "fitted_T":    round(T, 6),
            "fit_on":      f"validation ({len(val_labels)} images)",
            "val_ece_raw": round(val_raw_ece, 6),
            "val_ece_cal": round(val_cal_ece, 6),
        },
        "test_set": {
            "images":      int(len(test_labels)),
            "raw": {
                **raw_clf,
                "ece":   round(raw_ece, 6),
                "brier": round(raw_brier, 6),
                "confidence": raw_conf,
            },
            "calibrated": {
                **cal_clf,
                "ece":   round(cal_ece, 6),
                "brier": round(cal_brier, 6),
                "confidence": cal_conf,
            },
            "delta_ece":   round(cal_ece - raw_ece, 6),
            "delta_brier": round(cal_brier - raw_brier, 6),
            "high_conf_error_rate": {
                "raw_pct":      high_conf_err_raw,
                "calibrated_pct": high_conf_err_cal,
                "improved":     high_conf_improved,
            },
        },
        "production_impact": {
            "real_classifier_modified": False,
            "referral_threshold_changed": False,
            "classifier_mode": "mock",
            "recommendation": (
                "T > 1 indicates the model is over-confident. Temperature scaling "
                "reduces confidence scores closer to empirical accuracy. "
                "Consider applying T in production if high-confidence error rate "
                "is reduced meaningfully and OOD evaluation is performed."
                if T > 1 else
                "T < 1 indicates the model is slightly under-confident. "
                "Temperature scaling increases confidence scores. "
                "Monitor the high-confidence error rate before production use."
            ),
        },
        "calibration_scope_note": (
            "Calibration improves the reliability of confidence estimates on the "
            "evaluated distribution (PlantVillage test set); it does NOT establish "
            "field accuracy or OOD performance."
        ),
        "weights_modified": False,
    }
    report_path = os.path.join(REPORTS_DIR, "calibration_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ── 11. Save human-readable Markdown report ───────────────────────────────
    inc_raw_n = raw_conf["incorrect_predictions"]["count"]
    inc_raw_n_ge70 = round(inc_raw_n * high_conf_err_raw / 100)
    inc_cal_n_ge70 = round(inc_raw_n * high_conf_err_cal / 100)

    md = f"""# CropGuard Phase 2D — Confidence Calibration Report

**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Model**: CropGuard EfficientNet-B0
**Checkpoint**: `ml/models/checkpoints/best_model.pt`
**SHA256**: `{pre_sha}`
**Weights modified**: NO
**Production changed**: NO (`CLASSIFIER_MODE=mock`)

---

## Method

**Temperature Scaling** — a single scalar parameter T is fit by minimising
Negative Log-Likelihood (NLL) on the validation set logits using LBFGS.

```
calibrated_logits = raw_logits / T
calibrated_probs  = softmax(calibrated_logits)
```

| Parameter | Value |
|---|---|
| Fitted temperature T | **{T:.6f}** |
| T > 1 (over-confident model) | **{T > 1}** |
| Fit on | Validation split ({len(val_labels):,} images) |
| Evaluated on | Test split ({len(test_labels):,} images) |
| Optimiser | LBFGS (lr=0.01, max_iter=100) |

---

## Test Set Results — Raw vs Temperature-Scaled

| Metric | Raw Softmax | Temperature Scaled | Delta |
|---|---|---|---|
| Top-1 Accuracy | {raw_clf['top1_accuracy']*100:.2f}% | {cal_clf['top1_accuracy']*100:.2f}% | {(cal_clf['top1_accuracy']-raw_clf['top1_accuracy'])*100:+.4f}% |
| Top-3 Accuracy | {raw_clf['top3_accuracy']*100:.2f}% | {cal_clf['top3_accuracy']*100:.2f}% | {(cal_clf['top3_accuracy']-raw_clf['top3_accuracy'])*100:+.4f}% |
| Macro F1 | {raw_clf['macro_f1']:.4f} | {cal_clf['macro_f1']:.4f} | {cal_clf['macro_f1']-raw_clf['macro_f1']:+.4f} |
| ECE (15 bins) | {raw_ece:.4f} | {cal_ece:.4f} | {cal_ece-raw_ece:+.4f} |
| Brier Score | {raw_brier:.4f} | {cal_brier:.4f} | {cal_brier-raw_brier:+.4f} |

### Confidence Analysis

| Metric | Raw Softmax | Temperature Scaled | Delta |
|---|---|---|---|
| Mean confidence | {raw_conf['mean_confidence']:.2f}% | {cal_conf['mean_confidence']:.2f}% | {cal_conf['mean_confidence']-raw_conf['mean_confidence']:+.2f}% |
| Median confidence | {raw_conf['median_confidence']:.2f}% | {cal_conf['median_confidence']:.2f}% | {cal_conf['median_confidence']-raw_conf['median_confidence']:+.2f}% |
| Correct — mean conf | {raw_conf['correct_predictions']['mean_confidence']:.2f}% | {cal_conf['correct_predictions']['mean_confidence']:.2f}% | {cal_conf['correct_predictions']['mean_confidence']-raw_conf['correct_predictions']['mean_confidence']:+.2f}% |
| Incorrect — mean conf | {raw_conf['incorrect_predictions']['mean_confidence']:.2f}% | {cal_conf['incorrect_predictions']['mean_confidence']:.2f}% | {cal_conf['incorrect_predictions']['mean_confidence']-raw_conf['incorrect_predictions']['mean_confidence']:+.2f}% |
| **Incorrect with conf ≥ 70%** | **{high_conf_err_raw:.2f}% ({inc_raw_n_ge70}/{inc_raw_n})** | **{high_conf_err_cal:.2f}% (~{inc_cal_n_ge70}/{inc_raw_n})** | **{high_conf_err_cal-high_conf_err_raw:+.2f}%** |

**High-confidence error rate improved**: `{high_conf_improved}`

---

## Calibration Quality (Validation Set)

| Metric | Value |
|---|---|
| Validation ECE — raw | {val_raw_ece:.6f} |
| Validation ECE — calibrated | {val_cal_ece:.6f} |

---

## Important Limitations

> **Calibration improves the reliability of confidence estimates on the evaluated
> distribution (PlantVillage test set); it does NOT establish field accuracy or
> OOD (out-of-distribution) performance.**

1. **Distribution**: Calibration was performed only on PlantVillage images.
   Confidence reliability on real field images (PlantDoc OOD) is unknown.
2. **Accuracy unchanged**: Temperature scaling does not change predictions,
   only the confidence scores. Top-1 and Top-3 accuracy are identical.
3. **No production change**: `RealClassifierService` still outputs raw softmax
   confidence. The fitted T is recorded in `calibration_config.json` for
   future integration consideration only.
4. **Threshold not changed**: The 70% officer-referral threshold remains in
   production unchanged.

## Production Impact

| Check | Status |
|---|---|
| `best_model.pt` modified | NO |
| `RealClassifierService` modified | NO |
| Referral threshold changed | NO |
| `CLASSIFIER_MODE` | `mock` |
"""

    md_path = os.path.join(REPORTS_DIR, "calibration_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[INFO] Calibration report (MD) saved:  {md_path}")
    print(f"[INFO] Calibration report (JSON) saved: {report_path}")
    print(f"[INFO] Calibration config saved:        {config_path}")

    # ── 12. Post-run checkpoint integrity re-check ────────────────────────────
    post_sha = checkpoint_sha256(BEST_CKPT)
    assert post_sha == BASELINE_SHA, \
        f"CRITICAL: best_model.pt SHA changed after calibration!\nPre: {pre_sha}\nPost: {post_sha}"
    print()
    print(f"[INTEGRITY] best_model.pt SHA256 (post): {post_sha}")
    print(f"[INTEGRITY] Checkpoint integrity: VERIFIED (unchanged)")

    print()
    print("=" * 68)
    print("  Phase 2D Calibration Complete")
    print("=" * 68)


if __name__ == "__main__":
    main()
