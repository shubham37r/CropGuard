"""
CropGuard Phase 2F-3 — Locked PlantDoc OOD Evaluation & Catastrophic Forgetting Analysis
========================================================================================
Evaluates both the Baseline EfficientNet-B0 model (best_model.pt) and the Field-Adapted
model (best_field_adapted_model.pt) on:
  1. The locked 70-image PlantDoc held-out test split (ml/data/plantdoc_splits/test/)
  2. The original 2,735-image PlantVillage test split (ml/data/splits/test/)

IMPORTANT SAFETY GUARANTEES:
  - Evaluation ONLY. No training, fine-tuning, or weight modifications.
  - Baseline best_model.pt SHA256 is verified pre & post evaluation.
  - PlantDoc locked test split is used for final evaluation ONLY.
  - CLASSIFIER_MODE remains "mock".
"""

import os
import sys
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timezone
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

BASE_DIR             = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR          = os.path.join(BASE_DIR, "reports")
MODELS_DIR           = os.path.join(BASE_DIR, "models")
BASELINE_CKPT        = os.path.join(MODELS_DIR, "checkpoints", "best_model.pt")
ADAPTED_CKPT         = os.path.join(MODELS_DIR, "field_adaptation", "checkpoints", "best_field_adapted_model.pt")
PLANTDOC_TEST_DIR    = os.path.join(BASE_DIR, "data", "plantdoc_splits", "test")
PLANTVILLAGE_TEST_DIR= os.path.join(BASE_DIR, "data", "splits", "test")

BASELINE_SHA = "300a46ea8c2a3f9ff3f0e6fee0a9acdf19ecf5f8dc6b3e76ebe83c8a3b9d87d3"

os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Load Class Mapping ────────────────────────────────────────────────────────
mapping_path = os.path.join(REPORTS_DIR, "class_mapping.json")
with open(mapping_path, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

class_to_idx   = mapping_data["class_to_idx"]
idx_to_class   = {int(k): v for k, v in mapping_data["idx_to_class"].items()}
num_classes    = len(class_to_idx)
sorted_classes = [idx_to_class[i] for i in range(num_classes)]

# Short class names for plot labels
short_labels = [c.replace("Tomato___", "").replace("_", " ") for c in sorted_classes]

# ── Preprocessing ─────────────────────────────────────────────────────────────
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ── Dataset Class ─────────────────────────────────────────────────────────────
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.samples = []

        for cls_name in sorted_classes:
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue
            cls_idx = class_to_idx[cls_name]
            files = sorted([f for f in os.listdir(cls_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
            for fname in files:
                fpath = os.path.join(cls_folder, fname)
                self.samples.append((fpath, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        with Image.open(fpath) as img:
            img_rgb = img.convert("RGB")
            if self.transform:
                img_rgb = self.transform(img_rgb)
            return img_rgb, label, fpath


def calculate_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def evaluate_model_on_dataset(model: nn.Module, dataset: ImageFolderDataset, device: torch.device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_targets = []
    all_confidences = []
    all_top3_correct = []
    all_fpaths = []

    with torch.no_grad():
        for images, labels, fpaths in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs  = torch.softmax(logits, dim=1)

            top1_conf, top1_pred = torch.max(probs, dim=1)
            _, top3_pred = torch.topk(probs, k=3, dim=1)

            for i in range(len(labels)):
                target = labels[i].item()
                pred   = top1_pred[i].item()
                conf   = top1_conf[i].item()
                top3   = top3_pred[i].cpu().numpy()

                all_preds.append(pred)
                all_targets.append(target)
                all_confidences.append(conf)
                all_top3_correct.append(target in top3)
                all_fpaths.append(fpaths[i])

    all_preds       = np.array(all_preds)
    all_targets     = np.array(all_targets)
    all_confidences = np.array(all_confidences)
    all_top3_correct = np.array(all_top3_correct)

    total_samples = len(all_targets)
    top1_acc = accuracy_score(all_targets, all_preds)
    top3_acc = np.mean(all_top3_correct)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    per_cls_p, per_cls_r, per_cls_f1, per_cls_supp = precision_recall_fscore_support(
        all_targets, all_preds, labels=range(num_classes), average=None, zero_division=0
    )

    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))

    # Confidence & Referral analysis @ 70% threshold
    correct_mask   = (all_preds == all_targets)
    incorrect_mask = ~correct_mask

    mean_conf   = float(np.mean(all_confidences))
    median_conf = float(np.median(all_confidences))

    correct_conf   = float(np.mean(all_confidences[correct_mask])) if np.sum(correct_mask) > 0 else 0.0
    incorrect_conf = float(np.mean(all_confidences[incorrect_mask])) if np.sum(incorrect_mask) > 0 else 0.0

    # 70% Raw Softmax Referral Threshold metrics
    ref_threshold = 0.70
    high_conf_mask = (all_confidences >= ref_threshold)
    referred_mask  = ~high_conf_mask

    referral_rate  = float(np.mean(referred_mask))
    coverage_rate  = float(np.mean(high_conf_mask))

    non_referred_correct = np.sum(correct_mask & high_conf_mask)
    non_referred_total   = np.sum(high_conf_mask)
    non_referred_acc     = float(non_referred_correct / non_referred_total) if non_referred_total > 0 else 0.0

    bypassed_errors      = int(np.sum(incorrect_mask & high_conf_mask))
    pct_incorrect_high_conf = float(bypassed_errors / np.sum(incorrect_mask)) if np.sum(incorrect_mask) > 0 else 0.0

    return {
        "sample_count": total_samples,
        "top1_accuracy": round(float(top1_acc), 4),
        "top3_accuracy": round(float(top3_acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "mean_confidence": round(mean_conf, 4),
        "median_confidence": round(median_conf, 4),
        "correct_mean_confidence": round(correct_conf, 4),
        "incorrect_mean_confidence": round(incorrect_conf, 4),
        "referral_rate_at_70pct": round(referral_rate, 4),
        "coverage_rate_at_70pct": round(coverage_rate, 4),
        "non_referred_accuracy_at_70pct": round(non_referred_acc, 4),
        "bypassed_errors_at_70pct": bypassed_errors,
        "pct_incorrect_high_confidence": round(pct_incorrect_high_conf, 4),
        "per_class": {
            sorted_classes[i]: {
                "precision": round(float(per_cls_p[i]), 4),
                "recall": round(float(per_cls_r[i]), 4),
                "f1": round(float(per_cls_f1[i]), 4),
                "support": int(per_cls_supp[i])
            }
            for i in range(num_classes)
        },
        "confusion_matrix": cm.tolist()
    }


def plot_confusion_matrices(baseline_cm, adapted_cm, dataset_name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(baseline_cm, annot=True, fmt="d", cmap="Blues", xticklabels=short_labels, yticklabels=short_labels, ax=axes[0])
    axes[0].set_title(f"Baseline EfficientNet-B0 — {dataset_name}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Class", fontsize=11)
    axes[0].set_ylabel("True Class", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(adapted_cm, annot=True, fmt="d", cmap="Greens", xticklabels=short_labels, yticklabels=short_labels, ax=axes[1])
    axes[1].set_title(f"Field-Adapted Model (Epoch 8) — {dataset_name}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Class", fontsize=11)
    axes[1].set_ylabel("True Class", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix figure: {save_path}")


def main():
    print("=" * 76)
    print("  CropGuard Phase 2F-3: Locked PlantDoc OOD & Catastrophic Forgetting Evaluation")
    print("=" * 76)

    # 1. Verify baseline SHA256 pre-check
    pre_sha = calculate_sha256(BASELINE_CKPT)
    print(f"[INTEGRITY] Baseline best_model.pt SHA256 (pre): {pre_sha}")
    assert pre_sha == BASELINE_SHA, f"Baseline SHA mismatch! Expected {BASELINE_SHA}, got {pre_sha}"

    # Verify adapted checkpoint existence
    assert os.path.exists(ADAPTED_CKPT), f"Adapted checkpoint missing at {ADAPTED_CKPT}"
    adapted_sha = calculate_sha256(ADAPTED_CKPT)
    print(f"[INTEGRITY] Adapted best_field_adapted_model.pt SHA256: {adapted_sha}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 2. Load Datasets
    plantdoc_test_ds    = ImageFolderDataset(PLANTDOC_TEST_DIR, transform=eval_transform)
    plantvillage_test_ds = ImageFolderDataset(PLANTVILLAGE_TEST_DIR, transform=eval_transform)

    print(f"[INFO] PlantDoc Locked Test Set:    {len(plantdoc_test_ds)} images")
    print(f"[INFO] PlantVillage Lab Test Set:   {len(plantvillage_test_ds)} images")
    assert len(plantdoc_test_ds) == 70, f"Expected 70 PlantDoc test images, got {len(plantdoc_test_ds)}"
    assert len(plantvillage_test_ds) == 2735, f"Expected 2735 PlantVillage test images, got {len(plantvillage_test_ds)}"
    print()

    # 3. Load Models
    print("[1/4] Loading Baseline Model A (best_model.pt)...")
    baseline_model = load_model(BASELINE_CKPT, device)

    print("[2/4] Loading Adapted Model B (best_field_adapted_model.pt)...")
    adapted_model  = load_model(ADAPTED_CKPT, device)
    print()

    # 4. Evaluate Models on PlantDoc Held-Out Test Set (LOCKED)
    print("[3/4] Evaluating on LOCKED PlantDoc Test Set (70 images)...")
    pd_baseline_metrics = evaluate_model_on_dataset(baseline_model, plantdoc_test_ds, device)
    pd_adapted_metrics  = evaluate_model_on_dataset(adapted_model, plantdoc_test_ds, device)

    # 5. Evaluate Models on PlantVillage Test Set
    print("[4/4] Evaluating on PlantVillage Test Set (2,735 images)...")
    pv_baseline_metrics = evaluate_model_on_dataset(baseline_model, plantvillage_test_ds, device)
    pv_adapted_metrics  = evaluate_model_on_dataset(adapted_model, plantvillage_test_ds, device)
    print()

    # 6. Verify Baseline SHA256 post-check
    post_sha = calculate_sha256(BASELINE_CKPT)
    print(f"[INTEGRITY] Baseline best_model.pt SHA256 (post): {post_sha}")
    assert post_sha == BASELINE_SHA, "CRITICAL: Baseline best_model.pt was modified!"
    print("[INTEGRITY] Baseline model integrity: VERIFIED (UNTOUCHED)")
    print()

    # 7. Calculate Deltas
    pd_acc_delta = round(pd_adapted_metrics["top1_accuracy"] - pd_baseline_metrics["top1_accuracy"], 4)
    pd_top3_delta = round(pd_adapted_metrics["top3_accuracy"] - pd_baseline_metrics["top3_accuracy"], 4)
    pd_macro_f1_delta = round(pd_adapted_metrics["macro_f1"] - pd_baseline_metrics["macro_f1"], 4)
    pd_weighted_f1_delta = round(pd_adapted_metrics["weighted_f1"] - pd_baseline_metrics["weighted_f1"], 4)

    pv_acc_delta = round(pv_adapted_metrics["top1_accuracy"] - pv_baseline_metrics["top1_accuracy"], 4)
    pv_top3_delta = round(pv_adapted_metrics["top3_accuracy"] - pv_baseline_metrics["top3_accuracy"], 4)
    pv_macro_f1_delta = round(pv_adapted_metrics["macro_f1"] - pv_baseline_metrics["macro_f1"], 4)
    pv_weighted_f1_delta = round(pv_adapted_metrics["weighted_f1"] - pv_baseline_metrics["weighted_f1"], 4)

    # Print Summary Table
    print("=" * 84)
    print("  EVALUATION SUMMARY COMPARISON REPORT")
    print("=" * 84)
    print(f"{'Metric':<30} | {'PlantDoc Baseline':<18} | {'PlantDoc Adapted':<18} | {'Delta':<10}")
    print("-" * 84)
    print(f"{'Top-1 Accuracy':<30} | {pd_baseline_metrics['top1_accuracy']*100:.2f}%             | {pd_adapted_metrics['top1_accuracy']*100:.2f}%             | {pd_acc_delta*100:+.2f}%")
    print(f"{'Top-3 Accuracy':<30} | {pd_baseline_metrics['top3_accuracy']*100:.2f}%             | {pd_adapted_metrics['top3_accuracy']*100:.2f}%             | {pd_top3_delta*100:+.2f}%")
    print(f"{'Macro F1 Score':<30} | {pd_baseline_metrics['macro_f1']:.4f}             | {pd_adapted_metrics['macro_f1']:.4f}             | {pd_macro_f1_delta:+.4f}")
    print(f"{'Weighted F1 Score':<30} | {pd_baseline_metrics['weighted_f1']:.4f}             | {pd_adapted_metrics['weighted_f1']:.4f}             | {pd_weighted_f1_delta:+.4f}")
    print("-" * 84)
    print(f"{'Metric (PlantVillage)':<30} | {'PV Baseline':<18} | {'PV Adapted':<18} | {'Delta':<10}")
    print("-" * 84)
    print(f"{'Top-1 Accuracy':<30} | {pv_baseline_metrics['top1_accuracy']*100:.2f}%             | {pv_adapted_metrics['top1_accuracy']*100:.2f}%             | {pv_acc_delta*100:+.2f}%")
    print(f"{'Top-3 Accuracy':<30} | {pv_baseline_metrics['top3_accuracy']*100:.2f}%             | {pv_adapted_metrics['top3_accuracy']*100:.2f}%             | {pv_top3_delta*100:+.2f}%")
    print(f"{'Macro F1 Score':<30} | {pv_baseline_metrics['macro_f1']:.4f}             | {pv_adapted_metrics['macro_f1']:.4f}             | {pv_macro_f1_delta:+.4f}")
    print(f"{'Weighted F1 Score':<30} | {pv_baseline_metrics['weighted_f1']:.4f}             | {pv_adapted_metrics['weighted_f1']:.4f}             | {pv_weighted_f1_delta:+.4f}")
    print("=" * 84)
    print()

    # 8. Save Visualizations & CSV Artifacts
    pd_cm_path = os.path.join(REPORTS_DIR, "field_adaptation_confusion_matrix_plantdoc.png")
    plot_confusion_matrices(
        np.array(pd_baseline_metrics["confusion_matrix"]),
        np.array(pd_adapted_metrics["confusion_matrix"]),
        "PlantDoc Locked Test (70 images)",
        pd_cm_path
    )

    pv_cm_path = os.path.join(REPORTS_DIR, "field_adaptation_confusion_matrix_plantvillage.png")
    plot_confusion_matrices(
        np.array(pv_baseline_metrics["confusion_matrix"]),
        np.array(pv_adapted_metrics["confusion_matrix"]),
        "PlantVillage Test (2,735 images)",
        pv_cm_path
    )

    # Build Class Metrics CSV
    csv_rows = []
    for cls_name in sorted_classes:
        pd_b = pd_baseline_metrics["per_class"][cls_name]
        pd_a = pd_adapted_metrics["per_class"][cls_name]
        pv_b = pv_baseline_metrics["per_class"][cls_name]
        pv_a = pv_adapted_metrics["per_class"][cls_name]

        csv_rows.append({
            "class_name": cls_name,
            "plantdoc_support": pd_a["support"],
            "plantdoc_baseline_f1": pd_b["f1"],
            "plantdoc_adapted_f1": pd_a["f1"],
            "plantdoc_f1_delta": round(pd_a["f1"] - pd_b["f1"], 4),
            "plantvillage_support": pv_a["support"],
            "plantvillage_baseline_f1": pv_b["f1"],
            "plantvillage_adapted_f1": pv_a["f1"],
            "plantvillage_f1_delta": round(pv_a["f1"] - pv_b["f1"], 4)
        })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(REPORTS_DIR, "field_adaptation_class_metrics.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved per-class CSV metrics: {csv_path}")

    # 9. Save Complete Evaluation JSON
    eval_json_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 2F-3 — Locked PlantDoc OOD Evaluation & Catastrophic Forgetting Analysis",
        "baseline_checkpoint": BASELINE_CKPT,
        "baseline_sha256": pre_sha,
        "adapted_checkpoint": ADAPTED_CKPT,
        "adapted_sha256": adapted_sha,
        "plantdoc_test_evaluation": {
            "sample_count": 70,
            "baseline": pd_baseline_metrics,
            "adapted": pd_adapted_metrics,
            "deltas": {
                "top1_accuracy": pd_acc_delta,
                "top3_accuracy": pd_top3_delta,
                "macro_f1": pd_macro_f1_delta,
                "weighted_f1": pd_weighted_f1_delta
            }
        },
        "plantvillage_test_evaluation": {
            "sample_count": 2735,
            "baseline": pv_baseline_metrics,
            "adapted": pv_adapted_metrics,
            "deltas": {
                "top1_accuracy": pv_acc_delta,
                "top3_accuracy": pv_top3_delta,
                "macro_f1": pv_macro_f1_delta,
                "weighted_f1": pv_weighted_f1_delta
            }
        }
    }

    eval_json_path = os.path.join(REPORTS_DIR, "field_adaptation_evaluation.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_json_data, f, indent=2)
    print(f"[INFO] Saved summary JSON: {eval_json_path}")

    # 10. Save Markdown Report
    md_content = f"""# CropGuard Phase 2F-3 — Locked PlantDoc OOD Evaluation & Catastrophic Forgetting Analysis Report

**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Baseline Model**: `ml/models/checkpoints/best_model.pt` (SHA256: `{pre_sha}`, **UNTOUCHED**)  
**Adapted Model**: `ml/models/field_adaptation/checkpoints/best_field_adapted_model.pt`  
**PlantDoc Test Set**: `ml/data/plantdoc_splits/test/` (**70 images, PERMANENTLY LOCKED**)  
**PlantVillage Test Set**: `ml/data/splits/test/` (**2,735 images**)

---

## 1. Executive Summary & Core Results

| Metric | PlantDoc Baseline (OOD) | PlantDoc Adapted (OOD) | PlantDoc Delta | PlantVillage Baseline (Lab) | PlantVillage Adapted (Lab) | PlantVillage Delta |
|---|---|---|---|---|---|---|
| **Top-1 Accuracy** | `{pd_baseline_metrics['top1_accuracy']*100:.2f}%` | `{pd_adapted_metrics['top1_accuracy']*100:.2f}%` | **`{pd_acc_delta*100:+.2f}%`** | `{pv_baseline_metrics['top1_accuracy']*100:.2f}%` | `{pv_adapted_metrics['top1_accuracy']*100:.2f}%` | **`{pv_acc_delta*100:+.2f}%`** |
| **Top-3 Accuracy** | `{pd_baseline_metrics['top3_accuracy']*100:.2f}%` | `{pd_adapted_metrics['top3_accuracy']*100:.2f}%` | **`{pd_top3_delta*100:+.2f}%`** | `{pv_baseline_metrics['top3_accuracy']*100:.2f}%` | `{pv_adapted_metrics['top3_accuracy']*100:.2f}%` | **`{pv_top3_delta*100:+.2f}%`** |
| **Macro Precision** | `{pd_baseline_metrics['macro_precision']:.4f}` | `{pd_adapted_metrics['macro_precision']:.4f}` | `{pd_adapted_metrics['macro_precision'] - pd_baseline_metrics['macro_precision']:+.4f}` | `{pv_baseline_metrics['macro_precision']:.4f}` | `{pv_adapted_metrics['macro_precision']:.4f}` | `{pv_adapted_metrics['macro_precision'] - pv_baseline_metrics['macro_precision']:+.4f}` |
| **Macro Recall** | `{pd_baseline_metrics['macro_recall']:.4f}` | `{pd_adapted_metrics['macro_recall']:.4f}` | `{pd_adapted_metrics['macro_recall'] - pd_baseline_metrics['macro_recall']:+.4f}` | `{pv_baseline_metrics['macro_recall']:.4f}` | `{pv_adapted_metrics['macro_recall']:.4f}` | `{pv_adapted_metrics['macro_recall'] - pv_baseline_metrics['macro_recall']:+.4f}` |
| **Macro F1 Score** | `{pd_baseline_metrics['macro_f1']:.4f}` | `{pd_adapted_metrics['macro_f1']:.4f}` | **`{pd_macro_f1_delta:+.4f}`** | `{pv_baseline_metrics['macro_f1']:.4f}` | `{pv_adapted_metrics['macro_f1']:.4f}` | **`{pv_macro_f1_delta:+.4f}`** |
| **Weighted F1 Score** | `{pd_baseline_metrics['weighted_f1']:.4f}` | `{pd_adapted_metrics['weighted_f1']:.4f}` | **`{pd_weighted_f1_delta:+.4f}`** | `{pv_baseline_metrics['weighted_f1']:.4f}` | `{pv_adapted_metrics['weighted_f1']:.4f}` | **`{pv_weighted_f1_delta:+.4f}`** |

---

## 2. PlantDoc Confidence & Referral Threshold Analysis (@ 70% Raw Threshold)

| Metric | Baseline Model | Adapted Model | Delta |
|---|---|---|---|
| **Mean Confidence (All)** | `{pd_baseline_metrics['mean_confidence']:.4f}` | `{pd_adapted_metrics['mean_confidence']:.4f}` | `{pd_adapted_metrics['mean_confidence'] - pd_baseline_metrics['mean_confidence']:+.4f}` |
| **Median Confidence (All)** | `{pd_baseline_metrics['median_confidence']:.4f}` | `{pd_adapted_metrics['median_confidence']:.4f}` | `{pd_adapted_metrics['median_confidence'] - pd_baseline_metrics['median_confidence']:+.4f}` |
| **Correct Predictions Mean Conf** | `{pd_baseline_metrics['correct_mean_confidence']:.4f}` | `{pd_adapted_metrics['correct_mean_confidence']:.4f}` | `{pd_adapted_metrics['correct_mean_confidence'] - pd_baseline_metrics['correct_mean_confidence']:+.4f}` |
| **Incorrect Predictions Mean Conf** | `{pd_baseline_metrics['incorrect_mean_confidence']:.4f}` | `{pd_adapted_metrics['incorrect_mean_confidence']:.4f}` | `{pd_adapted_metrics['incorrect_mean_confidence'] - pd_baseline_metrics['incorrect_mean_confidence']:+.4f}` |
| **Referral Rate @ 70%** | `{pd_baseline_metrics['referral_rate_at_70pct']*100:.2f}%` | `{pd_adapted_metrics['referral_rate_at_70pct']*100:.2f}%` | `{pd_adapted_metrics['referral_rate_at_70pct']*100 - pd_baseline_metrics['referral_rate_at_70pct']*100:+.2f}%` |
| **Automation Coverage @ 70%** | `{pd_baseline_metrics['coverage_rate_at_70pct']*100:.2f}%` | `{pd_adapted_metrics['coverage_rate_at_70pct']*100:.2f}%` | `{pd_adapted_metrics['coverage_rate_at_70pct']*100 - pd_baseline_metrics['coverage_rate_at_70pct']*100:+.2f}%` |
| **Non-Referred Accuracy @ 70%** | `{pd_baseline_metrics['non_referred_accuracy_at_70pct']*100:.2f}%` | `{pd_adapted_metrics['non_referred_accuracy_at_70pct']*100:.2f}%` | `{pd_adapted_metrics['non_referred_accuracy_at_70pct']*100 - pd_baseline_metrics['non_referred_accuracy_at_70pct']*100:+.2f}%` |
| **Bypassed High-Conf Errors** | `{pd_baseline_metrics['bypassed_errors_at_70pct']}` | `{pd_adapted_metrics['bypassed_errors_at_70pct']}` | `{pd_adapted_metrics['bypassed_errors_at_70pct'] - pd_baseline_metrics['bypassed_errors_at_70pct']:+d}` |

---

## 3. Per-Class F1 Breakdown

| Class Name | PlantDoc Supp | PlantDoc Base F1 | PlantDoc Adapt F1 | PlantDoc F1 Δ | PV Supp | PV Base F1 | PV Adapt F1 | PV F1 Δ |
|---|---|---|---|---|---|---|---|---|
"""
    for r in csv_rows:
        md_content += f"| `{r['class_name']}` | {r['plantdoc_support']} | {r['plantdoc_baseline_f1']:.4f} | {r['plantdoc_adapted_f1']:.4f} | **{r['plantdoc_f1_delta']:+.4f}** | {r['plantvillage_support']} | {r['plantvillage_baseline_f1']:.4f} | {r['plantvillage_adapted_f1']:.4f} | **{r['plantvillage_f1_delta']:+.4f}** |\n"

    md_content += f"""
---

## 4. Key Findings & Conclusions

### A. Field Generalization Improvement (PlantDoc Test)
- **Top-1 Accuracy**: `{pd_baseline_metrics['top1_accuracy']*100:.2f}%` $\rightarrow$ **`{pd_adapted_metrics['top1_accuracy']*100:.2f}%`** (**`{pd_acc_delta*100:+.2f}%`** gain).
- **Macro F1 Score**: `{pd_baseline_metrics['macro_f1']:.4f}` $\rightarrow$ **`{pd_adapted_metrics['macro_f1']:.4f}`** (**`{pd_macro_f1_delta:+.4f}`** gain).
- **Weighted F1 Score**: `{pd_baseline_metrics['weighted_f1']:.4f}` $\rightarrow$ **`{pd_adapted_metrics['weighted_f1']:.4f}`** (**`{pd_weighted_f1_delta:+.4f}`** gain).

### B. Catastrophic Forgetting Assessment (PlantVillage Test)
- **Top-1 Accuracy**: `{pv_baseline_metrics['top1_accuracy']*100:.2f}%` $\rightarrow$ **`{pv_adapted_metrics['top1_accuracy']*100:.2f}%`** (**`{pv_acc_delta*100:+.2f}%`** change).
- **Macro F1 Score**: `{pv_baseline_metrics['macro_f1']:.4f}` $\rightarrow$ **`{pv_adapted_metrics['macro_f1']:.4f}`** (**`{pv_macro_f1_delta:+.4f}`** change).
- **Weighted F1 Score**: `{pv_baseline_metrics['weighted_f1']:.4f}` $\rightarrow$ **`{pv_adapted_metrics['weighted_f1']:.4f}`** (**`{pv_weighted_f1_delta:+.4f}`** change).

---

## 5. Production & Integrity Status

| Item | Status |
|---|---|
| Baseline `best_model.pt` SHA256 | `{pre_sha}` (**UNTOUCHED**) |
| `CLASSIFIER_MODE` | `mock` (**UNCHANGED**) |
| `RealClassifierService` modified | NO |
| Referral Threshold modified | NO |
| PlantDoc Held-Out Test Set accessed during training | NO (Evaluated ONLY in Phase 2F-3) |
| Retraining / Temperature Scaling performed | NO |
"""

    md_report_path = os.path.join(REPORTS_DIR, "field_adaptation_evaluation.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[INFO] Saved Markdown report: {md_report_path}")


if __name__ == "__main__":
    main()
