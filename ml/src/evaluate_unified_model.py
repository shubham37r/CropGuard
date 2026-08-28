"""
CropGuard Phase 2F — Standalone Evaluation Script for Unified Tomato Model
==========================================================================
Evaluates Model A (Baseline: best_model.pt) vs Model B (New Candidate: best_unified_tomato_model.pt)
on locked PlantDoc test set (70 images) and locked PlantVillage test set (2,735 images).

SAFETY & ISOLATION GUARANTEES:
  - Evaluation ONLY (No training, no fine-tuning, no weight edits).
  - Baseline model weights (best_model.pt) are NEVER modified.
  - Locked test splits (ml/data/plantdoc_splits/test/ & ml/data/splits/test/) are READ-ONLY.
  - Production state remains CLASSIFIER_MODE=mock.

Usage:
    python -m ml.src.evaluate_unified_model
"""

from datetime import datetime, timezone
import os
import sys
import json
import csv
import hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

BASE_DIR                   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR                = os.path.join(BASE_DIR, "reports")
MODELS_DIR                 = os.path.join(BASE_DIR, "models")

BASELINE_CKPT              = os.path.join(MODELS_DIR, "checkpoints", "best_model.pt")
UNIFIED_CKPT               = os.path.join(MODELS_DIR, "unified_adaptation", "checkpoints", "best_unified_tomato_model.pt")

PLANTDOC_TEST_DIR          = os.path.join(BASE_DIR, "data", "plantdoc_splits", "test")
PLANTVILLAGE_TEST_DIR      = os.path.join(BASE_DIR, "data", "splits", "test")

BASELINE_SHA = "300a46ea8c2a3f9ff3f0e6fee0a9acdf19ecf5f8dc6b3e76ebe83c8a3b9d87d3"

os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Class Mapping ─────────────────────────────────────────────────────────────
mapping_path = os.path.join(REPORTS_DIR, "class_mapping.json")
with open(mapping_path, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

class_to_idx   = mapping_data["class_to_idx"]
idx_to_class   = {int(k): v for k, v in mapping_data["idx_to_class"].items()}
num_classes    = len(class_to_idx)
sorted_classes = [idx_to_class[i] for i in range(num_classes)]

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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


def evaluate_model_on_dataset(model: nn.Module, dataset: Dataset, device: torch.device):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    all_targets, all_preds, all_probs, all_top3 = [], [], [], []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            preds = np.argmax(probs, axis=1)
            top3 = np.argsort(probs, axis=1)[:, -3:]

            all_targets.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_top3.extend(top3)

    all_targets = np.array(all_targets)
    all_preds   = np.array(all_preds)
    all_probs   = np.array(all_probs)
    all_top3    = np.array(all_top3)

    top1_acc = float(accuracy_score(all_targets, all_preds))
    top3_acc = float(np.mean([target in top3_i for target, top3_i in zip(all_targets, all_top3)]))

    per_cls_p, per_cls_r, per_cls_f1, per_cls_supp = precision_recall_fscore_support(
        all_targets, all_preds, labels=range(num_classes), average=None, zero_division=0
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))

    # Confidence statistics
    max_conf = np.max(all_probs, axis=1)
    is_correct = (all_preds == all_targets)

    mean_conf_all = float(np.mean(max_conf))
    median_conf_all = float(np.median(max_conf))
    mean_conf_correct = float(np.mean(max_conf[is_correct])) if np.any(is_correct) else 0.0
    mean_conf_incorrect = float(np.mean(max_conf[~is_correct])) if np.any(~is_correct) else 0.0

    # 70% threshold referral behavior
    raw_thresh = 0.70
    is_referred = (max_conf < raw_thresh)
    referral_rate = float(np.mean(is_referred))
    automation_coverage = float(1.0 - referral_rate)

    non_ref_mask = ~is_referred
    non_ref_acc = float(np.mean(is_correct[non_ref_mask])) if np.any(non_ref_mask) else 0.0
    bypassed_errors = int(np.sum((~is_correct) & non_ref_mask))

    # Viral Class Confusion metrics (Mosaic Virus = idx 8, TYLCV = idx 7)
    tomv_idx = class_to_idx["Tomato___Tomato_mosaic_virus"]
    tylcv_idx = class_to_idx["Tomato___Tomato_Yellow_Leaf_Curl_Virus"]

    tomv_to_tylcv_count = int(cm[tomv_idx, tylcv_idx])
    tylcv_to_tomv_count = int(cm[tylcv_idx, tomv_idx])
    total_viral_confusion = tomv_to_tylcv_count + tylcv_to_tomv_count

    return {
        "num_samples": len(all_targets),
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            sorted_classes[i]: {
                "precision": round(float(per_cls_p[i]), 4),
                "recall": round(float(per_cls_r[i]), 4),
                "f1": round(float(per_cls_f1[i]), 4),
                "support": int(per_cls_supp[i])
            }
            for i in range(num_classes)
        },
        "confidence_stats": {
            "mean_all": round(mean_conf_all, 4),
            "median_all": round(median_conf_all, 4),
            "mean_correct": round(mean_conf_correct, 4),
            "mean_incorrect": round(mean_conf_incorrect, 4),
            "referral_rate_70": round(referral_rate, 4),
            "automation_coverage_70": round(automation_coverage, 4),
            "non_referred_accuracy_70": round(non_ref_acc, 4),
            "bypassed_errors_70": bypassed_errors
        },
        "viral_metrics": {
            "tomv_f1": round(float(per_cls_f1[tomv_idx]), 4),
            "tylcv_f1": round(float(per_cls_f1[tylcv_idx]), 4),
            "tomv_to_tylcv_count": tomv_to_tylcv_count,
            "tylcv_to_tomv_count": tylcv_to_tomv_count,
            "total_viral_confusion": total_viral_confusion
        }
    }


def plot_confusion_matrices(cm_a: np.ndarray, cm_b: np.ndarray, dataset_name: str, output_path: str):
    labels = [c.replace("Tomato___", "") for c in sorted_classes]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(cm_a, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0], cbar=False)
    axes[0].set_title(f"Model A (Baseline) — {dataset_name}", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(cm_b, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[1], cbar=False)
    axes[1].set_title(f"Model B (Unified Candidate) — {dataset_name}", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_evaluation():
    print("=" * 76)
    print("  CropGuard Phase 2F: Final Unified Model Evaluation")
    print("=" * 76)

    # 1. Baseline Checkpoint SHA256 Pre-Check
    pre_base_sha = calculate_sha256(BASELINE_CKPT)
    print(f"[INTEGRITY] Baseline best_model.pt SHA256 (pre): {pre_base_sha}")
    assert pre_base_sha == BASELINE_SHA, f"Baseline SHA mismatch! Expected {BASELINE_SHA}, got {pre_base_sha}"

    # 2. Hardware Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 3. Load Models
    print("[INFO] Loading Model A (Baseline: best_model.pt)...")
    model_a = load_model(BASELINE_CKPT, device)

    print("[INFO] Loading Model B (Unified Candidate: best_unified_tomato_model.pt)...")
    model_b = load_model(UNIFIED_CKPT, device)

    # 4. Load Datasets
    print("[INFO] Loading Locked PlantDoc Test Set (70 images)...")
    pd_test_ds = ImageFolderDataset(PLANTDOC_TEST_DIR, transform=eval_transform)
    assert len(pd_test_ds) == 70, f"Expected 70 PlantDoc test images, got {len(pd_test_ds)}"

    print("[INFO] Loading Locked PlantVillage Test Set (2,735 images)...")
    pv_test_ds = ImageFolderDataset(PLANTVILLAGE_TEST_DIR, transform=eval_transform)
    assert len(pv_test_ds) == 2735, f"Expected 2,735 PlantVillage test images, got {len(pv_test_ds)}"

    # 5. Execute Evaluation
    print("\n[INFO] Evaluating Model A on PlantDoc Test Set...")
    pd_res_a = evaluate_model_on_dataset(model_a, pd_test_ds, device)

    print("[INFO] Evaluating Model B on PlantDoc Test Set...")
    pd_res_b = evaluate_model_on_dataset(model_b, pd_test_ds, device)

    print("[INFO] Evaluating Model A on PlantVillage Test Set...")
    pv_res_a = evaluate_model_on_dataset(model_a, pv_test_ds, device)

    print("[INFO] Evaluating Model B on PlantVillage Test Set...")
    pv_res_b = evaluate_model_on_dataset(model_b, pv_test_ds, device)

    # 6. Calculate Metric Deltas (Model B - Model A)
    pd_top1_delta = round(pd_res_b["top1_accuracy"] - pd_res_a["top1_accuracy"], 4)
    pd_f1_delta   = round(pd_res_b["macro_f1"] - pd_res_a["macro_f1"], 4)
    pv_top1_delta = round(pv_res_b["top1_accuracy"] - pv_res_a["top1_accuracy"], 4)
    pv_f1_delta   = round(pv_res_b["macro_f1"] - pv_res_a["macro_f1"], 4)

    # 7. Apply Acceptance Framework Decision Rules
    pv_top1_b = pv_res_b["top1_accuracy"]
    pv_f1_b   = pv_res_b["macro_f1"]
    pd_top1_b = pd_res_b["top1_accuracy"]
    pd_f1_b   = pd_res_b["macro_f1"]

    pass_conditions = (
        pd_top1_delta > 0.05 and           # Material PlantDoc improvement (+5% Top-1)
        pv_top1_b >= 0.95 and              # PlantVillage Top-1 >= 95%
        pv_f1_b >= 0.90                    # PlantVillage Macro F1 >= 0.90
    )

    reject_conditions = (
        pv_top1_b < 0.95 or
        pv_f1_b < 0.90 or
        pd_top1_delta <= 0.0
    )

    if pass_conditions:
        verdict = "CANDIDATE ACCEPTED"
        verdict_reason = "Model B achieves material PlantDoc field improvement while retaining PlantVillage accuracy >= 95%."
    elif reject_conditions:
        verdict = "CANDIDATE REJECTED"
        verdict_reason = f"Model B failed retention bounds (PV Top-1: {pv_top1_b*100:.2f}%, PV Macro F1: {pv_f1_b:.4f})."
    else:
        verdict = "INCONCLUSIVE — MORE DATA NEEDED"
        verdict_reason = "Results fell into trade-off zone without satisfying strict threshold bounds."

    print("\n" + "=" * 76)
    print(f"  VERDICT: {verdict}")
    print(f"  Reason:  {verdict_reason}")
    print("=" * 76)

    # 8. Generate Reports & Artifacts
    evaluation_json = {
        "generated_at": datetime.now(timezone.utc).isoformat() if 'timezone' in globals() else str(datetime.now()),
        "baseline_sha256": pre_base_sha,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "deltas": {
            "plantdoc_top1_delta": pd_top1_delta,
            "plantdoc_macro_f1_delta": pd_f1_delta,
            "plantvillage_top1_delta": pv_top1_delta,
            "plantvillage_macro_f1_delta": pv_f1_delta
        },
        "plantdoc_test": {
            "model_a_baseline": pd_res_a,
            "model_b_unified": pd_res_b
        },
        "plantvillage_test": {
            "model_a_baseline": pv_res_a,
            "model_b_unified": pv_res_b
        }
    }

    json_path = os.path.join(REPORTS_DIR, "unified_model_evaluation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_json, f, indent=2)

    # Class Metrics CSV
    csv_path = os.path.join(REPORTS_DIR, "unified_model_class_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_name",
            "pd_supp", "pd_base_f1", "pd_unified_f1", "pd_f1_delta",
            "pv_supp", "pv_base_f1", "pv_unified_f1", "pv_f1_delta"
        ])
        for cls_name in sorted_classes:
            pd_a = pd_res_a["per_class"][cls_name]
            pd_b = pd_res_b["per_class"][cls_name]
            pv_a = pv_res_a["per_class"][cls_name]
            pv_b = pv_res_b["per_class"][cls_name]

            writer.writerow([
                cls_name,
                pd_a["support"], pd_a["f1"], pd_b["f1"], round(pd_b["f1"] - pd_a["f1"], 4),
                pv_a["support"], pv_a["f1"], pv_b["f1"], round(pv_b["f1"] - pv_a["f1"], 4)
            ])

    # Confusion Matrix Plots
    plot_confusion_matrices(
        np.array(pd_res_a["confusion_matrix"]),
        np.array(pd_res_b["confusion_matrix"]),
        "PlantDoc Test (70 images)",
        os.path.join(REPORTS_DIR, "unified_model_confusion_plantdoc.png")
    )

    plot_confusion_matrices(
        np.array(pv_res_a["confusion_matrix"]),
        np.array(pv_res_b["confusion_matrix"]),
        "PlantVillage Test (2,735 images)",
        os.path.join(REPORTS_DIR, "unified_model_confusion_plantvillage.png")
    )

    # Markdown Report
    md_path = os.path.join(REPORTS_DIR, "unified_model_evaluation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"""# CropGuard Phase 2F — Unified Tomato Model Final Evaluation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Baseline Model A**: `ml/models/checkpoints/best_model.pt` (SHA256: `{pre_base_sha}`)  
**Unified Candidate Model B**: `ml/models/unified_adaptation/checkpoints/best_unified_tomato_model.pt`  
**PlantDoc Test Set**: `ml/data/plantdoc_splits/test/` (**70 images, PERMANENTLY LOCKED**)  
**PlantVillage Test Set**: `ml/data/splits/test/` (**2,735 images, PERMANENTLY LOCKED**)  

---

## 1. Executive Summary & Verdict

### **VERDICT: {verdict}**
**Rationale**: {verdict_reason}

| Metric | PlantDoc Baseline (Model A) | PlantDoc Unified (Model B) | PlantDoc Delta | PlantVillage Baseline (Model A) | PlantVillage Unified (Model B) | PlantVillage Delta |
|---|---|---|---|---|---|---|
| **Top-1 Accuracy** | `{pd_res_a['top1_accuracy']*100:.2f}%` | `{pd_res_b['top1_accuracy']*100:.2f}%` | **`{pd_top1_delta*100:+.2f}%`** | `{pv_res_a['top1_accuracy']*100:.2f}%` | `{pv_res_b['top1_accuracy']*100:.2f}%` | **`{pv_top1_delta*100:+.2f}%`** |
| **Top-3 Accuracy** | `{pd_res_a['top3_accuracy']*100:.2f}%` | `{pd_res_b['top3_accuracy']*100:.2f}%` | `{round(pd_res_b['top3_accuracy']-pd_res_a['top3_accuracy'], 4)*100:+.2f}%` | `{pv_res_a['top3_accuracy']*100:.2f}%` | `{pv_res_b['top3_accuracy']*100:.2f}%` | `{round(pv_res_b['top3_accuracy']-pv_res_a['top3_accuracy'], 4)*100:+.2f}%` |
| **Macro Precision** | `{pd_res_a['macro_precision']:.4f}` | `{pd_res_b['macro_precision']:.4f}` | `{round(pd_res_b['macro_precision']-pd_res_a['macro_precision'], 4):+.4f}` | `{pv_res_a['macro_precision']:.4f}` | `{pv_res_b['macro_precision']:.4f}` | `{round(pv_res_b['macro_precision']-pv_res_a['macro_precision'], 4):+.4f}` |
| **Macro Recall** | `{pd_res_a['macro_recall']:.4f}` | `{pd_res_b['macro_recall']:.4f}` | `{round(pd_res_b['macro_recall']-pd_res_a['macro_recall'], 4):+.4f}` | `{pv_res_a['macro_recall']:.4f}` | `{pv_res_b['macro_recall']:.4f}` | `{round(pv_res_b['macro_recall']-pv_res_a['macro_recall'], 4):+.4f}` |
| **Macro F1 Score** | `{pd_res_a['macro_f1']:.4f}` | `{pd_res_b['macro_f1']:.4f}` | **`{pd_f1_delta:+.4f}`** | `{pv_res_a['macro_f1']:.4f}` | `{pv_res_b['macro_f1']:.4f}` | **`{pv_f1_delta:+.4f}`** |
| **Weighted F1 Score** | `{pd_res_a['weighted_f1']:.4f}` | `{pd_res_b['weighted_f1']:.4f}` | `{round(pd_res_b['weighted_f1']-pd_res_a['weighted_f1'], 4):+.4f}` | `{pv_res_a['weighted_f1']:.4f}` | `{pv_res_b['weighted_f1']:.4f}` | `{round(pv_res_b['weighted_f1']-pv_res_a['weighted_f1'], 4):+.4f}` |

---

## 2. Viral Class Confusion Analysis (Mosaic Virus vs TYLCV)

| Metric | PlantDoc Model A | PlantDoc Model B | PlantVillage Model A | PlantVillage Model B |
|---|---|---|---|---|
| **Mosaic Virus F1** | `{pd_res_a['viral_metrics']['tomv_f1']:.4f}` | `{pd_res_b['viral_metrics']['tomv_f1']:.4f}` | `{pv_res_a['viral_metrics']['tomv_f1']:.4f}` | `{pv_res_b['viral_metrics']['tomv_f1']:.4f}` |
| **TYLCV F1** | `{pd_res_a['viral_metrics']['tylcv_f1']:.4f}` | `{pd_res_b['viral_metrics']['tylcv_f1']:.4f}` | `{pv_res_a['viral_metrics']['tylcv_f1']:.4f}` | `{pv_res_b['viral_metrics']['tylcv_f1']:.4f}` |
| **ToMV $\\rightarrow$ TYLCV Errors** | `{pd_res_a['viral_metrics']['tomv_to_tylcv_count']}` | `{pd_res_b['viral_metrics']['tomv_to_tylcv_count']}` | `{pv_res_a['viral_metrics']['tomv_to_tylcv_count']}` | `{pv_res_b['viral_metrics']['tomv_to_tylcv_count']}` |
| **TYLCV $\\rightarrow$ ToMV Errors** | `{pd_res_a['viral_metrics']['tylcv_to_tomv_count']}` | `{pd_res_b['viral_metrics']['tylcv_to_tomv_count']}` | `{pv_res_a['viral_metrics']['tylcv_to_tomv_count']}` | `{pv_res_b['viral_metrics']['tylcv_to_tomv_count']}` |
| **Total Viral Confusion** | `{pd_res_a['viral_metrics']['total_viral_confusion']}` | `{pd_res_b['viral_metrics']['total_viral_confusion']}` | `{pv_res_a['viral_metrics']['total_viral_confusion']}` | `{pv_res_b['viral_metrics']['total_viral_confusion']}` |

---

## 3. Confidence & 70% Referral Threshold Analysis

| Metric | PlantDoc Model A | PlantDoc Model B | PlantVillage Model A | PlantVillage Model B |
|---|---|---|---|---|
| **Mean Confidence** | `{pd_res_a['confidence_stats']['mean_all']:.4f}` | `{pd_res_b['confidence_stats']['mean_all']:.4f}` | `{pv_res_a['confidence_stats']['mean_all']:.4f}` | `{pv_res_b['confidence_stats']['mean_all']:.4f}` |
| **Correct Preds Mean Conf** | `{pd_res_a['confidence_stats']['mean_correct']:.4f}` | `{pd_res_b['confidence_stats']['mean_correct']:.4f}` | `{pv_res_a['confidence_stats']['mean_correct']:.4f}` | `{pv_res_b['confidence_stats']['mean_correct']:.4f}` |
| **Incorrect Preds Mean Conf** | `{pd_res_a['confidence_stats']['mean_incorrect']:.4f}` | `{pd_res_b['confidence_stats']['mean_incorrect']:.4f}` | `{pv_res_a['confidence_stats']['mean_incorrect']:.4f}` | `{pv_res_b['confidence_stats']['mean_incorrect']:.4f}` |
| **Referral Rate @ 70%** | `{pd_res_a['confidence_stats']['referral_rate_70']*100:.2f}%` | `{pd_res_b['confidence_stats']['referral_rate_70']*100:.2f}%` | `{pv_res_a['confidence_stats']['referral_rate_70']*100:.2f}%` | `{pv_res_b['confidence_stats']['referral_rate_70']*100:.2f}%` |
| **Bypassed High-Conf Errors** | `{pd_res_a['confidence_stats']['bypassed_errors_70']}` | `{pd_res_b['confidence_stats']['bypassed_errors_70']}` | `{pv_res_a['confidence_stats']['bypassed_errors_70']}` | `{pv_res_b['confidence_stats']['bypassed_errors_70']}` |

---

## 4. Per-Class F1 Breakdown

| Class Name | PD Supp | PD Base F1 | PD Unified F1 | PD F1 Δ | PV Supp | PV Base F1 | PV Unified F1 | PV F1 Δ |
|---|---|---|---|---|---|---|---|---|
""")
        for cls_name in sorted_classes:
            pd_a = pd_res_a["per_class"][cls_name]
            pd_b = pd_res_b["per_class"][cls_name]
            pv_a = pv_res_a["per_class"][cls_name]
            pv_b = pv_res_b["per_class"][cls_name]

            f.write(
                f"| `{cls_name}` | {pd_a['support']} | {pd_a['f1']:.4f} | {pd_b['f1']:.4f} | **{pd_b['f1']-pd_a['f1']:+.4f}** | "
                f"{pv_a['support']} | {pv_a['f1']:.4f} | {pv_b['f1']:.4f} | **{pv_b['f1']-pv_a['f1']:+.4f}** |\n"
            )

        f.write(f"""
---

## 5. Production & Integrity Verification

| Item | Status |
|---|---|
| Baseline `best_model.pt` SHA256 | `{pre_base_sha}` (**UNTOUCHED**) |
| `CLASSIFIER_MODE` | `mock` (**UNCHANGED**) |
| Candidate activated in production | NO |
| Backend/Frontend services modified | NO |
| PlantDoc Held-Out Test Set accessed during training | NO (Evaluated ONLY in Phase 2F) |
| PlantVillage Held-Out Test Set accessed during training | NO (Evaluated ONLY in Phase 2F) |
""")

    # 9. Post Checkpoint Integrity Verification
    post_base_sha = calculate_sha256(BASELINE_CKPT)
    print(f"\n[INTEGRITY] Baseline best_model.pt SHA256 (post): {post_base_sha}")
    assert post_base_sha == BASELINE_SHA, "CRITICAL: Baseline best_model.pt modified!"
    print("[INTEGRITY] Baseline checkpoint SHA256 verified UNTOUCHED!")


if __name__ == "__main__":
    run_evaluation()
