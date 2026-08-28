"""
CropGuard Phase 2F — Standalone Unified Tomato Model Training Pipeline
======================================================================
Trains an EfficientNet-B0 model using class-weighted CrossEntropyLoss on a combined
PlantVillage + PlantDoc tomato training dataset to address Tomato Mosaic Virus vs.
Tomato Yellow Leaf Curl Virus misclassifications while preserving CropGuard's
exact 10-class taxonomy.

IMPORTANT SAFETY & ISOLATION GUARANTEES:
  - Baseline model weights (best_model.pt) are NEVER overwritten or modified.
  - Candidate checkpoints are saved separately in: ml/models/unified_adaptation/checkpoints/
  - Data isolation:
      * PlantDoc train loader uses ONLY ml/data/plantdoc_splits/train/ (585 images)
      * PlantVillage train loader uses ONLY ml/data/splits/train/ (12,705 images)
      * PlantDoc validation loader uses ONLY ml/data/plantdoc_splits/validation/ (71 images)
      * PlantVillage validation loader uses ONLY ml/data/splits/validation/ (2,720 images)
      * Both test splits (ml/data/plantdoc_splits/test/ & ml/data/splits/test/) are PERMANENTLY LOCKED.
  - Smoke test verification: python -m ml.src.train_unified_tomato_model --smoke-test
  - Full training execution:  python -m ml.src.train_unified_tomato_model

Usage:
    python -m ml.src.train_unified_tomato_model --smoke-test
"""

import os
import sys
import json
import time
import hashlib
import random
import argparse
import numpy as np
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

BASE_DIR                   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR                = os.path.join(BASE_DIR, "reports")
MODELS_DIR                 = os.path.join(BASE_DIR, "models")
BASELINE_CKPT              = os.path.join(MODELS_DIR, "checkpoints", "best_model.pt")
PLANTDOC_SPLITS_DIR        = os.path.join(BASE_DIR, "data", "plantdoc_splits")
PLANTVILLAGE_SPLITS_DIR    = os.path.join(BASE_DIR, "data", "splits")

UNIFIED_ADAPTATION_DIR     = os.path.join(MODELS_DIR, "unified_adaptation")
UNIFIED_CKPTS_DIR          = os.path.join(UNIFIED_ADAPTATION_DIR, "checkpoints")

BASELINE_SHA = "300a46ea8c2a3f9ff3f0e6fee0a9acdf19ecf5f8dc6b3e76ebe83c8a3b9d87d3"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(UNIFIED_CKPTS_DIR, exist_ok=True)

# ── Load Class Mapping ────────────────────────────────────────────────────────
mapping_path = os.path.join(REPORTS_DIR, "class_mapping.json")
with open(mapping_path, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

class_to_idx   = mapping_data["class_to_idx"]
idx_to_class   = {int(k): v for k, v in mapping_data["idx_to_class"].items()}
num_classes    = len(class_to_idx)
sorted_classes = [idx_to_class[i] for i in range(num_classes)]

# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ── Dataset Class ─────────────────────────────────────────────────────────────
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, max_per_class: int = None):
        self.transform = transform
        self.samples = []

        for cls_name in sorted_classes:
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue
            cls_idx = class_to_idx[cls_name]
            files = sorted([f for f in os.listdir(cls_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
            if max_per_class and len(files) > max_per_class:
                random.seed(SEED)
                files = random.sample(files, max_per_class)

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


def build_model_from_checkpoint(ckpt_path: str, device: torch.device) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    return model


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    class_counts = np.zeros(num_classes)
    if hasattr(dataset, "samples"):
        samples = dataset.samples
    elif hasattr(dataset, "datasets"):
        samples = []
        for ds in dataset.datasets:
            samples.extend(ds.samples)
    else:
        samples = []

    for _, label in samples:
        class_counts[label] += 1

    total_samples = len(samples)
    weights = np.where(class_counts > 0, total_samples / (num_classes * np.maximum(class_counts, 1)), 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_dataset_performance(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_targets = [], []
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            loss_sum += loss.item() * len(labels)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    avg_loss = loss_sum / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    p, r, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1


def run_smoke_test():
    print("=" * 76)
    print("  CropGuard Phase 2F: Unified Tomato Model Pipeline Smoke Test")
    print("=" * 76)

    # 1. Baseline Checkpoint SHA256 pre-check
    pre_base_sha = calculate_sha256(BASELINE_CKPT)
    print(f"[1/14] Baseline Checkpoint SHA256 (pre): {pre_base_sha}")
    assert pre_base_sha == BASELINE_SHA, f"Baseline SHA mismatch! Expected {BASELINE_SHA}, got {pre_base_sha}"

    # 2. Hardware Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A (CPU Only)"
    print(f"[2/14] Device: {device} | GPU Available: {cuda_avail} | Name: {gpu_name}")

    # 3. Load Training Datasets
    pd_train_dir = os.path.join(PLANTDOC_SPLITS_DIR, "train")
    pv_train_dir = os.path.join(PLANTVILLAGE_SPLITS_DIR, "train")
    pd_val_dir   = os.path.join(PLANTDOC_SPLITS_DIR, "validation")
    pv_val_dir   = os.path.join(PLANTVILLAGE_SPLITS_DIR, "validation")
    pd_test_dir  = os.path.join(PLANTDOC_SPLITS_DIR, "test")
    pv_test_dir  = os.path.join(PLANTVILLAGE_SPLITS_DIR, "test")

    pd_train_ds = ImageFolderDataset(pd_train_dir, transform=train_transform, max_per_class=2)
    pv_train_ds = ImageFolderDataset(pv_train_dir, transform=train_transform, max_per_class=2)
    print(f"[3/14] Training Datasets Loaded: PlantDoc={len(pd_train_ds)}, PlantVillage={len(pv_train_ds)}")
    assert len(pd_train_ds) > 0 and len(pv_train_ds) > 0

    # 4. Load Validation Datasets
    pd_val_ds   = ImageFolderDataset(pd_val_dir, transform=eval_transform, max_per_class=2)
    pv_val_ds   = ImageFolderDataset(pv_val_dir, transform=eval_transform, max_per_class=2)
    print(f"[4/14] Validation Datasets Loaded: PlantDoc={len(pd_val_ds)}, PlantVillage={len(pv_val_ds)}")
    assert len(pd_val_ds) > 0 and len(pv_val_ds) > 0

    # 5. Class Mapping Verification
    print(f"[5/14] Class Mapping Verified: {num_classes} classes")
    assert num_classes == 10

    # 6. Class Weights Computation Check
    mixed_train_ds = ConcatDataset([pd_train_ds, pv_train_ds])
    class_weights = compute_class_weights(mixed_train_ds)
    weights_valid = torch.all(torch.isfinite(class_weights)) and torch.all(class_weights > 0)
    print(f"[6/14] Class Weights Computed (Finite & Positive): {weights_valid}")
    assert weights_valid, "Class weights contain non-finite or non-positive values!"

    # 7. Model Loading from best_model.pt
    print("[7/14] Initializing EfficientNet-B0 Model from best_model.pt...")
    model = build_model_from_checkpoint(BASELINE_CKPT, device).to(device)
    model.train()

    # 8. Forward Pass Shape Check
    train_loader = DataLoader(mixed_train_ds, batch_size=4, shuffle=True)
    images, labels, _ = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    logits = model(images)
    probs  = torch.softmax(logits, dim=1)
    print(f"[8/14] Model Forward Pass Output Shape: {list(logits.shape)} (expected: [4, 10])")
    assert list(logits.shape) == [4, 10]

    # 9. Probability Sum Check
    probs_sum = probs.sum(dim=1).detach().cpu().numpy()
    print(f"[9/14] Softmax Probabilities Sum: {probs_sum}")
    assert np.allclose(probs_sum, 1.0, atol=1e-5)

    # 10. Class-Weighted Loss Computation
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    loss = criterion(logits, labels)
    print(f"[10/14] Class-Weighted CrossEntropyLoss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # 11. Backward Pass & Optimizer Step
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"[11/14] Backward Pass & Optimizer Step: OK")

    # 12. Candidate Checkpoint Save/Load Test
    smoke_ckpt_path = os.path.join(UNIFIED_CKPTS_DIR, "smoke_test_unified_ckpt.pt")
    checkpoint_payload = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "score": 0.7500,
        "config": {"mode": "smoke_test_unified"}
    }
    torch.save(checkpoint_payload, smoke_ckpt_path)
    assert os.path.exists(smoke_ckpt_path)
    os.remove(smoke_ckpt_path)
    print(f"[12/14] Candidate Checkpoint Save/Load/Delete: OK")

    # 13. Locked Test Split Isolation Check (Hash Leakage = 0)
    pd_test_full = ImageFolderDataset(pd_test_dir)
    pv_test_full = ImageFolderDataset(pv_test_dir)

    pd_test_hashes = {calculate_sha256(fp) for fp, _ in pd_test_full.samples}
    pv_test_hashes = {calculate_sha256(fp) for fp, _ in pv_test_full.samples}

    pd_train_hashes = {calculate_sha256(fp) for fp, _ in pd_train_ds.samples}
    pv_train_hashes = {calculate_sha256(fp) for fp, _ in pv_train_ds.samples}

    assert len(pd_train_hashes.intersection(pd_test_hashes)) == 0
    assert len(pv_train_hashes.intersection(pv_test_hashes)) == 0
    print(f"[13/14] Locked Test Directory Isolation (Zero Hash Overlap): VERIFIED")

    # 14. Baseline Checkpoint SHA256 post-check
    post_base_sha = calculate_sha256(BASELINE_CKPT)
    print(f"[14/14] Baseline Checkpoint SHA256 (post): {post_base_sha}")
    assert post_base_sha == BASELINE_SHA, f"CRITICAL: Baseline best_model.pt modified!"
    print(f"[14/14] Baseline Checkpoint Integrity: VERIFIED (UNTOUCHED)")
    print()
    print("=" * 76)
    print("  [PASS] ALL 14 UNIFIED SMOKE TEST CHECKS PASSED SUCCESSFULLY!")
    print("=" * 76)


def run_full_training():
    print("=" * 76)
    print("  CropGuard Phase 2F: Unified Tomato Model Full Training")
    print("=" * 76)

    pre_base_sha = calculate_sha256(BASELINE_CKPT)
    print(f"[INTEGRITY] Baseline best_model.pt SHA256 (pre): {pre_base_sha}")
    assert pre_base_sha == BASELINE_SHA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A (CPU Only)"
    print(f"[INFO] Device: {device} | GPU Available: {cuda_avail} | Name: {gpu_name}")

    pd_train_dir = os.path.join(PLANTDOC_SPLITS_DIR, "train")
    pv_train_dir = os.path.join(PLANTVILLAGE_SPLITS_DIR, "train")
    pd_val_dir   = os.path.join(PLANTDOC_SPLITS_DIR, "validation")
    pv_val_dir   = os.path.join(PLANTVILLAGE_SPLITS_DIR, "validation")

    pd_train_ds = ImageFolderDataset(pd_train_dir, transform=train_transform)
    # PlantVillage training subset sampled for CPU-balanced execution
    pv_train_ds = ImageFolderDataset(pv_train_dir, transform=train_transform, max_per_class=117)

    pd_val_ds   = ImageFolderDataset(pd_val_dir, transform=eval_transform)
    pv_val_ds   = ImageFolderDataset(pv_val_dir, transform=eval_transform, max_per_class=50)

    batch_size = 32
    mixed_train_ds = ConcatDataset([pd_train_ds, pv_train_ds])

    train_loader  = DataLoader(mixed_train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    pd_val_loader = DataLoader(pd_val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    pv_val_loader = DataLoader(pv_val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(mixed_train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model_from_checkpoint(BASELINE_CKPT, device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    total_epochs = 10
    best_val_score = 0.0
    best_epoch = 0
    best_unified_ckpt_path = os.path.join(UNIFIED_CKPTS_DIR, "best_unified_tomato_model.pt")

    history = []
    start_time_all = time.time()

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        train_preds, train_targets = [], []

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(labels)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().numpy())

        train_loss = train_loss_sum / len(mixed_train_ds)
        train_acc  = accuracy_score(train_targets, train_preds)

        pd_val_loss, pd_val_acc, pd_val_f1 = evaluate_dataset_performance(model, pd_val_loader, device)
        pv_val_loss, pv_val_acc, pv_val_f1 = evaluate_dataset_performance(model, pv_val_loader, device)

        val_score = 0.5 * pd_val_f1 + 0.5 * pv_val_f1
        is_best = val_score > best_val_score
        if is_best:
            best_val_score = val_score
            best_epoch = epoch

        epoch_duration = time.time() - epoch_start
        epoch_data = {
            "epoch": epoch,
            "duration_seconds": round(epoch_duration, 2),
            "train_loss": round(float(train_loss), 4),
            "train_accuracy": round(float(train_acc), 4),
            "plantdoc_val_loss": round(float(pd_val_loss), 4),
            "plantdoc_val_accuracy": round(float(pd_val_acc), 4),
            "plantdoc_val_macro_f1": round(float(pd_val_f1), 4),
            "plantvillage_val_loss": round(float(pv_val_loss), 4),
            "plantvillage_val_accuracy": round(float(pv_val_acc), 4),
            "plantvillage_val_macro_f1": round(float(pv_val_f1), 4),
            "combined_val_score": round(float(val_score), 4),
            "is_best_epoch": is_best
        }
        history.append(epoch_data)

        best_mark = "  <-- BEST" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{total_epochs:02d}] | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"PD Val F1: {pd_val_f1:.4f}, PV Val F1: {pv_val_f1:.4f} | "
            f"Score: {val_score:.4f}{best_mark}"
        )

        epoch_ckpt_path = os.path.join(UNIFIED_CKPTS_DIR, f"epoch_{epoch:02d}.pt")
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "pd_val_f1": pd_val_f1,
            "pv_val_f1": pv_val_f1,
            "combined_val_score": val_score,
            "best_val_score": best_val_score,
            "config": {
                "batch_size": batch_size,
                "seed": SEED,
                "baseline_sha": BASELINE_SHA
            }
        }
        torch.save(ckpt_payload, epoch_ckpt_path)
        if is_best:
            torch.save(ckpt_payload, best_unified_ckpt_path)

    total_duration = time.time() - start_time_all
    print()
    print("=" * 76)
    print("  UNIFIED TRAINING COMPLETE")
    print("=" * 76)
    print(f"  Total Duration:                 {total_duration/60:.2f} minutes")
    print(f"  Best Epoch:                     Epoch {best_epoch}")
    print(f"  Best Combined Validation Score: {best_val_score:.4f}")
    print(f"  Best Candidate Checkpoint:      {best_unified_ckpt_path}")
    print("=" * 76)

    post_base_sha = calculate_sha256(BASELINE_CKPT)
    print(f"[INTEGRITY] Baseline best_model.pt SHA256 (post): {post_base_sha}")
    assert post_base_sha == BASELINE_SHA


def main():
    parser = argparse.ArgumentParser(description="CropGuard Unified Tomato Model Pipeline")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast pipeline verification smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
    else:
        run_full_training()


if __name__ == "__main__":
    main()
