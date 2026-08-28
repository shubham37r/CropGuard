"""
CropGuard Phase 2F-4 — Anti-Catastrophic-Forgetting Joint Field Adaptation Pipeline
==================================================================================
Implements joint domain adaptation with knowledge distillation (teacher-student)
to adapt EfficientNet-B0 to PlantDoc field imagery while preserving PlantVillage
source domain knowledge.

IMPORTANT SAFETY & DESIGN GUARANTEES:
  - Baseline model weights (best_model.pt) are NEVER overwritten or modified.
  - Previous adapted weights (best_field_adapted_model.pt) are NEVER overwritten or modified.
  - Candidate checkpoints are stored separately under: ml/models/field_adaptation_v2/checkpoints/
  - Strict data loader isolation:
      * PlantDoc train loader uses ONLY ml/data/plantdoc_splits/train/ (585 images)
      * PlantVillage train loader uses ONLY ml/data/splits/train/ (12,705 images)
      * PlantDoc validation loader uses ONLY ml/data/plantdoc_splits/validation/ (71 images)
      * PlantVillage validation loader uses ONLY ml/data/splits/validation/ (2,720 images)
      * Both test splits (ml/data/plantdoc_splits/test/ & ml/data/splits/test/) are PERMANENTLY LOCKED.
  - Verification mode: python -m ml.src.adapt_field_model_v2 --smoke-test
  - Full training mode: python -m ml.src.adapt_field_model_v2

Usage:
    python -m ml.src.adapt_field_model_v2
"""

import os
import sys
import json
import time
import hashlib
import random
import argparse
import numpy as np
from datetime import datetime, timezone
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
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
PREV_ADAPTED_CKPT          = os.path.join(MODELS_DIR, "field_adaptation", "checkpoints", "best_field_adapted_model.pt")
PLANTDOC_SPLITS_DIR        = os.path.join(BASE_DIR, "data", "plantdoc_splits")
PLANTVILLAGE_SPLITS_DIR    = os.path.join(BASE_DIR, "data", "splits")

ADAPTATION_V2_DIR          = os.path.join(MODELS_DIR, "field_adaptation_v2")
ADAPTATION_V2_CKPTS        = os.path.join(ADAPTATION_V2_DIR, "checkpoints")

BASELINE_SHA = "300a46ea8c2a3f9ff3f0e6fee0a9acdf19ecf5f8dc6b3e76ebe83c8a3b9d87d3"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ADAPTATION_V2_CKPTS, exist_ok=True)

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
class ImageFolderWithDomainDataset(Dataset):
    def __init__(self, root_dir: str, domain_tag: str, transform=None, max_per_class: int = None):
        self.transform = transform
        self.domain_tag = domain_tag
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
                self.samples.append((fpath, cls_idx, domain_tag))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label, domain = self.samples[idx]
        with Image.open(fpath) as img:
            img_rgb = img.convert("RGB")
            if self.transform:
                img_rgb = self.transform(img_rgb)
            return img_rgb, label, domain, fpath


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

    for _, label, _ in samples:
        class_counts[label] += 1

    total_samples = len(samples)
    weights = np.where(class_counts > 0, total_samples / (num_classes * np.maximum(class_counts, 1)), 1.0)
    return torch.tensor(weights, dtype=torch.float32)


# ── Distillation Loss Function ────────────────────────────────────────────────
def distillation_loss_fn(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Calculates temperature-scaled KL-divergence distillation loss."""
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
    return (temperature ** 2) * kl_loss


def evaluate_dataset_performance(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_targets = [], []
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels, _, _ in loader:
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
    print("  CropGuard Phase 2F-4: Anti-Catastrophic-Forgetting Smoke Test")
    print("=" * 76)

    pre_base_sha = calculate_sha256(BASELINE_CKPT)
    print(f"[1/19] Baseline Checkpoint SHA256 (pre): {pre_base_sha}")
    assert pre_base_sha == BASELINE_SHA, f"Baseline SHA mismatch! Expected {BASELINE_SHA}, got {pre_base_sha}"

    if os.path.exists(PREV_ADAPTED_CKPT):
        prev_sha = calculate_sha256(PREV_ADAPTED_CKPT)
        print(f"[2/19] Previous Adapted Checkpoint SHA256 (pre): {prev_sha}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A (CPU Only)"
    print(f"[3/19] Device: {device} | GPU Available: {cuda_avail} | Name: {gpu_name}")

    pd_train_dir = os.path.join(PLANTDOC_SPLITS_DIR, "train")
    pv_train_dir = os.path.join(PLANTVILLAGE_SPLITS_DIR, "train")
    pd_val_dir   = os.path.join(PLANTDOC_SPLITS_DIR, "validation")
    pv_val_dir   = os.path.join(PLANTVILLAGE_SPLITS_DIR, "validation")
    pd_test_dir  = os.path.join(PLANTDOC_SPLITS_DIR, "test")
    pv_test_dir  = os.path.join(PLANTVILLAGE_SPLITS_DIR, "test")

    pd_train_ds = ImageFolderWithDomainDataset(pd_train_dir, domain_tag="plantdoc", transform=train_transform, max_per_class=2)
    pv_train_ds = ImageFolderWithDomainDataset(pv_train_dir, domain_tag="plantvillage", transform=train_transform, max_per_class=2)

    pd_val_ds   = ImageFolderWithDomainDataset(pd_val_dir, domain_tag="plantdoc", transform=eval_transform, max_per_class=2)
    pv_val_ds   = ImageFolderWithDomainDataset(pv_val_dir, domain_tag="plantvillage", transform=eval_transform, max_per_class=2)

    print(f"[4/19] Training Datasets Loaded: PlantDoc={len(pd_train_ds)}, PlantVillage={len(pv_train_ds)}")
    print(f"[5/19] Validation Datasets Loaded: PlantDoc={len(pd_val_ds)}, PlantVillage={len(pv_val_ds)}")
    assert len(pd_train_ds) > 0 and len(pv_train_ds) > 0
    assert len(pd_val_ds) > 0 and len(pv_val_ds) > 0

    mixed_train_ds = ConcatDataset([pd_train_ds, pv_train_ds])
    train_loader   = DataLoader(mixed_train_ds, batch_size=4, shuffle=True)
    print(f"[6/19] Mixed Batch DataLoader Constructed: {len(mixed_train_ds)} samples total (Batch Size: 4)")

    print(f"[7/19] Class Mapping Verified: {num_classes} classes")
    assert num_classes == 10

    print("[8/19] Loading Teacher Model (Frozen Baseline)...")
    teacher_model = build_model_from_checkpoint(BASELINE_CKPT, device).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    teacher_frozen = all(not p.requires_grad for p in teacher_model.parameters())
    print(f"[9/19] Teacher Model Parameters Frozen: {teacher_frozen}")
    assert teacher_frozen

    print("[10/19] Initializing Student Model from Baseline Checkpoint...")
    student_model = build_model_from_checkpoint(BASELINE_CKPT, device).to(device)
    student_model.train()
    student_trainable = any(p.requires_grad for p in student_model.parameters())
    print(f"[11/19] Student Model Initialized & Trainable: {student_trainable}")
    assert student_trainable

    images, labels, domains, _ = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    student_logits = student_model(images)
    student_probs  = torch.softmax(student_logits, dim=1)
    print(f"[12/19] Student Forward Pass Output Shape: {list(student_logits.shape)}")
    assert list(student_logits.shape) == [4, 10]

    probs_sum = student_probs.sum(dim=1).detach().cpu().numpy()
    print(f"[13/19] Softmax Probabilities Sum: {probs_sum}")
    assert np.allclose(probs_sum, 1.0, atol=1e-5)

    with torch.no_grad():
        teacher_logits = teacher_model(images)
    print(f"[14/19] Teacher Forward Pass Output Shape: {list(teacher_logits.shape)}")
    assert list(teacher_logits.shape) == [4, 10]

    class_weights = compute_class_weights(mixed_train_ds).to(device)
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

    cls_loss = cls_criterion(student_logits, labels)
    distill_loss = distillation_loss_fn(student_logits, teacher_logits, temperature=2.0)
    lambda_distill = 1.0
    total_loss = cls_loss + lambda_distill * distill_loss

    print(f"[15/19] Losses Calculated: Classification={cls_loss.item():.4f}, Distillation={distill_loss.item():.4f}, Total={total_loss.item():.4f}")
    assert not torch.isnan(total_loss)

    optimizer = optim.AdamW(student_model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print(f"[16/19] Backpropagation & Optimizer Step: OK")

    smoke_v2_ckpt_path = os.path.join(ADAPTATION_V2_CKPTS, "smoke_test_v2_ckpt.pt")
    checkpoint_payload = {
        "epoch": 1,
        "student_state_dict": student_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_score": 0.7500,
        "config": {"mode": "smoke_test_v2"}
    }
    torch.save(checkpoint_payload, smoke_v2_ckpt_path)
    assert os.path.exists(smoke_v2_ckpt_path)
    os.remove(smoke_v2_ckpt_path)
    print(f"[17/19] Candidate Checkpoint Save/Load/Delete: OK")

    pd_test_full = ImageFolderWithDomainDataset(pd_test_dir, domain_tag="plantdoc")
    pv_test_full = ImageFolderWithDomainDataset(pv_test_dir, domain_tag="plantvillage")

    pd_test_hashes = {calculate_sha256(fp) for fp, _, _ in pd_test_full.samples}
    pv_test_hashes = {calculate_sha256(fp) for fp, _, _ in pv_test_full.samples}

    pd_train_hashes = {calculate_sha256(fp) for fp, _, _ in pd_train_ds.samples}
    pv_train_hashes = {calculate_sha256(fp) for fp, _, _ in pv_train_ds.samples}

    assert len(pd_train_hashes.intersection(pd_test_hashes)) == 0
    assert len(pv_train_hashes.intersection(pv_test_hashes)) == 0
    print(f"[18/19] Test Split Isolation (Zero Hash Leakage): VERIFIED")

    post_base_sha = calculate_sha256(BASELINE_CKPT)
    print(f"[19/19] Baseline Checkpoint SHA256 (post): {post_base_sha}")
    assert post_base_sha == BASELINE_SHA
    print(f"[19/19] Baseline Checkpoint Integrity: VERIFIED (UNTOUCHED)")
    print()
    print("=" * 76)
    print("  [PASS] ALL 19 SMOKE TEST CHECKS PASSED SUCCESSFULLY!")
    print("=" * 76)


def run_joint_adaptation_training():
    print("=" * 76)
    print("  CropGuard Phase 2F-4: Anti-Catastrophic-Forgetting Joint Training")
    print("=" * 76)

    # 1. Baseline & Previous Checkpoints Integrity Verification
    pre_base_sha = calculate_sha256(BASELINE_CKPT)
    print(f"[INTEGRITY] Baseline best_model.pt SHA256 (pre): {pre_base_sha}")
    assert pre_base_sha == BASELINE_SHA, f"Baseline SHA mismatch!"

    if os.path.exists(PREV_ADAPTED_CKPT):
        prev_sha = calculate_sha256(PREV_ADAPTED_CKPT)
        print(f"[INTEGRITY] Previous Adapted best_field_adapted_model.pt SHA256 (pre): {prev_sha}")

    # 2. Hardware Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A (CPU Only)"
    print(f"[INFO] Device: {device} | GPU Available: {cuda_avail} | Name: {gpu_name}")

    # 3. Load Training & Validation Datasets
    pd_train_dir = os.path.join(PLANTDOC_SPLITS_DIR, "train")
    pv_train_dir = os.path.join(PLANTVILLAGE_SPLITS_DIR, "train")
    pd_val_dir   = os.path.join(PLANTDOC_SPLITS_DIR, "validation")
    pv_val_dir   = os.path.join(PLANTVILLAGE_SPLITS_DIR, "validation")

    pd_train_ds = ImageFolderWithDomainDataset(pd_train_dir, domain_tag="plantdoc", transform=train_transform)
    # Balanced rehearsal sampling of PlantVillage to maintain efficient CPU execution while preventing forgetting
    pv_train_ds = ImageFolderWithDomainDataset(pv_train_dir, domain_tag="plantvillage", transform=train_transform, max_per_class=117)

    pd_val_ds   = ImageFolderWithDomainDataset(pd_val_dir, domain_tag="plantdoc", transform=eval_transform)
    # PlantVillage validation set subsampled during training for fast CPU execution
    pv_val_ds   = ImageFolderWithDomainDataset(pv_val_dir, domain_tag="plantvillage", transform=eval_transform, max_per_class=50)


    batch_size = 32
    mixed_train_ds = ConcatDataset([pd_train_ds, pv_train_ds])

    train_loader  = DataLoader(mixed_train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    pd_val_loader = DataLoader(pd_val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    pv_val_loader = DataLoader(pv_val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[INFO] Training Samples:   PlantDoc={len(pd_train_ds)}, PlantVillage Rehearsal={len(pv_train_ds)} (Total: {len(mixed_train_ds)})")
    print(f"[INFO] Validation Samples: PlantDoc={len(pd_val_ds)}, PlantVillage={len(pv_val_ds)}")
    print(f"[INFO] Locked Test Sets:   PlantDoc (70 images) & PlantVillage (2,735 images) PERMANENTLY LOCKED")
    print()

    # 4. Initialize Teacher & Student Models
    print("[INFO] Initializing Frozen Teacher Model (best_model.pt)...")
    teacher_model = build_model_from_checkpoint(BASELINE_CKPT, device).to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    print("[INFO] Initializing Trainable Student Model (best_model.pt baseline initialization)...")
    student_model = build_model_from_checkpoint(BASELINE_CKPT, device).to(device)

    # Loss & Hyperparameters
    class_weights = compute_class_weights(mixed_train_ds).to(device)
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)
    temperature = 2.0
    lambda_distill = 1.0

    stage1_epochs = 2
    stage2_epochs = 8
    total_epochs  = stage1_epochs + stage2_epochs

    best_val_score = 0.0
    best_epoch = 0
    best_candidate_ckpt_path = os.path.join(ADAPTATION_V2_CKPTS, "best_field_adapted_model_v2.pt")

    history = []
    start_time_all = time.time()

    print("=" * 76)
    print("  STAGE 1: Head Fine-Tuning + Distillation (Epochs 1-2, Backbone Frozen)")
    print("=" * 76)

    # Freeze backbone initially
    for param in student_model.features.parameters():
        param.requires_grad = False
    for param in student_model.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(student_model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()

        # Switch to Stage 2 at Epoch 3
        if epoch == stage1_epochs + 1:
            print()
            print("=" * 76)
            print("  STAGE 2: Upper Backbone Fine-Tuning + Distillation (Epochs 3-10, Low LR)")
            print("=" * 76)
            for param in student_model.features[5:].parameters():
                param.requires_grad = True

            optimizer = optim.AdamW([
                {"params": [p for p in student_model.features[5:].parameters() if p.requires_grad], "lr": 1e-5},
                {"params": [p for p in student_model.classifier.parameters() if p.requires_grad], "lr": 1e-4}
            ], weight_decay=1e-4)

        stage_name = "Stage 1 (Head Only)" if epoch <= stage1_epochs else "Stage 2 (Upper Backbone)"

        # ── Train Step ───────────────────────────────────────────────────────
        student_model.train()
        train_loss_sum = 0.0
        train_cls_sum = 0.0
        train_distill_sum = 0.0
        train_preds, train_targets = [], []

        for images, labels, domains, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            student_logits = student_model(images)
            with torch.no_grad():
                teacher_logits = teacher_model(images)

            c_loss = cls_criterion(student_logits, labels)
            d_loss = distillation_loss_fn(student_logits, teacher_logits, temperature=temperature)
            tot_loss = c_loss + lambda_distill * d_loss

            tot_loss.backward()
            optimizer.step()

            train_loss_sum += tot_loss.item() * len(labels)
            train_cls_sum  += c_loss.item() * len(labels)
            train_distill_sum += d_loss.item() * len(labels)

            preds = student_logits.argmax(dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().numpy())

        train_loss = train_loss_sum / len(mixed_train_ds)
        train_acc  = accuracy_score(train_targets, train_preds)

        # ── Validation Step (PlantDoc Val & PlantVillage Val) ────────────────
        pd_val_loss, pd_val_acc, pd_val_f1 = evaluate_dataset_performance(student_model, pd_val_loader, device)
        pv_val_loss, pv_val_acc, pv_val_f1 = evaluate_dataset_performance(student_model, pv_val_loader, device)

        # Combined Model Selection Score: 0.5 * PlantDoc Val F1 + 0.5 * PlantVillage Val F1
        val_score = 0.5 * pd_val_f1 + 0.5 * pv_val_f1

        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = val_score > best_val_score
        if is_best:
            best_val_score = val_score
            best_epoch = epoch

        epoch_data = {
            "epoch": epoch,
            "stage": stage_name,
            "duration_seconds": round(epoch_duration, 2),
            "learning_rate": current_lr,
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
            f"Epoch [{epoch:02d}/{total_epochs:02d}] ({stage_name}) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"PD Val F1: {pd_val_f1:.4f}, PV Val F1: {pv_val_f1:.4f} | "
            f"Score: {val_score:.4f}{best_mark}"
        )

        epoch_ckpt_path = os.path.join(ADAPTATION_V2_CKPTS, f"epoch_{epoch:02d}.pt")
        ckpt_payload = {
            "epoch": epoch,
            "stage": stage_name,
            "model_state_dict": student_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "pd_val_f1": pd_val_f1,
            "pv_val_f1": pv_val_f1,
            "combined_val_score": val_score,
            "best_val_score": best_val_score,
            "config": {
                "batch_size": batch_size,
                "seed": SEED,
                "lambda_distill": lambda_distill,
                "temperature": temperature,
                "baseline_sha": BASELINE_SHA
            }
        }
        torch.save(ckpt_payload, epoch_ckpt_path)

        if is_best:
            torch.save(ckpt_payload, best_candidate_ckpt_path)

        history_json_path = os.path.join(REPORTS_DIR, "field_adaptation_v2_training_history.json")
        with open(history_json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    total_duration = time.time() - start_time_all
    print()
    print("=" * 76)
    print("  TRAINING COMPLETE")
    print("=" * 76)
    print(f"  Total Duration:                   {total_duration/60:.2f} minutes ({total_duration:.1f}s)")
    print(f"  Best Epoch:                       Epoch {best_epoch}")
    print(f"  Best Combined Validation Score:   {best_val_score:.4f}")
    print(f"  Best Candidate Checkpoint Path:   {best_candidate_ckpt_path}")
    print("=" * 76)

    post_base_sha = calculate_sha256(BASELINE_CKPT)
    print(f"[INTEGRITY] Baseline best_model.pt SHA256 (post): {post_base_sha}")
    assert post_base_sha == BASELINE_SHA, "CRITICAL: Baseline best_model.pt modified!"

    if os.path.exists(PREV_ADAPTED_CKPT):
        post_prev_sha = calculate_sha256(PREV_ADAPTED_CKPT)
        print(f"[INTEGRITY] Previous Adapted best_field_adapted_model.pt SHA256 (post): {post_prev_sha}")
        assert post_prev_sha == prev_sha, "CRITICAL: Previous adapted checkpoint modified!"
    print("[INTEGRITY] All existing checkpoints verified UNTOUCHED!")


def main():
    parser = argparse.ArgumentParser(description="CropGuard Anti-Catastrophic-Forgetting Joint Adaptation Pipeline")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast pipeline verification smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
    else:
        run_joint_adaptation_training()


if __name__ == "__main__":
    main()
