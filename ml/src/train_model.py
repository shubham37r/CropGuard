import os
import sys
import json
import csv
import random
import time
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
MISCLASSIFIED_DIR = os.path.join(REPORTS_DIR, "misclassified")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)

# Load Class Mapping
mapping_path = os.path.join(REPORTS_DIR, "class_mapping.json")
with open(mapping_path, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

class_to_idx = mapping_data["class_to_idx"]
idx_to_class = {int(k): v for k, v in mapping_data["idx_to_class"].items()}
num_classes = len(class_to_idx)
sorted_classes = [idx_to_class[i] for i in range(num_classes)]

# Image Transforms
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

class CropDiseaseDataset(Dataset):
    def __init__(self, split_dir, transform=None, max_per_class=None):
        self.samples = []
        self.transform = transform

        for cls_name in sorted_classes:
            cls_folder = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue
            cls_idx = class_to_idx[cls_name]
            files = sorted([f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
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

def calculate_class_weights(train_dataset):
    class_counts = np.zeros(num_classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    total_samples = len(train_dataset)
    weights = total_samples / (num_classes * class_counts)
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights, class_counts

def build_efficientnet_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # Freeze backbone initially for Stage 1
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels, _ in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    for inputs, labels, paths in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_paths.extend(paths)

    total = len(all_labels)
    eval_loss = running_loss / total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    return eval_loss, acc, f1, all_preds, all_labels, all_probs, all_paths

def save_checkpoint(filepath, model, optimizer, scheduler, epoch, val_f1, best_val_f1, config):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_macro_f1": val_f1,
        "best_val_macro_f1": best_val_f1,
        "config": config
    }
    torch.save(checkpoint, filepath)

def load_latest_checkpoint(checkpoints_dir, device):
    if not os.path.exists(checkpoints_dir):
        return None
    checkpoint_files = [
        f for f in os.listdir(checkpoints_dir)
        if f.startswith("epoch_") and f.endswith(".pt")
    ]
    if not checkpoint_files:
        return None
    checkpoint_files.sort()
    latest_file = os.path.join(checkpoints_dir, checkpoint_files[-1])
    return latest_file

def main():
    parser = argparse.ArgumentParser(description="CropGuard Disease Model Training Pipeline")
    parser.add_argument("--fresh", action="store_true", help="Start training fresh from epoch 1, ignoring checkpoints")
    parser.add_argument("--smoke-test", action="store_true", help="Run a fast 1-epoch smoke test pipeline verification")
    args = parser.parse_args()

    print("=== CropGuard ML Pipeline Configuration & Resource Detection ===", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A (CPU Only)"

    print(f"Device:               {device}", flush=True)
    print(f"GPU Available:        {gpu_available}", flush=True)
    print(f"GPU Name:             {gpu_name}", flush=True)
    print(f"Random Seed:          {SEED}", flush=True)

    train_dir = os.path.join(SPLITS_DIR, "train")
    val_dir = os.path.join(SPLITS_DIR, "validation")
    test_dir = os.path.join(SPLITS_DIR, "test")

    max_per_class = 2 if args.smoke_test else None
    train_ds = CropDiseaseDataset(train_dir, transform=train_transform, max_per_class=max_per_class)
    val_ds = CropDiseaseDataset(val_dir, transform=eval_transform, max_per_class=max_per_class)
    test_ds = CropDiseaseDataset(test_dir, transform=eval_transform, max_per_class=max_per_class)

    batch_size = 64 if not args.smoke_test else 4
    num_train = len(train_ds)
    num_val = len(val_ds)
    batches_per_epoch = (num_train + batch_size - 1) // batch_size

    print(f"Batch Size:           {batch_size}", flush=True)
    print(f"Train Image Count:    {num_train}", flush=True)
    print(f"Validation Count:     {num_val}", flush=True)
    print(f"Number of Classes:    {num_classes}", flush=True)
    print(f"Estimated Batches/Epoch: {batches_per_epoch}", flush=True)

    class_weights, train_counts = calculate_class_weights(train_ds)
    class_weights = class_weights.to(device)

    weights_json = {
        "total_train_samples": num_train,
        "num_classes": num_classes,
        "class_counts": {idx_to_class[i]: int(train_counts[i]) for i in range(num_classes)},
        "class_weights": {idx_to_class[i]: round(class_weights[i].item(), 4) for i in range(num_classes)}
    }
    with open(os.path.join(REPORTS_DIR, "class_weights.json"), "w", encoding="utf-8") as f:
        json.dump(weights_json, f, indent=2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_efficientnet_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    stage1_epochs = 2 if not args.smoke_test else 1
    stage2_epochs = 3 if not args.smoke_test else 1

    config_dict = {
        "framework": "PyTorch",
        "torch_version": torch.__version__,
        "model_architecture": "EfficientNet-B0",
        "pretrained_weights": "ImageNet (DEFAULT)",
        "input_dimensions": "224x224x3 RGB",
        "random_seed": SEED,
        "batch_size": batch_size,
        "loss_function": "CrossEntropyLoss(label_smoothing=0.1, class_weighted=True)",
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "stage1_lr": 1e-3,
        "stage2_lr": 1e-4,
        "smoke_test": args.smoke_test
    }

    if not args.smoke_test:
        with open(os.path.join(REPORTS_DIR, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

    start_epoch = 1
    best_val_f1 = 0.0
    history_file = os.path.join(REPORTS_DIR, "training_history.json") if not args.smoke_test else os.path.join(REPORTS_DIR, "smoke_history.json")
    history = []

    if os.path.exists(history_file) and not args.fresh:
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    latest_checkpoint = None if args.fresh else load_latest_checkpoint(CHECKPOINTS_DIR, device)
    if latest_checkpoint:
        print(f"\n[RESUME] Found existing checkpoint: {latest_checkpoint}", flush=True)
        ckpt = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_macro_f1", 0.0)
        print(f"[RESUME] Resuming training from Epoch {start_epoch} (Best Val Macro F1: {best_val_f1:.4f})", flush=True)
    else:
        print(f"\n[FRESH] Starting fresh training run from Epoch 1", flush=True)

    if args.smoke_test:
        print("\n--- RUNNING PIPELINE SMOKE TEST ---", flush=True)
        optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_f1, _, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        smoke_ckpt_path = os.path.join(CHECKPOINTS_DIR, "smoke_checkpoint.pt")
        save_checkpoint(smoke_ckpt_path, model, optimizer, None, 1, v_f1, v_f1, config_dict)
        assert os.path.exists(smoke_ckpt_path), "Smoke checkpoint writing failed"

        ckpt_test = torch.load(smoke_ckpt_path, map_location=device, weights_only=False)
        assert "model_state_dict" in ckpt_test, "Smoke checkpoint verification failed"

        os.remove(smoke_ckpt_path)

        print(f"[SMOKE TEST OK] Pipeline Verification Passed in {elapsed:.2f}s!", flush=True)
        print("  - Dataset loading: OK", flush=True)
        print("  - EfficientNet-B0 forward pass: OK", flush=True)
        print("  - Loss computation: OK", flush=True)
        print("  - Backpropagation: OK", flush=True)
        print("  - Validation evaluation: OK", flush=True)
        print("  - Checkpoint save/load resume logic: OK", flush=True)
        return

    # --- FULL TRAINING STAGES ---
    total_epochs = stage1_epochs + stage2_epochs
    optimizer_s1 = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(start_epoch, stage1_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer_s1, device)
        v_loss, v_acc, v_f1, _, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, "best_model.pt"))
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "cropguard_disease_model.pt"))

        ckpt_file = os.path.join(CHECKPOINTS_DIR, f"epoch_{epoch:02d}.pt")
        save_checkpoint(ckpt_file, model, optimizer_s1, None, epoch, v_f1, best_val_f1, config_dict)

        epoch_record = {
            "epoch": epoch,
            "stage": 1,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(v_loss, 4),
            "val_acc": round(v_acc, 4),
            "val_macro_f1": round(v_f1, 4),
            "lr": 1e-3,
            "duration_sec": round(elapsed, 1)
        }
        history.append(epoch_record)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(f"Stage 1 Epoch {epoch}/{stage1_epochs} [{elapsed:.1f}s] - Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}, Val Macro F1: {v_f1:.4f}", flush=True)

    if start_epoch <= total_epochs:
        print("\n--- STAGE 2: Unfreezing Upper Backbone Layers for Fine-Tuning ---", flush=True)
        for param in model.features[5:].parameters():
            param.requires_grad = True

        optimizer_s2 = optim.AdamW([
            {'params': model.features[5:].parameters(), 'lr': 1e-4},
            {'params': model.classifier.parameters(), 'lr': 2e-4}
        ], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=stage2_epochs)

        s2_start = max(start_epoch, stage1_epochs + 1)
        for epoch in range(s2_start, total_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer_s2, device)
            v_loss, v_acc, v_f1, _, _, _, _ = evaluate_model(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0

            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, "best_model.pt"))
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, "cropguard_disease_model.pt"))

            ckpt_file = os.path.join(CHECKPOINTS_DIR, f"epoch_{epoch:02d}.pt")
            save_checkpoint(ckpt_file, model, optimizer_s2, scheduler, epoch, v_f1, best_val_f1, config_dict)

            epoch_record = {
                "epoch": epoch,
                "stage": 2,
                "train_loss": round(tr_loss, 4),
                "train_acc": round(tr_acc, 4),
                "val_loss": round(v_loss, 4),
                "val_acc": round(v_acc, 4),
                "val_macro_f1": round(v_f1, 4),
                "lr": scheduler.get_last_lr()[0],
                "duration_sec": round(elapsed, 1)
            }
            history.append(epoch_record)
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            print(f"Stage 2 Epoch {epoch}/{total_epochs} [{elapsed:.1f}s] - Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}, Val Macro F1: {v_f1:.4f}", flush=True)

    print("\n[OK] Training Complete!", flush=True)

if __name__ == "__main__":
    main()
