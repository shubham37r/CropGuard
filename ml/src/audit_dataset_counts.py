import os
import csv
import json
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
TRAIN_DIR = os.path.join(SPLITS_DIR, "train")
MANIFEST_PATH = os.path.join(REPORTS_DIR, "dataset_manifest.csv")

print("=== DATASET COUNT AUDIT ===")

# 1. Physical file counts in ml/data/splits/train/
physical_counts = defaultdict(int)
total_physical = 0
if os.path.exists(TRAIN_DIR):
    for cls in os.listdir(TRAIN_DIR):
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if os.path.isdir(cls_dir):
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            physical_counts[cls] = len(files)
            total_physical += len(files)

print(f"Physical Files in ml/data/splits/train/: {total_physical}")
for cls, cnt in sorted(physical_counts.items()):
    print(f"  - {cls}: {cnt}")

# 2. Manifest counts
manifest_split_counts = defaultdict(lambda: defaultdict(int))
total_manifest_train = 0
if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            cls = row["class_name"]
            manifest_split_counts[split][cls] += 1
            if split == "train":
                total_manifest_train += 1

print(f"\nManifest Entries for split='train': {total_manifest_train}")
for cls, cnt in sorted(manifest_split_counts["train"].items()):
    print(f"  - {cls}: {cnt}")

# 3. Check for leftover/duplicate directories in raw or splits
print(f"\nManifest Entries for split='validation': {sum(manifest_split_counts['validation'].values())}")
print(f"Manifest Entries for split='test': {sum(manifest_split_counts['test'].values())}")
