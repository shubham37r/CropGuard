import os
import json
import csv
import random
import shutil
from collections import defaultdict
from typing import Dict, Any, List
from ml.src.dataset_utils import TARGET_TOMATO_CLASSES, inspect_single_image
from ml.src.inspect_dataset import find_tomato_class_folders

SEED = 42
RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
SPLITS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "splits"))
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

def prepare_dataset() -> Dict[str, Any]:
    random.seed(SEED)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=== CropGuard Dataset Preparation & Leakage-Safe Splitting ===")

    # 1. Deterministic Class Mapping (Sorted Alphabetically)
    sorted_classes = sorted(TARGET_TOMATO_CLASSES)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted_classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(sorted_classes)}

    class_mapping_path = os.path.join(REPORTS_DIR, "class_mapping.json")
    with open(class_mapping_path, "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)
    print(f"[OK] Deterministic class mapping saved to {class_mapping_path}")

    # 2. Locate Class Directories
    class_folders = find_tomato_class_folders(RAW_DATA_DIR)
    if not class_folders:
        print(f"[ERROR] No Tomato class directories found in {RAW_DATA_DIR}")
        return {}

    # 3. Group Images by Class and Hash Group (to prevent near-duplicate / identical hash leakage)
    all_manifest_entries = []
    class_split_counts = defaultdict(lambda: {"train": 0, "validation": 0, "test": 0})
    hash_to_assigned_split = {}
    leakage_violations = 0

    for cls_name in sorted_classes:
        cls_dir = class_folders.get(cls_name)
        if not cls_dir or not os.path.exists(cls_dir):
            print(f"[WARNING] Skipping missing class directory: {cls_name}")
            continue

        image_files = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]

        # Group images by unique hash
        hash_groups = defaultdict(list)
        for img_path in image_files:
            info = inspect_single_image(img_path)
            if info["valid"]:
                hash_groups[info["hash"]].append(info)

        # Shuffle hash groups deterministically
        unique_hashes = sorted(list(hash_groups.keys()))
        random.shuffle(unique_hashes)

        # 70% Train, 15% Validation, 15% Test hash allocation
        n_hashes = len(unique_hashes)
        n_train = int(0.70 * n_hashes)
        n_val = int(0.15 * n_hashes)

        train_hashes = set(unique_hashes[:n_train])
        val_hashes = set(unique_hashes[n_train:n_train + n_val])
        test_hashes = set(unique_hashes[n_train + n_val:])

        for h, info_list in hash_groups.items():
            if h in train_hashes:
                split_name = "train"
            elif h in val_hashes:
                split_name = "validation"
            else:
                split_name = "test"

            # Check for cross-split hash leakage
            if h in hash_to_assigned_split:
                if hash_to_assigned_split[h] != split_name:
                    leakage_violations += 1
                    # Force assign to existing split to preserve 100% hash isolation
                    split_name = hash_to_assigned_split[h]
            else:
                hash_to_assigned_split[h] = split_name

            for info in info_list:
                src_path = info["filepath"]
                dest_dir = os.path.join(SPLITS_DIR, split_name, cls_name)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, os.path.basename(src_path))

                # Copy file into split folder
                if not os.path.exists(dest_path):
                    shutil.copy2(src_path, dest_path)

                all_manifest_entries.append({
                    "image_path": os.path.relpath(dest_path, os.path.dirname(SPLITS_DIR)),
                    "class_name": cls_name,
                    "class_index": class_to_idx[cls_name],
                    "split": split_name,
                    "file_hash": h
                })
                class_split_counts[cls_name][split_name] += 1

    # 4. Save Dataset Manifest CSV
    manifest_csv_path = os.path.join(REPORTS_DIR, "dataset_manifest.csv")
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "class_name", "class_index", "split", "file_hash"])
        writer.writeheader()
        writer.writerows(all_manifest_entries)
    print(f"[OK] Dataset manifest saved to {manifest_csv_path} ({len(all_manifest_entries)} total images)")

    # 5. Summary Report & Quality Checks
    print("\n--- Split Summary Breakdown ---")
    total_train = sum(counts["train"] for counts in class_split_counts.values())
    total_val = sum(counts["validation"] for counts in class_split_counts.values())
    total_test = sum(counts["test"] for counts in class_split_counts.values())
    total_all = total_train + total_val + total_test

    print(f"Total Prepared Images: {total_all}")
    print(f"Train Count: {total_train} ({round(total_train / total_all * 100, 1)}%)")
    print(f"Validation Count: {total_val} ({round(total_val / total_all * 100, 1)}%)")
    print(f"Test Count: {total_test} ({round(total_test / total_all * 100, 1)}%)")
    print(f"Hash Cross-Split Leakage Violations: {leakage_violations} (Zero Leakage Enforced)")

    return {
        "total_prepared": total_all,
        "train_count": total_train,
        "val_count": total_val,
        "test_count": total_test,
        "leakage_violations": leakage_violations,
        "class_split_counts": dict(class_split_counts)
    }

if __name__ == "__main__":
    prepare_dataset()
