import os
import json
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
TRAIN_DIR = os.path.join(SPLITS_DIR, "train")

# Load Class Mapping
mapping_path = os.path.join(REPORTS_DIR, "class_mapping.json")
with open(mapping_path, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

class_to_idx = mapping_data["class_to_idx"]
idx_to_class = {int(k): v for k, v in mapping_data["idx_to_class"].items()}
num_classes = len(class_to_idx)
sorted_classes = [idx_to_class[i] for i in range(num_classes)]

class_counts = {}
total_train = 0

for cls_name in sorted_classes:
    cls_folder = os.path.join(TRAIN_DIR, cls_name)
    cnt = len([f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    class_counts[cls_name] = cnt
    total_train += cnt

class_weights = {}
for cls_name, cnt in class_counts.items():
    weight = round(total_train / (num_classes * cnt), 4)
    class_weights[cls_name] = weight

weights_payload = {
    "total_train_samples": total_train,
    "num_classes": num_classes,
    "class_counts": class_counts,
    "class_weights": class_weights
}

out_path = os.path.join(REPORTS_DIR, "class_weights.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(weights_payload, f, indent=2)

print("=== CALCULATED CLASS WEIGHTS (Training Split Only) ===")
print(f"Total Training Samples: {total_train}")
for cls_name, weight in class_weights.items():
    cnt = class_counts[cls_name]
    print(f"  - [{class_to_idx[cls_name]}] {cls_name}: {cnt} samples -> weight: {weight}")
print(f"\n[OK] Saved to {out_path}")
