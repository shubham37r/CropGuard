import os
import json
from collections import defaultdict
from typing import Dict, Any, List
from ml.src.dataset_utils import TARGET_TOMATO_CLASSES, inspect_single_image

RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

def find_tomato_class_folders(base_dir: str) -> Dict[str, str]:
    """Finds directory paths for each target Tomato class inside raw data directory."""
    found = {}
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d in TARGET_TOMATO_CLASSES:
                found[d] = os.path.join(root, d)
    return found

def inspect_dataset() -> Dict[str, Any]:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("=== CropGuard Dataset Inspection ===")

    class_folders = find_tomato_class_folders(RAW_DATA_DIR)
    
    if not class_folders:
        print(f"[WARNING] No target PlantVillage Tomato class folders found in {RAW_DATA_DIR}")
        print("Expected folder names like: Tomato___Early_blight, Tomato___healthy, etc.")

    report = {
        "dataset_name": "PlantVillage Tomato Disease Dataset",
        "dataset_source": "https://github.com/spMohanty/PlantVillage-Dataset",
        "target_classes_count": len(TARGET_TOMATO_CLASSES),
        "target_classes": TARGET_TOMATO_CLASSES,
        "classes_found_count": len(class_folders),
        "classes_stat": {},
        "total_raw_images": 0,
        "total_valid_images": 0,
        "total_corrupt_images": 0,
        "corrupt_files_list": [],
        "hash_to_files": defaultdict(list),
        "exact_duplicate_files_count": 0,
        "class_imbalance": {}
    }

    total_images_all = 0
    total_valid_all = 0
    total_corrupt_all = 0
    corrupt_list = []
    hash_map = defaultdict(list)

    for cls_name in TARGET_TOMATO_CLASSES:
        cls_dir = class_folders.get(cls_name)
        if not cls_dir or not os.path.exists(cls_dir):
            report["classes_stat"][cls_name] = {
                "exists": False,
                "path": None,
                "total_images": 0,
                "valid_images": 0,
                "corrupt_images": 0,
                "min_dimensions": None,
                "max_dimensions": None,
                "mean_dimensions": None,
                "formats": {}
            }
            continue

        image_files = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]

        widths = []
        heights = []
        formats = defaultdict(int)
        valid_cnt = 0
        corrupt_cnt = 0

        for img_path in image_files:
            info = inspect_single_image(img_path)
            total_images_all += 1
            if info["valid"]:
                valid_cnt += 1
                total_valid_all += 1
                widths.append(info["width"])
                heights.append(info["height"])
                formats[str(info["format"])] += 1
                hash_map[info["hash"]].append({
                    "path": img_path,
                    "class": cls_name
                })
            else:
                corrupt_cnt += 1
                total_corrupt_all += 1
                corrupt_list.append({
                    "path": img_path,
                    "class": cls_name,
                    "error": info["error"]
                })

        min_dim = (min(widths), min(heights)) if widths else None
        max_dim = (max(widths), max(heights)) if widths else None
        mean_dim = (round(sum(widths) / len(widths), 1), round(sum(heights) / len(heights), 1)) if widths else None

        report["classes_stat"][cls_name] = {
            "exists": True,
            "path": cls_dir,
            "total_images": len(image_files),
            "valid_images": valid_cnt,
            "corrupt_images": corrupt_cnt,
            "min_dimensions": min_dim,
            "max_dimensions": max_dim,
            "mean_dimensions": mean_dim,
            "formats": dict(formats)
        }

    exact_duplicates_count = 0
    duplicate_groups = {}
    for h, files in hash_map.items():
        if len(files) > 1:
            exact_duplicates_count += (len(files) - 1)
            duplicate_groups[h] = files

    report["total_raw_images"] = total_images_all
    report["total_valid_images"] = total_valid_all
    report["total_corrupt_images"] = total_corrupt_all
    report["corrupt_files_list"] = corrupt_list
    report["exact_duplicate_files_count"] = exact_duplicates_count
    report["duplicate_hash_groups_count"] = len(duplicate_groups)

    valid_counts = [stat["valid_images"] for stat in report["classes_stat"].values() if stat["exists"]]
    if valid_counts:
        min_cls_cnt = min(valid_counts)
        max_cls_cnt = max(valid_counts)
        report["class_imbalance"] = {
            "min_samples_per_class": min_cls_cnt,
            "max_samples_per_class": max_cls_cnt,
            "imbalance_ratio": round(max_cls_cnt / min_cls_cnt, 2) if min_cls_cnt > 0 else 0
        }

    json_path = os.path.join(REPORTS_DIR, "dataset_report.json")
    report_to_save = dict(report)
    report_to_save["hash_to_files"] = {k: v for k, v in list(duplicate_groups.items())[:50]}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_to_save, f, indent=2)

    md_path = os.path.join(REPORTS_DIR, "dataset_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PlantVillage Tomato Dataset Inspection Report\n\n")
        f.write(f"**Dataset Name**: {report['dataset_name']}\n")
        f.write(f"**Source**: {report['dataset_source']}\n")
        f.write(f"**Total Raw Images Scanned**: {total_images_all}\n")
        f.write(f"**Valid Images**: {total_valid_all}\n")
        f.write(f"**Corrupt/Unreadable Images**: {total_corrupt_all}\n")
        f.write(f"**Exact Duplicate Files**: {exact_duplicates_count}\n\n")

        if report["class_imbalance"]:
            f.write("### Class Imbalance Summary\n")
            f.write(f"- Min Class Count: {report['class_imbalance']['min_samples_per_class']}\n")
            f.write(f"- Max Class Count: {report['class_imbalance']['max_samples_per_class']}\n")
            f.write(f"- Imbalance Ratio: {report['class_imbalance']['imbalance_ratio']}x\n\n")

        f.write("### Class Breakdown\n")
        f.write("| Class Name | Status | Valid Images | Dimensions (Min/Max) | Formats |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for cls_name, stat in report["classes_stat"].items():
            if stat["exists"]:
                dims = f"{stat['min_dimensions']} / {stat['max_dimensions']}"
                fmts = ", ".join([f"{k}:{v}" for k, v in stat["formats"].items()])
                f.write(f"| `{cls_name}` | Found | {stat['valid_images']} | {dims} | {fmts} |\n")
            else:
                f.write(f"| `{cls_name}` | Missing | 0 | N/A | N/A |\n")

        f.write("\n---\n*Report generated automatically by CropGuard ML Inspection Pipeline.*\n")

    print("\n[OK] Dataset Inspection Completed!")
    print(f"- Total Raw Images: {total_images_all}")
    print(f"- Valid Images: {total_valid_all}")
    print(f"- Corrupt Images: {total_corrupt_all}")
    print(f"- Exact Duplicates: {exact_duplicates_count}")
    print(f"- Saved JSON Report: {json_path}")
    print(f"- Saved Markdown Report: {md_path}\n")

    return report

if __name__ == "__main__":
    inspect_dataset()
