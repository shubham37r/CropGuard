# CropGuard ML Workspace (Phase 2B-1 Dataset Preparation)

This workspace houses dataset acquisition, inspection, data quality checks, and leakage-safe splitting pipelines for training CropGuard's computer-vision crop health classifiers.

---

## 📁 Directory Structure

```
ml/
├── data/
│   ├── raw/         # Raw downloaded PlantVillage dataset (git-ignored)
│   ├── processed/   # Cleaned images
│   └── splits/      # 70/15/15 Train/Validation/Test leakage-safe splits (git-ignored)
├── src/
│   ├── dataset_utils.py     # Image inspection & SHA-256 content hashing utilities
│   ├── download_dataset.py  # PlantVillage dataset downloader & extraction
│   ├── inspect_dataset.py   # Quality check, imbalance analysis & report generator
│   └── prepare_dataset.py   # Deterministic class mapping, splitting & manifest builder
├── reports/
│   ├── dataset_report.json
│   ├── dataset_report.md
│   ├── class_mapping.json
│   ├── dataset_manifest.csv
│   └── field_dataset_compatibility.md
├── models/          # Trained model weights storage (git-ignored)
├── notebooks/       # ML exploration notebooks
├── requirements.txt
└── README.md
```

---

## 🎯 Target Tomato Classes (10 Scope Classes)

1. `Tomato___healthy`
2. `Tomato___Bacterial_spot`
3. `Tomato___Early_blight`
4. `Tomato___Late_blight`
5. `Tomato___Leaf_Mold`
6. `Tomato___Septoria_leaf_spot`
7. `Tomato___Spider_mites`
8. `Tomato___Target_Spot`
9. `Tomato___Tomato_mosaic_virus`
10. `Tomato___Tomato_Yellow_Leaf_Curl_Virus`

---

## 🚀 How to Run Dataset Pipeline

### 1. Download PlantVillage Dataset
```bash
python -m ml.src.download_dataset
```

### 2. Inspect Dataset Quality & Generate Reports
```bash
python -m ml.src.inspect_dataset
```

### 3. Generate Leakage-Safe Splits & Manifest
```bash
python -m ml.src.prepare_dataset
```

---

## 🛡️ Reproducibility & Leakage Prevention Rules

- **Fixed Random Seed**: `SEED = 42` for all train/val/test splits.
- **Content-Hash Isolation**: SHA-256 content hashing is used to group duplicate/identical files together before splitting. Identical hashes are prevented from crossing between train, validation, or test sets.
- **Deterministic Class Mapping**: Alphabetically sorted class indices saved to `ml/reports/class_mapping.json`.
- **PlantVillage Scope Disclaimer**: PlantVillage imagery consists primarily of controlled, curated laboratory foliage photography. A separate field evaluation dataset (e.g., PlantDoc) will be evaluated separately to test real-world field generalization.
