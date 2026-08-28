import os
import json
import csv

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")

TEST_DIR = os.path.join(SPLITS_DIR, "test")
MAPPING_PATH = os.path.join(REPORTS_DIR, "class_mapping.json")
BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pt")


# ---------------------------------------------------------
# Class mapping
# ---------------------------------------------------------

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

class_to_idx = mapping_data["class_to_idx"]

idx_to_class = {
    int(k): v
    for k, v in mapping_data["idx_to_class"].items()
}

num_classes = len(class_to_idx)

sorted_classes = [
    idx_to_class[i]
    for i in range(num_classes)
]


# ---------------------------------------------------------
# Evaluation preprocessing
# Must match training / inference preprocessing
# ---------------------------------------------------------

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------
# Test Dataset
# ---------------------------------------------------------

class TestDataset(Dataset):

    def __init__(self, split_dir, transform=None):
        self.samples = []
        self.transform = transform

        for class_name in sorted_classes:

            class_folder = os.path.join(
                split_dir,
                class_name
            )

            if not os.path.isdir(class_folder):
                continue

            class_index = class_to_idx[class_name]

            for filename in sorted(
                os.listdir(class_folder)
            ):

                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp")
                ):

                    path = os.path.join(
                        class_folder,
                        filename
                    )

                    self.samples.append(
                        (path, class_index)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

        return image, label, path


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def build_model():

    model = models.efficientnet_b0(
        weights=None
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate():

    print(
        "=== CropGuard Model Evaluation "
        "(PlantVillage Test Set) ==="
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if not os.path.exists(BEST_MODEL_PATH):
        print(
            f"[ERROR] Best checkpoint not found:\n"
            f"{BEST_MODEL_PATH}"
        )
        return

    print(f"Checkpoint: {BEST_MODEL_PATH}")

    # Load checkpoint
    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model = build_model()

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        epoch = checkpoint.get(
            "epoch",
            "Unknown"
        )

        val_f1 = checkpoint.get(
            "best_val_macro_f1",
            checkpoint.get(
                "val_macro_f1",
                "N/A"
            ),
        )

    else:

        model.load_state_dict(checkpoint)

        epoch = "Unknown"
        val_f1 = "N/A"

    model.to(device)
    model.eval()

    print(f"Best checkpoint epoch: {epoch}")
    print(f"Validation Macro F1: {val_f1}")

    # Dataset
    test_dataset = TestDataset(
        TEST_DIR,
        transform=eval_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Test images: {len(test_dataset)}"
    )

    if len(test_dataset) == 0:
        print("[ERROR] Test dataset is empty.")
        return

    # Predictions
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels, _ in test_loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    probabilities = np.array(
        all_probabilities
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    top1_accuracy = accuracy_score(
        y_true,
        y_pred
    )

    top3_predictions = np.argsort(
        probabilities,
        axis=1
    )[:, -3:]

    top3_accuracy = np.mean(
        [
            true_label in top3
            for true_label, top3
            in zip(y_true, top3_predictions)
        ]
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    class_precision, class_recall, class_f1, class_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            labels=list(range(num_classes)),
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\n=== TEST RESULTS ===")

    print(
        f"Top-1 Accuracy:    "
        f"{top1_accuracy * 100:.2f}%"
    )

    print(
        f"Top-3 Accuracy:    "
        f"{top3_accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision:   "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall:      "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1:       "
        f"{weighted_f1:.4f}"
    )

    # -----------------------------------------------------
    # Per-class report
    # -----------------------------------------------------

    classification_report_data = (
        classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=sorted_classes,
            output_dict=True,
            zero_division=0,
        )
    )

    with open(
        os.path.join(
            REPORTS_DIR,
            "classification_report_test.json"
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            classification_report_data,
            f,
            indent=2,
        )

    # CSV
    csv_path = os.path.join(
        REPORTS_DIR,
        "class_metrics_test.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "class_index",
            "class_name",
            "precision",
            "recall",
            "f1_score",
            "samples",
        ])

        for index in range(num_classes):

            writer.writerow([
                index,
                sorted_classes[index],
                round(
                    float(class_precision[index]),
                    4,
                ),
                round(
                    float(class_recall[index]),
                    4,
                ),
                round(
                    float(class_f1[index]),
                    4,
                ),
                int(class_support[index]),
            ])

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    )

    plt.figure(figsize=(11, 9))

    plt.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
    )

    plt.title(
        "CropGuard — PlantVillage Test Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(num_classes)

    short_names = [
        class_name.split("___")[-1][:18]
        for class_name in sorted_classes
    ]

    plt.xticks(
        tick_marks,
        short_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        tick_marks,
        short_names,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORTS_DIR,
            "confusion_matrix_test.png"
        ),
        dpi=150,
    )

    plt.close()

    # -----------------------------------------------------
    # Confidence analysis
    # -----------------------------------------------------

    confidences = (
        np.max(probabilities, axis=1)
        * 100.0
    )

    correct_mask = (
        y_pred == y_true
    )

    incorrect_mask = (
        ~correct_mask
    )

    correct_confidences = (
        confidences[correct_mask]
    )

    incorrect_confidences = (
        confidences[incorrect_mask]
    )

    high_conf_wrong_percentage = (
        float(
            np.mean(
                incorrect_confidences >= 70
            ) * 100
        )
        if len(incorrect_confidences)
        else 0.0
    )

    confidence_report = {

        "mean_confidence": round(
            float(np.mean(confidences)),
            2,
        ),

        "median_confidence": round(
            float(np.median(confidences)),
            2,
        ),

        "correct_predictions": {
            "count": int(
                np.sum(correct_mask)
            ),

            "mean_confidence": round(
                float(
                    np.mean(
                        correct_confidences
                    )
                ),
                2,
            ),

            "median_confidence": round(
                float(
                    np.median(
                        correct_confidences
                    )
                ),
                2,
            ),
        },

        "incorrect_predictions": {

            "count": int(
                np.sum(incorrect_mask)
            ),

            "mean_confidence": round(
                float(
                    np.mean(
                        incorrect_confidences
                    )
                ),
                2,
            )
            if len(incorrect_confidences)
            else 0.0,

            "median_confidence": round(
                float(
                    np.median(
                        incorrect_confidences
                    )
                ),
                2,
            )
            if len(incorrect_confidences)
            else 0.0,

            "pct_incorrect_with_confidence_ge_70": round(
                high_conf_wrong_percentage,
                2,
            ),
        },
    }

    with open(
        os.path.join(
            REPORTS_DIR,
            "confidence_analysis_test.json"
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            confidence_report,
            f,
            indent=2,
        )

    print(
        "\n[OK] Evaluation reports saved "
        "to ml/reports/"
    )


if __name__ == "__main__":
    evaluate()