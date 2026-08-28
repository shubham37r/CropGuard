import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEST_SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits", "test")

# Load Class Mapping
mapping_path = os.path.join(REPORTS_DIR, "class_mapping.json")
with open(mapping_path, "r", encoding="utf-8") as f:
    mapping_data = json.load(f)

class_to_idx = mapping_data["class_to_idx"]
idx_to_class = {int(k): v for k, v in mapping_data["idx_to_class"].items()}
num_classes = len(class_to_idx)
sorted_classes = [idx_to_class[i] for i in range(num_classes)]

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def sanity_check():
    print("=== CropGuard Model Sanity Check ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(MODELS_DIR, "cropguard_disease_model.pt")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return False

    # 1. Load Model Architecture & Weights
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("[OK] Model weights loaded successfully.")

    # 2. Find sample test images
    sample_images = []
    for root, _, files in os.walk(TEST_SPLIT_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                sample_images.append(os.path.join(root, f))
                if len(sample_images) >= 5:
                    break
        if len(sample_images) >= 5:
            break

    if not sample_images:
        print("[WARNING] No test split images found for sanity check.")
        return True

    print(f"\nPerforming inference sanity check on {len(sample_images)} test images:")
    all_valid = True

    with torch.no_grad():
        for fpath in sample_images:
            with Image.open(fpath) as img:
                tensor = eval_transform(img.convert("RGB")).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                # Validation Assertions
                assert logits.shape[1] == 10, f"Expected output shape 10, got {logits.shape[1]}"
                assert abs(np.sum(probs) - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {np.sum(probs)}"

                top_idx = int(np.argmax(probs))
                top_class = idx_to_class[top_idx]
                top_conf = round(float(probs[top_idx] * 100.0), 2)

                assert 0.0 <= top_conf <= 100.0, f"Confidence outside [0, 100]: {top_conf}"

                actual_folder = os.path.basename(os.path.dirname(fpath))
                print(f"  - File: {os.path.basename(fpath)[:25]}...")
                print(f"    Expected Folder Class: {actual_folder}")
                print(f"    Top Predicted Class:   [{top_idx}] {top_class}")
                print(f"    Confidence:            {top_conf}%")
                print(f"    Output Shape:          {list(logits.shape)}")
                print(f"    Probabilities Sum:     {np.sum(probs):.4f}")

    print("\n[OK] Model Sanity Check Passed! Output shape = 10, probabilities sum to 1.0, confidence in [0, 100]%.")
    return True

if __name__ == "__main__":
    sanity_check()
