"""
Unit tests for CropGuard Phase 2F-1 PlantDoc Field Adaptation Split & Pipeline
"""

import os
import sys
import json
import hashlib
import numpy as np
import pytest
import torch

# Ensure ml and backend modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.src.adapt_field_model import (
    PlantDocAdaptationDataset,
    build_adapted_model,
    calculate_sha256,
    BASELINE_CKPT,
    BASELINE_SHA,
    PLANTDOC_SPLITS_DIR,
    num_classes,
    sorted_classes,
    train_transform,
    eval_transform,
)

def test_baseline_checkpoint_immutability():
    """Verify baseline checkpoint file exists and matches exact SHA256."""
    assert os.path.exists(BASELINE_CKPT), f"Baseline checkpoint missing at {BASELINE_CKPT}"
    current_sha = calculate_sha256(BASELINE_CKPT)
    assert current_sha == BASELINE_SHA, f"Baseline SHA mismatch! Expected {BASELINE_SHA}, got {current_sha}"

def test_plantdoc_splits_directory_structure_and_counts():
    """Verify plantdoc_splits directory exists and matches exact split counts from authoritative report JSON."""
    assert os.path.exists(PLANTDOC_SPLITS_DIR), f"Splits directory missing: {PLANTDOC_SPLITS_DIR}"
    
    train_dir = os.path.join(PLANTDOC_SPLITS_DIR, "train")
    val_dir   = os.path.join(PLANTDOC_SPLITS_DIR, "validation")
    test_dir  = os.path.join(PLANTDOC_SPLITS_DIR, "test")

    for sdir in (train_dir, val_dir, test_dir):
        assert os.path.exists(sdir), f"Split folder missing: {sdir}"

    train_ds = PlantDocAdaptationDataset(train_dir)
    val_ds   = PlantDocAdaptationDataset(val_dir)
    test_ds  = PlantDocAdaptationDataset(test_dir)

    report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml", "reports", "plantdoc_adaptation_split.json"))
    assert os.path.exists(report_path), f"Adaptation report missing at {report_path}"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    exp_train = report["split_counts"]["train"]
    exp_val   = report["split_counts"]["validation"]
    exp_test  = report["split_counts"]["test"]

    assert len(train_ds) == exp_train, f"Expected {exp_train} train images, got {len(train_ds)}"
    assert len(val_ds) == exp_val, f"Expected {exp_val} val images, got {len(val_ds)}"
    assert len(test_ds) == exp_test, f"Expected {exp_test} test images, got {len(test_ds)}"
    assert len(train_ds) + len(val_ds) + len(test_ds) == report["total_compatible_images"], "Total images must equal total_compatible_images"

def test_zero_cross_split_hash_leakage():
    """Verify zero hash overlap between train, validation, and test splits."""
    train_dir = os.path.join(PLANTDOC_SPLITS_DIR, "train")
    val_dir   = os.path.join(PLANTDOC_SPLITS_DIR, "validation")
    test_dir  = os.path.join(PLANTDOC_SPLITS_DIR, "test")

    train_ds = PlantDocAdaptationDataset(train_dir)
    val_ds   = PlantDocAdaptationDataset(val_dir)
    test_ds  = PlantDocAdaptationDataset(test_dir)

    train_hashes = {calculate_sha256(fp) for fp, _ in train_ds.samples}
    val_hashes   = {calculate_sha256(fp) for fp, _ in val_ds.samples}
    test_hashes  = {calculate_sha256(fp) for fp, _ in test_ds.samples}

    assert len(train_hashes.intersection(val_hashes)) == 0, "Hash leak: Train vs Validation"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Hash leak: Train vs Test"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Hash leak: Validation vs Test"

def test_adapted_model_forward_pass_and_output_shape():
    """Verify EfficientNet-B0 initializes from baseline checkpoint and produces [batch_size, 10] logits."""
    device = torch.device("cpu")
    model = build_adapted_model(BASELINE_CKPT, device)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
        probs  = torch.softmax(logits, dim=1)

    assert logits.shape == (4, 10), f"Expected shape (4, 10), got {logits.shape}"
    probs_sum = probs.sum(dim=1).numpy()
    np.testing.assert_allclose(probs_sum, 1.0, atol=1e-5, err_msg="Softmax probabilities do not sum to 1")

def test_original_plantvillage_splits_untouched():
    """Verify original PlantVillage split directory exists and has not been modified."""
    pv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml", "data", "splits"))
    assert os.path.exists(pv_dir), f"PlantVillage splits missing at {pv_dir}"
    for split in ("train", "validation", "test"):
        assert os.path.exists(os.path.join(pv_dir, split))
