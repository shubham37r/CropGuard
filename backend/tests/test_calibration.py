"""
Unit tests for CropGuard Phase 2D Temperature Scaling Calibration
================================================================
Verifies mathematical correctness, constraint satisfaction, and safety guarantees
of the TemperatureScaler and calibration pipeline.
"""

import os
import sys
import hashlib
import numpy as np
import pytest
import torch

# Ensure ml and backend modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.src.calibrate_model import (
    TemperatureScaler,
    compute_ece,
    compute_brier,
    checkpoint_sha256,
    BEST_CKPT,
    BASELINE_SHA,
    VAL_DIR,
    TEST_DIR
)

def test_temperature_positive():
    """Verify that fitted temperature parameter is strictly positive."""
    scaler = TemperatureScaler()
    # Mock validation logits (100 samples, 10 classes)
    np.random.seed(42)
    mock_logits = np.random.randn(100, 10).astype(np.float32)
    mock_labels = np.random.randint(0, 10, size=100)
    
    T = scaler.fit(mock_logits, mock_labels)
    assert T > 0.0, f"Fitted temperature must be > 0, got {T}"
    assert scaler.temperature > 0.0, f"Scaler temperature attribute must be > 0, got {scaler.temperature}"

def test_calibrated_probabilities_valid_distribution():
    """Verify that calibrated probabilities sum to 1.0 and remain within [0, 1]."""
    scaler = TemperatureScaler()
    scaler.temperature = 1.5  # Test with T > 1
    
    np.random.seed(42)
    mock_logits = np.random.randn(50, 10).astype(np.float32)
    
    cal_probs = scaler.calibrated_probs(mock_logits)
    
    # 1. Bounds check [0, 1]
    assert (cal_probs >= 0.0).all(), "Calibrated probabilities contain negative values"
    assert (cal_probs <= 1.0).all(), "Calibrated probabilities contain values > 1.0"
    
    # 2. Probability sum check per sample == 1.0
    sums = cal_probs.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5, err_msg="Calibrated probabilities do not sum to 1")

def test_calibration_does_not_modify_model_weights():
    """Verify that calibration fitting and scaling do NOT mutate best_model.pt."""
    assert os.path.exists(BEST_CKPT), f"Checkpoint not found at {BEST_CKPT}"
    current_sha = checkpoint_sha256(BEST_CKPT)
    
    assert current_sha == "300a46ea8c2a3f9ff3f0e6fee0a9acdf19ecf5f8dc6b3e76ebe83c8a3b9d87d3", \
        f"best_model.pt SHA256 mismatch before test!\nExpected: 300a46ea8c2a3f9ff3f0e6fee0a9acdf19ecf5f8dc6b3e76ebe83c8a3b9d87d3\nGot:      {current_sha}"
    
    # Instantiate scaler and run fit/scale
    scaler = TemperatureScaler()
    mock_logits = np.random.randn(20, 10).astype(np.float32)
    mock_labels = np.random.randint(0, 10, size=20)
    scaler.fit(mock_logits, mock_labels)
    _ = scaler.calibrated_probs(mock_logits)
    
    post_sha = checkpoint_sha256(BEST_CKPT)
    assert post_sha == current_sha, "best_model.pt was modified during calibration test!"

def test_calibration_uses_validation_data_for_fitting():
    """Verify data isolation: validation directory is distinct from test directory."""
    assert os.path.normpath(VAL_DIR) != os.path.normpath(TEST_DIR), \
        "Validation and Test directories must be distinct for proper data isolation"
    assert "validation" in VAL_DIR.lower()
    assert "test" in TEST_DIR.lower()

def test_ece_and_brier_computation():
    """Verify ECE and Brier score metrics return non-negative floats."""
    probs = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.4, 0.6]
    ])
    labels = np.array([0, 1, 0])
    
    ece = compute_ece(probs, labels, n_bins=5)
    brier = compute_brier(probs, labels)
    
    assert isinstance(ece, float)
    assert ece >= 0.0
    assert isinstance(brier, float)
    assert brier >= 0.0
