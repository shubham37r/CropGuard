"""
Unit tests for CropGuard Phase 2D-1 Referral Threshold Calculation & Data Isolation Logic
"""

import os
import sys
import numpy as np
import pytest

# Ensure ml and backend modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.src.optimize_referral_threshold import (
    calculate_threshold_metrics,
    evaluate_grid,
    select_best_validation_threshold,
    GRID_THRESHOLDS,
)

def test_calculate_threshold_metrics_basic():
    """Verify exact counting for referred vs non-referred samples."""
    # 4 samples: probabilities for 2 classes
    # Sample 0: prob 0.85 -> true label 0 (correct, conf 85%)
    # Sample 1: prob 0.65 -> true label 0 (correct, conf 65%)
    # Sample 2: prob 0.95 -> true label 1, pred 0 (incorrect, conf 95%)
    # Sample 3: prob 0.40 -> true label 1, pred 0 (incorrect, conf 60%)
    probs = np.array([
        [0.85, 0.15],
        [0.65, 0.35],
        [0.95, 0.05],
        [0.60, 0.40]
    ])
    labels = np.array([0, 0, 1, 1])

    # Test threshold = 70.0%
    res = calculate_threshold_metrics(probs, labels, 70.0)

    assert res["total_samples"] == 4
    # Referred: conf < 70.0 -> Sample 1 (65%) and Sample 3 (60%) -> count = 2
    assert res["referred_samples"] == 2
    assert res["referral_rate_pct"] == 50.0

    # Non-referred: conf >= 70.0 -> Samples 0 (85%) and 2 (95%) -> count = 2
    assert res["non_referred_samples"] == 2
    assert res["non_referral_coverage_pct"] == 50.0


    # Sample 0 (85%): pred=0, label=0 -> Correct
    # Sample 2 (95%): pred=0, label=1 -> Incorrect (Bypassed Error)
    assert res["correct_non_referred"] == 1
    assert res["incorrect_non_referred_bypassed"] == 1
    assert res["non_referred_accuracy_pct"] == 50.0

def test_higher_threshold_increases_or_maintains_referral_rate():
    """Verify monotonic non-decreasing referral rate as threshold increases."""
    np.random.seed(42)
    probs = np.random.dirichlet(np.ones(10), size=100)
    labels = np.random.randint(0, 10, size=100)

    grid = evaluate_grid(probs, labels, GRID_THRESHOLDS)
    ref_rates = [r["referral_rate_pct"] for r in grid]

    for i in range(len(ref_rates) - 1):
        assert ref_rates[i + 1] >= ref_rates[i], \
            f"Referral rate must be non-decreasing: {ref_rates[i]} -> {ref_rates[i+1]}"

def test_selection_logic_prefers_safety_target():
    """Verify threshold selection logic selects threshold meeting target accuracy with lowest referral rate."""
    val_grid = [
        {"threshold_pct": 50.0, "non_referred_accuracy_pct": 98.0, "referral_rate_pct": 2.0, "non_referral_coverage_pct": 98.0},
        {"threshold_pct": 70.0, "non_referred_accuracy_pct": 99.2, "referral_rate_pct": 5.0, "non_referral_coverage_pct": 95.0},
        {"threshold_pct": 80.0, "non_referred_accuracy_pct": 99.5, "referral_rate_pct": 8.0, "non_referral_coverage_pct": 92.0},
    ]

    selected_th, rationale = select_best_validation_threshold(val_grid)
    assert selected_th == 70.0, f"Expected 70.0%, got {selected_th}"
    assert "99.0%" in rationale
