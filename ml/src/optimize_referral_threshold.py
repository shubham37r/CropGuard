"""
CropGuard Phase 2D-1 — Empirical Expert Referral Threshold Optimization
========================================================================
Analyzes the Pareto trade-off between coverage, non-referred accuracy,
and bypassed errors across a threshold grid.

DATA ISOLATION MANDATE:
  - Validation split (2,720 images) is used ONLY for threshold exploration and selection.
  - Exactly ONE threshold is selected and locked based on validation set analysis.
  - The locked threshold is evaluated ONCE on the held-out TEST split (2,735 images).
  - Test set results are NEVER used to select or adjust the threshold.

CALIBRATION:
  - Uses existing Phase 2D temperature T = 0.530563.
  - Model weights (best_model.pt) are completely UNTOUCHED.
  - CLASSIFIER_MODE remains "mock".

Usage:
    python -m ml.src.optimize_referral_threshold
"""

import os
import sys
import json
import csv
import hashlib
import numpy as np
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from ml.src.calibrate_model import (
    TemperatureScaler,
    SplitDataset,
    build_model,
    collect_logits,
    checkpoint_sha256,
    eval_transform,
    BEST_CKPT,
    BASELINE_SHA,
    VAL_DIR,
    TEST_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
)


FITTED_TEMPERATURE = 0.530563
GRID_THRESHOLDS = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
FINE_GRID = [round(x, 1) for x in np.arange(50.0, 99.5, 2.5)]


def calculate_threshold_metrics(probs: np.ndarray, labels: np.ndarray, threshold_pct: float) -> dict:
    """
    Computes exact referral and non-referral metrics for a given confidence threshold (0-100%).

    Definitions:
      - Referred: confidence < threshold
      - Not Referred: confidence >= threshold
    """
    preds = probs.argmax(axis=1)
    confidences_pct = probs.max(axis=1) * 100.0
    total = len(labels)

    total_errors_mask = (preds != labels)
    total_errors_count = int(total_errors_mask.sum())

    referred_mask = confidences_pct < threshold_pct
    referred_count = int(referred_mask.sum())
    referral_rate = referred_count / total if total > 0 else 0.0

    non_referred_mask = confidences_pct >= threshold_pct
    non_referred_count = int(non_referred_mask.sum())
    non_referral_coverage = non_referred_count / total if total > 0 else 0.0

    correct_non_referred = int(((preds == labels) & non_referred_mask).sum())
    incorrect_non_referred = int(((preds != labels) & non_referred_mask).sum())  # Bypassed errors

    non_referred_acc = (correct_non_referred / non_referred_count) if non_referred_count > 0 else 1.0
    non_referred_err = (incorrect_non_referred / non_referred_count) if non_referred_count > 0 else 0.0

    bypass_rate_of_all_errors = (incorrect_non_referred / total_errors_count) if total_errors_count > 0 else 0.0
    bypass_rate_of_total_samples = incorrect_non_referred / total if total > 0 else 0.0

    return {
        "threshold_pct": round(float(threshold_pct), 1),
        "total_samples": total,
        "referred_samples": referred_count,
        "referral_rate_pct": round(float(referral_rate * 100.0), 2),
        "non_referred_samples": non_referred_count,
        "non_referral_coverage_pct": round(float(non_referral_coverage * 100.0), 2),
        "correct_non_referred": correct_non_referred,
        "incorrect_non_referred_bypassed": incorrect_non_referred,
        "non_referred_accuracy_pct": round(float(non_referred_acc * 100.0), 2),
        "non_referred_error_rate_pct": round(float(non_referred_err * 100.0), 2),
        "total_errors": total_errors_count,
        "bypassed_error_rate_of_all_errors_pct": round(float(bypass_rate_of_all_errors * 100.0), 2),
        "bypassed_error_rate_of_total_samples_pct": round(float(bypass_rate_of_total_samples * 100.0), 2),
    }


def evaluate_grid(probs: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> list[dict]:
    return [calculate_threshold_metrics(probs, labels, th) for th in thresholds]


def select_best_validation_threshold(val_cal_grid: list[dict]) -> tuple[float, str]:
    """
    Selects the optimal threshold on VALIDATION data based on Pareto trade-off.

    Selection Criteria:
      1. Target Non-Referred Accuracy >= 99.0% (non-referred error rate <= 1.0%).
      2. If multiple meet target, pick the one with lowest referral rate (highest coverage).
      3. Fallback: If no threshold reaches 99.0% non-referred accuracy, select the threshold
         that achieves max non-referred accuracy with referral rate <= 30%.
    """
    candidates_99 = [row for row in val_cal_grid if row["non_referred_accuracy_pct"] >= 99.0]
    if candidates_99:
        # Sort by referral rate ascending (highest coverage)
        best = min(candidates_99, key=lambda x: x["referral_rate_pct"])
        rationale = (
            f"Selected {best['threshold_pct']}% because it achieves >=99.0% non-referred accuracy "
            f"({best['non_referred_accuracy_pct']}%) on validation data with the lowest referral rate "
            f"({best['referral_rate_pct']}% referral, {best['non_referral_coverage_pct']}% coverage)."
        )
        return best["threshold_pct"], rationale

    # Fallback: Highest non-referred accuracy with referral rate <= 35%
    practical = [row for row in val_cal_grid if row["referral_rate_pct"] <= 35.0]
    if practical:
        best = max(practical, key=lambda x: x["non_referred_accuracy_pct"])
        rationale = (
            f"Fallback selection {best['threshold_pct']}%: highest non-referred accuracy "
            f"({best['non_referred_accuracy_pct']}%) among thresholds with practical referral rate <= 35% "
            f"({best['referral_rate_pct']}% referral)."
        )
        return best["threshold_pct"], rationale

    best = max(val_cal_grid, key=lambda x: x["non_referred_accuracy_pct"])
    rationale = f"Max accuracy selection {best['threshold_pct']}%: non-referred accuracy {best['non_referred_accuracy_pct']}%."
    return best["threshold_pct"], rationale


def generate_tradeoff_plot(val_raw_grid: list[dict], val_cal_grid: list[dict], selected_th: float, save_path: str):
    """Plots Referral Rate vs Non-Referred Accuracy & Bypassed Errors."""
    ths = [r["threshold_pct"] for r in val_cal_grid]
    cal_ref_rates = [r["referral_rate_pct"] for r in val_cal_grid]
    cal_accs = [r["non_referred_accuracy_pct"] for r in val_cal_grid]
    cal_bypassed = [r["incorrect_non_referred_bypassed"] for r in val_cal_grid]

    raw_ref_rates = [r["referral_rate_pct"] for r in val_raw_grid]
    raw_accs = [r["non_referred_accuracy_pct"] for r in val_raw_grid]
    raw_bypassed = [r["incorrect_non_referred_bypassed"] for r in val_raw_grid]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_acc = "tab:blue"
    color_byp = "tab:red"
    color_ref = "tab:green"

    ax1.set_xlabel("Confidence Threshold (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Non-Referred Accuracy (%)", color=color_acc, fontsize=11, fontweight="bold")
    line1 = ax1.plot(ths, cal_accs, "o-", color=color_acc, lw=2, label="Calibrated Non-Referred Acc (%)")
    line1_raw = ax1.plot(ths, raw_accs, "o--", color=color_acc, alpha=0.5, label="Raw Non-Referred Acc (%)")
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(95, 100.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Bypassed Errors (Count) / Referral Rate (%)", color=color_byp, fontsize=11, fontweight="bold")
    line2 = ax2.plot(ths, cal_bypassed, "s-", color=color_byp, lw=2, label="Calibrated Bypassed Errors (Count)")
    line2_raw = ax2.plot(ths, raw_bypassed, "s--", color=color_byp, alpha=0.5, label="Raw Bypassed Errors (Count)")
    line3 = ax2.plot(ths, cal_ref_rates, "^-", color=color_ref, lw=2, label="Calibrated Referral Rate (%)")
    ax2.tick_params(axis="y", labelcolor=color_byp)

    # Vertical line at selected threshold
    plt.axvline(x=selected_th, color="purple", linestyle=":", linewidth=2.5, label=f"Selected Threshold ({selected_th}%)")

    # Combine legends
    lines = line1 + line1_raw + line2 + line2_raw + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", fontsize=8)

    plt.title(
        f"CropGuard Validation Set — Threshold Trade-off Curve\n(Selected Optimal Threshold: {selected_th}% Calibrated)",
        fontsize=12,
        fontweight="bold"
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Trade-off plot saved: {save_path}")


def save_csv_table(val_raw_grid: list[dict], val_cal_grid: list[dict], save_path: str):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold_pct",
            "cal_referral_rate_pct",
            "cal_coverage_pct",
            "cal_non_referred_accuracy_pct",
            "cal_bypassed_errors_count",
            "cal_bypass_error_rate_of_total_pct",
            "raw_referral_rate_pct",
            "raw_coverage_pct",
            "raw_non_referred_accuracy_pct",
            "raw_bypassed_errors_count",
            "raw_bypass_error_rate_of_total_pct",
        ])
        for raw, cal in zip(val_raw_grid, val_cal_grid):
            writer.writerow([
                cal["threshold_pct"],
                cal["referral_rate_pct"],
                cal["non_referral_coverage_pct"],
                cal["non_referred_accuracy_pct"],
                cal["incorrect_non_referred_bypassed"],
                cal["bypassed_error_rate_of_total_samples_pct"],
                raw["referral_rate_pct"],
                raw["non_referral_coverage_pct"],
                raw["non_referred_accuracy_pct"],
                raw["incorrect_non_referred_bypassed"],
                raw["bypassed_error_rate_of_total_samples_pct"],
            ])
    print(f"[INFO] CSV threshold table saved: {save_path}")


def main():
    print("=" * 72)
    print("  CropGuard Phase 2D-1: Empirical Expert Referral Threshold Optimization")
    print("=" * 72)
    print()

    # ── 0. Integrity Pre-check ───────────────────────────────────────────────
    pre_sha = checkpoint_sha256(BEST_CKPT)
    print(f"[INTEGRITY] best_model.pt SHA256 (pre):  {pre_sha}")
    assert pre_sha == BASELINE_SHA, f"Checkpoint modified! Expected {BASELINE_SHA}, got {pre_sha}"
    print(f"[INTEGRITY] Checkpoint integrity: VERIFIED")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── 1. Load model (eval, frozen) ─────────────────────────────────────────
    raw_ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    sd = raw_ckpt["model_state_dict"] if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt else raw_ckpt

    model = build_model()
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[INFO] Model loaded from {BEST_CKPT} (frozen)")
    print(f"[INFO] Fixed calibration temperature T = {FITTED_TEMPERATURE}")
    print()

    # ── 2. Collect VALIDATION split logits (FOR THRESHOLD SELECTION ONLY) ─────
    print("[STEP 1] Collecting VALIDATION set logits (2,720 images)...")
    val_ds = SplitDataset(VAL_DIR, transform=eval_transform)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    val_logits, val_labels = collect_logits(model, val_loader, device)
    print(f"  Validation images loaded: {len(val_labels)}")

    scaler = TemperatureScaler()
    scaler.temperature = FITTED_TEMPERATURE

    # Compute raw & calibrated probabilities for validation set
    exp_val_raw = np.exp(val_logits - val_logits.max(axis=1, keepdims=True))
    val_raw_probs = exp_val_raw / exp_val_raw.sum(axis=1, keepdims=True)
    val_cal_probs = scaler.calibrated_probs(val_logits)

    # ── 3. Evaluate Threshold Grid on VALIDATION Set ──────────────────────────
    print("\n[STEP 2] Evaluating threshold grid on VALIDATION set...")
    val_raw_grid = evaluate_grid(val_raw_probs, val_labels, GRID_THRESHOLDS)
    val_cal_grid = evaluate_grid(val_cal_probs, val_labels, GRID_THRESHOLDS)

    print("\n" + "=" * 80)
    print("  VALIDATION SET — CALIBRATED CONFIDENCE THRESHOLD GRID")
    print("=" * 80)
    print(f"  {'Thresh(%)':<10} {'Ref Count':<10} {'Ref Rate(%)':<12} {'Coverage(%)':<12} {'NonRef Acc(%)':<15} {'Bypassed Errs':<14} {'Bypass Rate(Total %)':<18}")
    print("  " + "-" * 88)
    for r in val_cal_grid:
        print(
            f"  {r['threshold_pct']:<10.1f} {r['referred_samples']:<10d} {r['referral_rate_pct']:<12.2f} "
            f"{r['non_referral_coverage_pct']:<12.2f} {r['non_referred_accuracy_pct']:<15.2f} "
            f"{r['incorrect_non_referred_bypassed']:<14d} {r['bypassed_error_rate_of_total_samples_pct']:<18.2f}"
        )

    # ── 4. Select and Lock ONE Optimal Threshold from VALIDATION Data ─────────
    selected_th, rationale = select_best_validation_threshold(val_cal_grid)
    print("\n" + "=" * 80)
    print(f"  OPTIMAL THRESHOLD SELECTION (VALIDATION SET ONLY)")
    print("=" * 80)
    print(f"  LOCKED THRESHOLD: {selected_th}% Calibrated Confidence")
    print(f"  RATIONALE:        {rationale}")
    print("=" * 80)
    print()

    # ── 5. ONE-TIME Held-Out TEST Set Evaluation (LOCKED THRESHOLD) ───────────
    print("[STEP 3] ONE-TIME evaluation of locked threshold on held-out TEST set (2,735 images)...")
    test_ds = SplitDataset(TEST_DIR, transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    test_logits, test_labels = collect_logits(model, test_loader, device)


    exp_test_raw = np.exp(test_logits - test_logits.max(axis=1, keepdims=True))
    test_raw_probs = exp_test_raw / exp_test_raw.sum(axis=1, keepdims=True)
    test_cal_probs = scaler.calibrated_probs(test_logits)

    # Evaluated at locked threshold
    test_cal_locked = calculate_threshold_metrics(test_cal_probs, test_labels, selected_th)
    test_raw_locked = calculate_threshold_metrics(test_raw_probs, test_labels, selected_th)

    # Compare against default 70% threshold
    test_cal_70 = calculate_threshold_metrics(test_cal_probs, test_labels, 70.0)
    test_raw_70 = calculate_threshold_metrics(test_raw_probs, test_labels, 70.0)

    # Full test grid for reporting completeness
    test_cal_grid = evaluate_grid(test_cal_probs, test_labels, GRID_THRESHOLDS)
    test_raw_grid = evaluate_grid(test_raw_probs, test_labels, GRID_THRESHOLDS)

    print("\n" + "=" * 80)
    print(f"  HELD-OUT TEST SET EVALUATION (LOCKED THRESHOLD = {selected_th}%)")
    print("=" * 80)
    print(f"  Total Test Images:                {test_cal_locked['total_samples']}")
    print(f"  Total Errors in Test Model:       {test_cal_locked['total_errors']}")
    print(f"  Referred Count:                   {test_cal_locked['referred_samples']}")
    print(f"  Referral Rate:                    {test_cal_locked['referral_rate_pct']}%")
    print(f"  Non-Referral Coverage:            {test_cal_locked['non_referral_coverage_pct']}%")
    print(f"  Non-Referred Accuracy:            {test_cal_locked['non_referred_accuracy_pct']}%")
    print(f"  Bypassed Errors (Count):          {test_cal_locked['incorrect_non_referred_bypassed']}")
    print(f"  Bypassed Error Rate (% of total): {test_cal_locked['bypassed_error_rate_of_total_samples_pct']}%")
    print(f"  Bypassed Error Rate (% of errors):{test_cal_locked['bypassed_error_rate_of_all_errors_pct']}%")
    print("-" * 80)
    print(f"  VS Baseline 70% Calibrated Threshold:")
    print(f"    - 70% Calibrated Referral Rate: {test_cal_70['referral_rate_pct']}% | Bypassed Errors: {test_cal_70['incorrect_non_referred_bypassed']}")
    print(f"    - {selected_th}% Calibrated Referral Rate: {test_cal_locked['referral_rate_pct']}% | Bypassed Errors: {test_cal_locked['incorrect_non_referred_bypassed']}")
    print(f"  VS Baseline 70% Raw Threshold:")
    print(f"    - 70% Raw Referral Rate:        {test_raw_70['referral_rate_pct']}% | Bypassed Errors: {test_raw_70['incorrect_non_referred_bypassed']}")
    print("=" * 80)
    print()

    # ── 6. Save Reports & Visualizations ──────────────────────────────────────
    plot_path = os.path.join(REPORTS_DIR, "referral_threshold_tradeoff.png")
    generate_tradeoff_plot(val_raw_grid, val_cal_grid, selected_th, plot_path)

    csv_path = os.path.join(REPORTS_DIR, "referral_threshold_table.csv")
    save_csv_table(val_raw_grid, val_cal_grid, csv_path)

    json_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 2D-1 — Empirical Expert Referral Threshold Optimization",
        "reproducibility": {
            "checkpoint_path": BEST_CKPT,
            "checkpoint_sha256": pre_sha,
            "calibration_temperature_T": FITTED_TEMPERATURE,
            "random_seed": RANDOM_SEED,
            "validation_sample_count": len(val_labels),
            "test_sample_count": len(test_labels),
            "threshold_grid_evaluated": GRID_THRESHOLDS,
        },
        "validation_selection": {
            "selected_threshold_pct": selected_th,
            "selection_rationale": rationale,
            "validation_calibrated_grid": val_cal_grid,
            "validation_raw_grid": val_raw_grid,
        },
        "held_out_test_evaluation": {
            "locked_threshold_pct": selected_th,
            "calibrated_locked_performance": test_cal_locked,
            "raw_locked_performance": test_raw_locked,
            "calibrated_70pct_baseline": test_cal_70,
            "raw_70pct_baseline": test_raw_70,
            "test_calibrated_grid": test_cal_grid,
            "test_raw_grid": test_raw_grid,
        },
        "production_impact": {
            "classifier_mode": "mock",
            "production_threshold_changed": False,
            "real_classifier_modified": False,
            "frontend_modified": False,
            "weights_modified": False,
            "safety_recommendation": (
                f"Validation data supports locking a calibrated threshold of {selected_th}%. "
                f"This reduces bypassed errors on the test set from {test_cal_70['incorrect_non_referred_bypassed']} "
                f"(under 70% calibrated) down to {test_cal_locked['incorrect_non_referred_bypassed']} "
                f"while maintaining {test_cal_locked['non_referred_accuracy_pct']}% non-referred accuracy. "
                "Do not activate in live production until real-world field/OOD data validation is conducted."
            ),
        },
    }

    json_path = os.path.join(REPORTS_DIR, "referral_threshold_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)
    print(f"[INFO] JSON report saved: {json_path}")

    # Generate Markdown Report
    val_sel_row = next(r for r in val_cal_grid if r["threshold_pct"] == selected_th)

    md_content = f"""# CropGuard Phase 2D-1 — Empirical Expert Referral Threshold Optimization Report

**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Model**: EfficientNet-B0 (PyTorch)
**Checkpoint**: `ml/models/checkpoints/best_model.pt`
**SHA256**: `{pre_sha}`
**Calibration Temperature $T$**: `{FITTED_TEMPERATURE}` (Fitted on Validation set in Phase 2D)
**Data Isolation**: Validation Set ({len(val_labels):,} samples) used for threshold selection; Test Set ({len(test_labels):,} samples) evaluated ONCE on locked threshold.

---

## Executive Summary & Selection Result

- **Selected Calibrated Threshold**: **`{selected_th}%`**
- **Selection Rationale**: {rationale}
- **Validation Non-Referred Accuracy**: **`{val_sel_row['non_referred_accuracy_pct']}%`**
- **Validation Referral Rate**: **`{val_sel_row['referral_rate_pct']}%`** (`{val_sel_row['non_referral_coverage_pct']}%` coverage)
- **Validation Bypassed Errors**: **`{val_sel_row['incorrect_non_referred_bypassed']}`** out of `{val_sel_row['total_errors']}` total validation errors (`{val_sel_row['bypassed_error_rate_of_all_errors_pct']}%` of errors bypass)

---

## 1. Validation Set Threshold Grid (Threshold Selection Data)

> **Definition**: A prediction is REFERRED when confidence < threshold; NOT REFERRED (covered) when confidence >= threshold.

### Calibrated Confidence ($T = 0.530563$)

| Threshold (%) | Referred Count | Referral Rate (%) | Coverage (%) | Non-Referred Accuracy (%) | Bypassed Errors (Count) | Bypass Rate (% of Total) |
|---|---|---|---|---|---|---|
"""
    for r in val_cal_grid:
        sel_mark = " **(SELECTED)**" if r["threshold_pct"] == selected_th else ""
        md_content += f"| {r['threshold_pct']:.1f}%{sel_mark} | {r['referred_samples']} | {r['referral_rate_pct']:.2f}% | {r['non_referral_coverage_pct']:.2f}% | {r['non_referred_accuracy_pct']:.2f}% | {r['incorrect_non_referred_bypassed']} | {r['bypassed_error_rate_of_total_samples_pct']:.2f}% |\n"

    md_content += """
### Raw Softmax Confidence (Comparison Baseline)

| Threshold (%) | Referred Count | Referral Rate (%) | Coverage (%) | Non-Referred Accuracy (%) | Bypassed Errors (Count) | Bypass Rate (% of Total) |
|---|---|---|---|---|---|---|
"""
    for r in val_raw_grid:
        md_content += f"| {r['threshold_pct']:.1f}% | {r['referred_samples']} | {r['referral_rate_pct']:.2f}% | {r['non_referral_coverage_pct']:.2f}% | {r['non_referred_accuracy_pct']:.2f}% | {r['incorrect_non_referred_bypassed']} | {r['bypassed_error_rate_of_total_samples_pct']:.2f}% |\n"

    md_content += f"""
---

## 2. One-Time Held-Out Test Set Evaluation (Locked Threshold = {selected_th}%)

> **Held-Out Test Set**: 2,735 images | Total Model Errors: {test_cal_locked['total_errors']}

| Metric | Locked {selected_th}% (Calibrated) | Baseline 70% (Calibrated) | Baseline 70% (Raw) | Locked {selected_th}% (Raw) |
|---|---|---|---|---|
| **Referred Count** | **{test_cal_locked['referred_samples']}** | {test_cal_70['referred_samples']} | {test_raw_70['referred_samples']} | {test_raw_locked['referred_samples']} |
| **Referral Rate (%)** | **{test_cal_locked['referral_rate_pct']}%** | {test_cal_70['referral_rate_pct']}% | {test_raw_70['referral_rate_pct']}% | {test_raw_locked['referral_rate_pct']}% |
| **Non-Referral Coverage (%)** | **{test_cal_locked['non_referral_coverage_pct']}%** | {test_cal_70['non_referral_coverage_pct']}% | {test_raw_70['non_referral_coverage_pct']}% | {test_raw_locked['non_referral_coverage_pct']}% |
| **Non-Referred Accuracy (%)** | **{test_cal_locked['non_referred_accuracy_pct']}%** | {test_cal_70['non_referred_accuracy_pct']}% | {test_raw_70['non_referred_accuracy_pct']}% | {test_raw_locked['non_referred_accuracy_pct']}% |
| **Bypassed Errors (Count)** | **{test_cal_locked['incorrect_non_referred_bypassed']}** | {test_cal_70['incorrect_non_referred_bypassed']} | {test_raw_70['incorrect_non_referred_bypassed']} | {test_raw_locked['incorrect_non_referred_bypassed']} |
| **Bypass Rate (% of Total)** | **{test_cal_locked['bypassed_error_rate_of_total_samples_pct']}%** | {test_cal_70['bypassed_error_rate_of_total_samples_pct']}% | {test_raw_70['bypassed_error_rate_of_total_samples_pct']}% | {test_raw_locked['bypassed_error_rate_of_total_samples_pct']}% |

---

## Key Findings & Safety Assessment

1. **Why 70% Calibrated Threshold Was Unsafe**:
   - In Phase 2D, temperature scaling $T=0.5306$ sharpened confidences, causing **25 out of 59 errors** (42.37% of errors) to bypass referral at the 70% threshold.
2. **Impact of Raising Threshold to {selected_th}% Calibrated**:
   - On the held-out test set, raising the calibrated threshold to **{selected_th}%** reduces bypassed errors from **{test_cal_70['incorrect_non_referred_bypassed']} down to {test_cal_locked['incorrect_non_referred_bypassed']}** (a **{round((test_cal_70['incorrect_non_referred_bypassed'] - test_cal_locked['incorrect_non_referred_bypassed']) / test_cal_70['incorrect_non_referred_bypassed'] * 100, 1)}% reduction in bypassed errors**).
   - Non-referred accuracy on the test set increases to **{test_cal_locked['non_referred_accuracy_pct']}%**.
   - Referral rate increases to **{test_cal_locked['referral_rate_pct']}%** ({test_cal_locked['non_referral_coverage_pct']}% of cases automated without officer intervention).

---

## Production Safety Checklist

| Rule | Status |
|---|---|
| Model weights (`best_model.pt`) modified | NO (SHA256 verified) |
| `CLASSIFIER_MODE` | `mock` |
| `RealClassifierService` modified | NO |
| Production 70% threshold modified | NO (saved as report only) |
| Test set used for threshold selection | NO (Validation set only) |
"""

    md_path = os.path.join(REPORTS_DIR, "referral_threshold_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[INFO] Markdown report saved: {md_path}")

    # ── 7. Post-run Checkpoint SHA Check ──────────────────────────────────────
    post_sha = checkpoint_sha256(BEST_CKPT)
    assert post_sha == BASELINE_SHA, f"CRITICAL: best_model.pt SHA changed! Pre: {pre_sha}, Post: {post_sha}"
    print(f"[INTEGRITY] best_model.pt SHA256 (post): {post_sha}")
    print(f"[INTEGRITY] Checkpoint integrity: VERIFIED (unchanged)")
    print()
    print("=" * 72)
    print("  Phase 2D-1 Referral Threshold Optimization Complete")
    print("=" * 72)


if __name__ == "__main__":
    main()
