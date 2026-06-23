"""Shared metric suite + CI / calibration helpers."""

from __future__ import annotations

import math

import numpy as np

from als.training import metrics as M


def test_binary_metrics_perfect():
    labels = [0, 0, 1, 1]
    probs = [0.1, 0.2, 0.8, 0.9]
    m = M.binary_metrics(labels, probs, threshold=0.5)
    assert m["accuracy"] == 1.0 and m["balanced_accuracy"] == 1.0
    assert m["roc_auc"] == 1.0 and m["sensitivity"] == 1.0 and m["specificity"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_balanced_accuracy_reflects_imbalance():
    # Predict everything negative on a 9:1 imbalance → high accuracy, bal-acc 0.5.
    labels = [0] * 9 + [1]
    probs = [0.1] * 10
    m = M.binary_metrics(labels, probs, threshold=0.5)
    assert m["accuracy"] == 0.9
    assert math.isclose(m["balanced_accuracy"], 0.5)


def test_youden_threshold_separates():
    labels = [0] * 20 + [1] * 20
    probs = list(np.linspace(0.0, 0.4, 20)) + list(np.linspace(0.6, 1.0, 20))
    thr = M.youden_threshold(labels, probs)
    assert 0.4 <= thr <= 0.6


def test_bootstrap_and_delong_ci_finite():
    rng = np.random.default_rng(0)
    labels = [0] * 40 + [1] * 40
    probs = list(rng.normal(0.3, 0.1, 40)) + list(rng.normal(0.7, 0.1, 40))
    lo, hi = M.bootstrap_auc_ci(labels, probs, n_boot=300)
    assert 0.0 <= lo <= hi <= 1.0
    dlo, dhi = M.delong_ci(labels, probs)
    assert math.isfinite(dlo) and math.isfinite(dhi) and 0.0 <= dlo <= dhi <= 1.0


def test_ece_zero_for_perfect_calibration():
    labels = [0, 1] * 50
    probs = [0.0, 1.0] * 50
    assert M.expected_calibration_error(labels, probs, n_bins=10) < 1e-6


def test_single_class_is_nan_safe():
    assert math.isnan(M.safe_auc([1, 1, 1], [0.2, 0.3, 0.4]))
    lo, hi = M.delong_ci([0, 0, 0], [0.1, 0.2, 0.3])
    assert math.isnan(lo) and math.isnan(hi)
