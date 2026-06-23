"""Shared classification metrics so both models report identical numbers.

The dataset is class-imbalanced (more controls than patients in CALSNIC), so
accuracy alone is misleading — ``binary_metrics`` always reports balanced
accuracy, F1, sensitivity/specificity, ROC-AUC and PR-AUC alongside it, plus a
confusion matrix. The bootstrap / DeLong CI and calibration helpers are used by
the evaluation stage on the held-out test set.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _has_both_classes(labels) -> bool:
    return len({int(x) for x in labels}) >= 2


def safe_auc(labels, probs) -> float:
    return float(roc_auc_score(labels, probs)) if _has_both_classes(labels) else float("nan")


def safe_pr_auc(labels, probs) -> float:
    return float(average_precision_score(labels, probs)) if _has_both_classes(labels) else float("nan")


def youden_threshold(labels, probs) -> float:
    """Threshold maximizing sensitivity + specificity − 1 on the given set.

    Tuned on validation, then frozen and applied to the test set so the
    operating point is never chosen on the data it is scored on.
    """
    if not _has_both_classes(labels):
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr - fpr
    # sklearn >= 1.3 prepends an `inf` threshold (the "predict nothing positive"
    # operating point); mask non-finite thresholds so Youden never returns inf.
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    j = np.where(finite, j, -np.inf)
    return float(thresholds[int(np.argmax(j))])


def sens_spec(labels, preds) -> tuple[float, float]:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return float(sens), float(spec)


def binary_metrics(labels, probs, threshold: float = 0.5) -> dict:
    """Full metric suite for a set of labels + positive-class probabilities."""
    int_labels = [int(x) for x in labels]
    preds = [1 if p >= threshold else 0 for p in probs]
    sens, spec = sens_spec(int_labels, preds)
    return {
        "threshold": float(threshold),
        "n": len(int_labels),
        "accuracy": float(accuracy_score(int_labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(int_labels, preds)),
        "precision": float(precision_score(int_labels, preds, zero_division=0)),
        "recall": float(recall_score(int_labels, preds, zero_division=0)),
        "sensitivity": sens,
        "specificity": spec,
        "f1_score": float(f1_score(int_labels, preds, zero_division=0)),
        "roc_auc": safe_auc(int_labels, probs),
        "pr_auc": safe_pr_auc(int_labels, probs),
        "confusion_matrix": confusion_matrix(int_labels, preds, labels=[0, 1]).tolist(),
    }


# ── interval / calibration helpers (used by the evaluation stage) ──────────


def bootstrap_auc_ci(labels, probs, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for ROC-AUC. Returns (lo, hi) or (nan, nan)."""
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    probs_arr = np.asarray(probs)
    if not _has_both_classes(labels):
        return float("nan"), float("nan")
    aucs: list[float] = []
    n = len(labels_arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        l = labels_arr[idx]
        if len(set(l.tolist())) < 2:
            continue
        aucs.append(roc_auc_score(l, probs_arr[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _delong_var(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Single-AUC + its DeLong variance (Sun & Xu, 2014). Returns (auc, variance)."""
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    order = np.concatenate([pos, neg]).argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    pos_ranks = ranks[:m]
    auc = (pos_ranks.sum() - m * (m + 1) / 2) / (m * n)
    v10 = np.zeros(m)
    v01 = np.zeros(n)
    for i in range(m):
        v10[i] = (np.sum(pos[i] > neg) + 0.5 * np.sum(pos[i] == neg)) / n
    for j in range(n):
        v01[j] = (np.sum(pos > neg[j]) + 0.5 * np.sum(pos == neg[j])) / m
    s10 = v10.var(ddof=1) if m > 1 else 0.0
    s01 = v01.var(ddof=1) if n > 1 else 0.0
    return float(auc), float(s10 / m + s01 / n)


def delong_ci(labels, probs, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for AUC via logit-transformed DeLong; bootstrap fallback at edges."""
    labels_arr = np.asarray(labels)
    probs_arr = np.asarray(probs)
    if not _has_both_classes(labels):
        return float("nan"), float("nan")
    auc, var = _delong_var(labels_arr, probs_arr)
    if not math.isfinite(auc) or var <= 0 or auc in (0.0, 1.0):
        return bootstrap_auc_ci(labels, probs)
    logit_auc = math.log(auc / (1 - auc))
    se_logit = math.sqrt(var) / (auc * (1 - auc))
    z = 1.959963984540054  # two-sided 95%
    lo = logit_auc - z * se_logit
    hi = logit_auc + z * se_logit
    return float(1 / (1 + math.exp(-lo))), float(1 / (1 + math.exp(-hi)))


def expected_calibration_error(labels, probs, n_bins: int = 10) -> float:
    """ECE with equal-width bins on [0, 1]."""
    if not len(labels):
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    labels_arr = np.asarray(labels, dtype=float)
    probs_arr = np.asarray(probs, dtype=float)
    ece = 0.0
    n = len(labels_arr)
    for i in range(n_bins):
        in_bin = (probs_arr >= bins[i]) & (probs_arr < bins[i + 1] if i < n_bins - 1 else probs_arr <= bins[i + 1])
        if not in_bin.any():
            continue
        acc = labels_arr[in_bin].mean()
        conf = probs_arr[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(acc - conf)
    return float(ece)


def brier(labels, probs) -> float:
    return float(brier_score_loss([int(x) for x in labels], probs)) if len(labels) else float("nan")
