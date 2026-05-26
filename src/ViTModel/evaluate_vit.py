"""
Evaluate the trained spatial multi-modal ViT on the held-out test split.

Reports:
  - ROC-AUC with bootstrap 95% CI and DeLong CI.
  - PR-AUC, Brier score, expected calibration error (ECE).
  - Accuracy / precision / recall / F1 / sensitivity / specificity at the
    val-tuned threshold (saved in the checkpoint by train_vit.py).
  - Per-site metrics when more than one site is present.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from dataset import ALSSpatialFeatureDataset, indices_from, load_or_build_splits  # noqa: E402
from model import SpatialMultiModalViT                                            # noqa: E402
from paths import build_runtime_paths, ensure_output_dirs                         # noqa: E402

SEED = 42
BATCH_SIZE = 8


# ── statistical helpers ──────────────────────────────────────────────────


def _bootstrap_auc_ci(labels: list[int], probs: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for ROC-AUC. Returns (lo, hi) or (nan, nan)."""
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    probs_arr = np.asarray(probs)
    if len(set(labels)) < 2:
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
    """
    Compute single-AUC + its DeLong variance.
    Reference: Sun & Xu, 2014 — fast implementation O(n log n).
    Returns (auc, variance).
    """
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    m = len(pos)
    n = len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")

    order = np.concatenate([pos, neg]).argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    pos_ranks = ranks[:m]
    neg_ranks = ranks[m:]

    auc = (pos_ranks.sum() - m * (m + 1) / 2) / (m * n)

    # Structural components
    v10 = np.zeros(m)
    v01 = np.zeros(n)
    for i in range(m):
        v10[i] = (np.sum(pos[i] > neg) + 0.5 * np.sum(pos[i] == neg)) / n
    for j in range(n):
        v01[j] = (np.sum(pos > neg[j]) + 0.5 * np.sum(pos == neg[j])) / m
    s10 = v10.var(ddof=1) if m > 1 else 0.0
    s01 = v01.var(ddof=1) if n > 1 else 0.0
    var = s10 / m + s01 / n
    return float(auc), float(var)


def _delong_ci(labels: list[int], probs: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for AUC via logit-transformed DeLong (Sun & Xu)."""
    labels_arr = np.asarray(labels)
    probs_arr = np.asarray(probs)
    if len(set(labels)) < 2:
        return float("nan"), float("nan")
    auc, var = _delong_var(labels_arr, probs_arr)
    if not math.isfinite(auc) or var <= 0 or auc in (0.0, 1.0):
        # Fall back to bootstrap if logit is undefined.
        return _bootstrap_auc_ci(labels, probs)
    logit_auc = math.log(auc / (1 - auc))
    se_logit = math.sqrt(var) / (auc * (1 - auc))
    z = 1.959963984540054  # two-sided 95%
    lo = logit_auc - z * se_logit
    hi = logit_auc + z * se_logit
    return float(1 / (1 + math.exp(-lo))), float(1 / (1 + math.exp(-hi)))


def _expected_calibration_error(labels: list[int], probs: list[float], n_bins: int = 10) -> float:
    """ECE with equal-width bins on [0, 1]."""
    if not labels:
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


def _sens_spec(labels: list[int], preds: list[int]) -> tuple[float, float]:
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return float(sens), float(spec)


# ── evaluator ────────────────────────────────────────────────────────────


def _resolve_device(prefer: str) -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(
    features_dir: str | None = None,
    artifacts_dir: str | None = None,
    splits_path: str | None = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2,
    device_name: str = "auto",
    threshold_override: float | None = None,
    bootstrap_n: int = 2000,
) -> None:
    ensure_output_dirs()
    runtime_features_dir, runtime_artifacts_dir, runtime_checkpoint_dir, runtime_metrics_dir, runtime_checkpoint_path = (
        build_runtime_paths(features_dir=features_dir, artifacts_dir=artifacts_dir)
    )
    runtime_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runtime_metrics_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device_name)

    if not runtime_checkpoint_path.exists():
        print(f"Error: checkpoint not found: {runtime_checkpoint_path}")
        return

    dataset = ALSSpatialFeatureDataset(features_dir=str(runtime_features_dir))
    if len(dataset) == 0:
        print(f"No spatial features in {runtime_features_dir}.")
        return

    chosen_splits_path = Path(splits_path) if splits_path else (runtime_artifacts_dir.parent / "cnn_multimodal" / "splits.json")
    if not chosen_splits_path.exists():
        chosen_splits_path = runtime_artifacts_dir / "splits.json"
    splits = load_or_build_splits(dataset.samples, chosen_splits_path)
    test_idx = indices_from(dataset.samples, splits, "test")
    if not test_idx:
        print("Test split is empty.")
        return

    test_set = Subset(dataset, test_idx)
    pin_memory = device.type == "cuda"
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

    blob = torch.load(runtime_checkpoint_path, map_location=device, weights_only=False)
    cfg = blob["config"]
    model = SpatialMultiModalViT(
        in_channels=cfg["in_channels"],
        spatial_shape=tuple(cfg["spatial_shape"]),
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"],
        attn_dropout=cfg.get("attn_dropout", 0.1),
        modality_dropout_prob=cfg.get("modality_dropout_prob", 0.0),
    ).to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()

    threshold = float(threshold_override) if threshold_override is not None else float(blob.get("best_val_threshold", 0.5))

    labels: list[float] = []
    probs: list[float] = []
    ids: list[str] = []
    with torch.no_grad():
        for x, y, sids in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(1)
            p = torch.sigmoid(logits.float()).cpu().numpy().reshape(-1).tolist()
            probs.extend(p)
            labels.extend(y.cpu().numpy().reshape(-1).tolist())
            ids.extend(list(sids))

    int_labels = [int(l) for l in labels]
    preds = [1 if p >= threshold else 0 for p in probs]
    sens, spec = _sens_spec(int_labels, preds)
    boot_lo, boot_hi = _bootstrap_auc_ci(int_labels, probs, n_boot=bootstrap_n, seed=SEED)
    dl_lo, dl_hi = _delong_ci(int_labels, probs)

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(int_labels, preds)),
        "precision": float(precision_score(int_labels, preds, zero_division=0)),
        "recall": float(recall_score(int_labels, preds, zero_division=0)),
        "f1_score": float(f1_score(int_labels, preds, zero_division=0)),
        "sensitivity": sens,
        "specificity": spec,
        "roc_auc": float(roc_auc_score(int_labels, probs)) if len(set(int_labels)) > 1 else float("nan"),
        "roc_auc_bootstrap_95ci": [boot_lo, boot_hi],
        "roc_auc_delong_95ci": [dl_lo, dl_hi],
        "pr_auc": float(average_precision_score(int_labels, probs)) if len(set(int_labels)) > 1 else float("nan"),
        "brier_score": float(brier_score_loss(int_labels, probs)) if len(int_labels) > 0 else float("nan"),
        "ece_10bin": _expected_calibration_error(int_labels, probs, n_bins=10),
        "confusion_matrix": confusion_matrix(int_labels, preds, labels=[0, 1]).tolist(),
        "num_test_samples": len(test_set),
        "best_val_metric_name": blob.get("best_val_metric_name", "roc_auc"),
        "best_val_metric": float(blob.get("best_val_metric", float("nan"))),
        "epoch_of_best": int(blob.get("epoch", -1)),
    }

    # Per-site metrics if more than one site is present.
    sites = defaultdict(lambda: {"labels": [], "probs": [], "preds": []})
    by_id = {s.sample_id: s for s in dataset.samples}
    for sid, l, p, pr in zip(ids, int_labels, probs, preds):
        site = by_id[sid].site or "UNK"
        sites[site]["labels"].append(l)
        sites[site]["probs"].append(p)
        sites[site]["preds"].append(pr)
    if len(sites) > 1:
        per_site = {}
        for site, d in sites.items():
            if len(set(d["labels"])) > 1:
                site_auc = float(roc_auc_score(d["labels"], d["probs"]))
            else:
                site_auc = float("nan")
            per_site[site] = {
                "n": len(d["labels"]),
                "accuracy": float(accuracy_score(d["labels"], d["preds"])),
                "roc_auc": site_auc,
                "sensitivity": _sens_spec(d["labels"], d["preds"])[0],
                "specificity": _sens_spec(d["labels"], d["preds"])[1],
            }
        metrics["per_site"] = per_site

    out_metrics = runtime_metrics_dir / "vit_evaluation_metrics.json"
    out_preds = runtime_metrics_dir / "vit_test_predictions.json"
    out_metrics.write_text(json.dumps(metrics, indent=2))
    out_preds.write_text(json.dumps(
        [
            {"id": i, "label": int(l), "prob_als": p, "pred": int(pr)}
            for i, l, p, pr in zip(ids, int_labels, probs, preds)
        ],
        indent=2,
    ))

    print("--- Spatial ViT Evaluation ---")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics    : {out_metrics}")
    print(f"Saved predictions: {out_preds}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate spatial multi-modal ViT.")
    p.add_argument("--features-dir", type=str, default=None)
    p.add_argument("--artifacts-dir", type=str, default=None)
    p.add_argument("--splits-path", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--threshold", type=float, default=None,
                   help="Override the val-tuned threshold stored in the checkpoint.")
    p.add_argument("--bootstrap-n", type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
        splits_path=args.splits_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        threshold_override=args.threshold,
        bootstrap_n=args.bootstrap_n,
    )
