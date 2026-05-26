"""
Evaluate the tri-stream CNN on the held-out test split.

Reads `splits.json` (written by train.py) so the test set is identical to the
test set the ViT will see — preventing accidental cross-stage leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
)

from classifier import ALSTriStreamClassifier  # noqa: E402
from dataset import MultiModalALSDataset       # noqa: E402
from paths import ARTIFACTS_DIR, CHECKPOINT_PATH, DATA_DIR, METRICS_DIR, ensure_output_dirs  # noqa: E402
from splits import indices_from_split, read_splits  # noqa: E402


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
    *,
    data_dir: Path,
    ckpt_path: Path,
    splits_path: Path,
    metrics_dir: Path,
    device_name: str,
    batch_size: int,
    target_shape: tuple[int, int, int],
) -> None:
    ensure_output_dirs()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device_name)
    print(f"--- CNN evaluation on {device} ---")

    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        return
    if not splits_path.exists():
        print(f"Error: splits.json not found: {splits_path}. Run train.py first.")
        return

    full = MultiModalALSDataset(rootDirectory=data_dir, transform=False, targetShape=target_shape)
    splits = read_splits(splits_path)
    test_idx = indices_from_split(full.to_sample_meta(), splits, "test")
    if not test_idx:
        print("Test split is empty in splits.json.")
        return

    test_set = Subset(full, test_idx)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ALSTriStreamClassifier().to(device)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    model.load_state_dict(state, strict=False)
    model.eval()

    labels: list[float] = []
    probs: list[float] = []
    with torch.no_grad():
        for inputs, y in loader:
            t1, t2, flair = (v.to(device) for v in inputs)
            logits = model(t1, t2, flair).squeeze(1)
            p = torch.sigmoid(logits.float()).cpu().numpy().reshape(-1).tolist()
            probs.extend(p)
            labels.extend(y.cpu().numpy().reshape(-1).tolist())

    preds = [1 if p >= 0.5 else 0 for p in probs]
    int_labels = [int(l) for l in labels]

    metrics = {
        "accuracy": float(accuracy_score(int_labels, preds)),
        "precision": float(precision_score(int_labels, preds, zero_division=0)),
        "recall": float(recall_score(int_labels, preds, zero_division=0)),
        "f1_score": float(f1_score(int_labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(int_labels, probs)) if len(set(int_labels)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(int_labels, probs)) if len(set(int_labels)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(int_labels, preds).tolist(),
        "num_test_samples": len(test_set),
    }

    out = metrics_dir / "evaluation_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the tri-stream CNN.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    p.add_argument("--splits-path", type=Path, default=ARTIFACTS_DIR / "splits.json")
    p.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--target-shape", type=int, nargs=3, default=[128, 128, 128])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        data_dir=args.data_dir,
        ckpt_path=args.checkpoint,
        splits_path=args.splits_path,
        metrics_dir=args.metrics_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        target_shape=tuple(args.target_shape),
    )
