"""
Train the spatial multi-modal ViT on CNN spatial features.

Improvements vs. earlier version:
  * Uses the canonical splits.json from src/splits.py — no split drift.
  * AMP bf16 on CUDA (fp32 on macOS / CPU).
  * Tracks val ROC-AUC + PR-AUC. Best checkpoint selection metric is configurable.
  * Saves an operating-point threshold computed on val (Youden's J) into the ckpt.
  * Full RNG / optimizer / scheduler state in the checkpoint for clean resume.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Subset

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))   # adds .../src/
sys.path.insert(0, str(_THIS.parent))       # adds .../src/ViTModel/

from dataset import (                                # noqa: E402
    ALSSpatialFeatureDataset,
    compute_pos_weight,
    indices_from,
    load_or_build_splits,
)
from model import SpatialMultiModalViT              # noqa: E402
from paths import build_runtime_paths, ensure_output_dirs  # noqa: E402

# ── Defaults ──────────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
EPOCHS = 60
WARMUP_EPOCHS = 5
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
EARLY_STOP_PATIENCE = 15

EMBED_DIM = 384
DEPTH = 6
NUM_HEADS = 6
MLP_RATIO = 4.0
DROPOUT = 0.15
ATTN_DROPOUT = 0.1
MODALITY_DROPOUT_PROB = 0.25


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _safe_auc(labels, probs) -> float:
    if len(set(int(x) for x in labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def _safe_pr_auc(labels, probs) -> float:
    if len(set(int(x) for x in labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, probs))


def _make_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _youden_threshold(labels, probs) -> float:
    """Threshold maximizing sensitivity + specificity − 1 on the val set."""
    if len(set(int(x) for x in labels)) < 2:
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


def train(
    features_dir: str | None = None,
    artifacts_dir: str | None = None,
    splits_path: str | None = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2,
    epochs: int = EPOCHS,
    device_name: str = "auto",
    best_metric: str = "roc_auc",   # roc_auc | pr_auc
) -> None:
    ensure_output_dirs()
    _set_seed(SEED)

    runtime_features_dir, runtime_artifacts_dir, runtime_checkpoint_dir, runtime_metrics_dir, runtime_checkpoint_path = (
        build_runtime_paths(features_dir=features_dir, artifacts_dir=artifacts_dir)
    )
    runtime_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runtime_metrics_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(device_name)
    print(f"Device: {device}")
    print(f"Features dir : {runtime_features_dir}")
    print(f"Artifacts dir: {runtime_artifacts_dir}")

    dataset = ALSSpatialFeatureDataset(features_dir=str(runtime_features_dir))
    if len(dataset) < 3:
        print(f"Error: need at least 3 *_spatial.pt files in {runtime_features_dir}.")
        print("Run cnnModelMultiModality/generate_spatial_features.py first.")
        return

    # Shared splits.json — same one the CNN trainer wrote.
    chosen_splits_path = Path(splits_path) if splits_path else (runtime_artifacts_dir.parent / "cnn_multimodal" / "splits.json")
    if not chosen_splits_path.exists():
        # Fall back to building a fresh split here; downstream stages will still see the same file.
        chosen_splits_path = runtime_artifacts_dir / "splits.json"
    splits = load_or_build_splits(
        dataset.samples,
        chosen_splits_path,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )
    print(f"Using splits : {chosen_splits_path}")
    print(f"  class counts: {splits['class_counts']}")

    train_idx = indices_from(dataset.samples, splits, "train")
    val_idx = indices_from(dataset.samples, splits, "val")
    if not train_idx or not val_idx:
        print("Error: empty train or val split.")
        return

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    model = SpatialMultiModalViT(
        in_channels=dataset.in_channels,
        spatial_shape=dataset.spatial_shape,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        attn_dropout=ATTN_DROPOUT,
        modality_dropout_prob=MODALITY_DROPOUT_PROB,
    ).to(device)

    pos_weight = compute_pos_weight(dataset.samples, train_idx).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = _make_scheduler(optimizer, total_epochs=epochs, warmup_epochs=WARMUP_EPOCHS)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViT params   : {n_params:,}  tokens(incl CLS)={model.num_tokens}")
    print(f"pos_weight   : {pos_weight.item():.3f}")
    print(f"Best metric  : {best_metric}")
    if amp_dtype is not None:
        print(f"AMP enabled  : {amp_dtype}")

    best_val_metric = -float("inf")
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(epochs):
        t0 = time.time()
        # ─── train ───────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_labels: list[float] = []
        train_probs: list[float] = []
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else torch.autocast(device_type="cpu", enabled=False)
            with ctx:
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += float(loss.detach()) * x.size(0)
            with torch.no_grad():
                p = torch.sigmoid(logits.float()).cpu().numpy().reshape(-1)
                train_probs.extend(p.tolist())
                train_labels.extend(y.detach().cpu().numpy().reshape(-1).tolist())

        train_loss = train_loss_sum / max(1, len(train_set))
        train_auc = _safe_auc(train_labels, train_probs)
        train_pr = _safe_pr_auc(train_labels, train_probs)

        # ─── validate ────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_labels: list[float] = []
        val_probs: list[float] = []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).unsqueeze(1)
                ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else torch.autocast(device_type="cpu", enabled=False)
                with ctx:
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss_sum += float(loss) * x.size(0)
                p = torch.sigmoid(logits.float()).cpu().numpy().reshape(-1)
                val_probs.extend(p.tolist())
                val_labels.extend(y.cpu().numpy().reshape(-1).tolist())

        val_loss = val_loss_sum / max(1, len(val_set))
        val_auc = _safe_auc(val_labels, val_probs)
        val_pr = _safe_pr_auc(val_labels, val_probs)
        threshold = _youden_threshold(val_labels, val_probs)
        val_preds = [1 if p >= threshold else 0 for p in val_probs]
        val_acc = sum(1 for p, l in zip(val_preds, val_labels) if int(p) == int(l)) / max(1, len(val_labels))

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        select_value = val_pr if best_metric == "pr_auc" else val_auc
        improved = (not np.isnan(select_value)) and select_value > best_val_metric
        tag = ""
        if improved:
            best_val_metric = select_value
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_metric_name": best_metric,
                    "best_val_metric": best_val_metric,
                    "best_val_threshold": threshold,
                    "best_val_auc": val_auc,
                    "best_val_pr_auc": val_pr,
                    "config": {
                        "in_channels": dataset.in_channels,
                        "spatial_shape": list(dataset.spatial_shape),
                        "embed_dim": EMBED_DIM,
                        "depth": DEPTH,
                        "num_heads": NUM_HEADS,
                        "mlp_ratio": MLP_RATIO,
                        "dropout": DROPOUT,
                        "attn_dropout": ATTN_DROPOUT,
                        "modality_dropout_prob": MODALITY_DROPOUT_PROB,
                    },
                    "rng": {
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "numpy": np.random.get_state(),
                        "python": random.getstate(),
                    },
                    "pos_weight": float(pos_weight.item()),
                    "splits_path": str(chosen_splits_path),
                },
                runtime_checkpoint_path,
            )
            tag = " *best"
        else:
            epochs_no_improve += 1

        record = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_auc": train_auc,
            "train_pr_auc": train_pr,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_pr_auc": val_pr,
            "val_accuracy_at_threshold": val_acc,
            "val_threshold": threshold,
            "seconds": round(time.time() - t0, 2),
        }
        history.append(record)
        print(
            f"E{epoch + 1:02d}/{epochs} lr={current_lr:.2e} "
            f"tr_loss={train_loss:.4f} tr_auc={train_auc:.3f} "
            f"tr_pr={train_pr:.3f} | "
            f"va_loss={val_loss:.4f} va_auc={val_auc:.3f} "
            f"va_pr={val_pr:.3f} thr={threshold:.2f} "
            f"acc={val_acc:.3f} ({record['seconds']:.1f}s){tag}"
        )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stop @ epoch {epoch + 1}: no {best_metric} improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    history_path = runtime_metrics_dir / "vit_train_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Best val {best_metric}: {best_val_metric:.4f}")
    print(f"Checkpoint    : {runtime_checkpoint_path}")
    print(f"History       : {history_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train spatial multi-modal ViT.")
    p.add_argument("--features-dir", type=str, default=None)
    p.add_argument("--artifacts-dir", type=str, default=None)
    p.add_argument("--splits-path", type=str, default=None,
                   help="Path to splits.json shared with the CNN stage.")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--best-metric", type=str, default="roc_auc", choices=["roc_auc", "pr_auc"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
        splits_path=args.splits_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        device_name=args.device,
        best_metric=args.best_metric,
    )
