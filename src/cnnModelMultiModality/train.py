"""
CLI trainer for the tri-stream CNN (`ALSTriStreamClassifier`).

Reproducible end-to-end training:
  * subject + label stratified split via `src/splits.py` (writes splits.json once)
  * AMP bf16 on CUDA (fp32 elsewhere)
  * AdamW with layerwise LR decay (lower LR for backbone, higher for head)
  * Linear warmup + cosine decay
  * BCEWithLogits with pos_weight = N_neg/N_pos from the training split
  * AUC-best checkpoint with optimizer/scheduler/RNG saved for full resume

Run:
    python -m src.cnnModelMultiModality.train --epochs 60 --batch-size 4
or, from inside the module:
    python train.py --epochs 60 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))         # adds .../src/
sys.path.insert(0, str(_THIS.parent))             # adds .../src/cnnModelMultiModality/

from classifier import ALSTriStreamClassifier  # noqa: E402
from dataset import MultiModalALSDataset        # noqa: E402
from paths import (                              # noqa: E402
    ARTIFACTS_DIR,
    CHECKPOINT_DIR,
    CHECKPOINT_PATH,
    DATA_DIR,
    METRICS_DIR,
    ensure_output_dirs,
)
from splits import (                             # noqa: E402
    indices_from_split,
    make_subject_splits,
    read_splits,
    write_splits,
)

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    roc_auc_score = None
    average_precision_score = None


# ── Defaults ──────────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 4
EPOCHS = 60
WARMUP_EPOCHS = 5
TARGET_SHAPE = (128, 128, 128)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
EARLY_STOP_PATIENCE = 15

LR_BACKBONE = 1e-5     # MedicalNet-pretrained; small LR for stability
LR_HEAD = 1e-3         # projection + transformer + classifier; from scratch
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # faster on fixed-shape inputs


def _resolve_device(prefer: str) -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_splits(samples_meta, splits_path: Path, seed: int) -> dict:
    """Read splits.json if present; otherwise build it from `samples_meta`."""
    if splits_path.exists():
        splits = read_splits(splits_path)
        print(f"  Using existing splits: {splits_path}")
    else:
        splits = make_subject_splits(
            samples_meta,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            seed=seed,
            stratify_by_site=True,
        )
        write_splits(splits_path, splits)
        print(f"  Wrote new splits to: {splits_path}")
    return splits


def _build_optimizer(model: nn.Module, lr_backbone: float, lr_head: float, wd: float) -> torch.optim.Optimizer:
    """Two param groups: backbone (lower LR) and everything else (higher LR)."""
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "_backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)
    print(f"  optimizer: {len(backbone_params)} backbone params, {len(head_params)} head params")
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": wd},
            {"params": head_params, "lr": lr_head, "weight_decay": wd},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def _build_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _safe_auc(labels, probs) -> float:
    if roc_auc_score is None or len(set(int(x) for x in labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def _epoch(loader, model, criterion, device, train: bool, optimizer=None, amp_dtype=None, grad_clip: float = 0.0):
    model.train(mode=train)
    total_loss = 0.0
    total_n = 0
    all_labels: list[float] = []
    all_probs: list[float] = []

    autocast_enabled = (amp_dtype is not None and device.type == "cuda")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype) if autocast_enabled
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    )

    for (inputs, labels) in loader:
        t1, t2, flair = (v.to(device, non_blocking=True) for v in inputs)
        y = labels.to(device, non_blocking=True).unsqueeze(1)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train), autocast_ctx:
            logits = model(t1, t2, flair)
            loss = criterion(logits, y)

        if train:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += float(loss.detach()) * y.size(0)
        total_n += y.size(0)
        with torch.no_grad():
            probs = torch.sigmoid(logits.float()).cpu().numpy().reshape(-1)
        all_probs.extend(probs.tolist())
        all_labels.extend(y.detach().cpu().numpy().reshape(-1).tolist())

    return {
        "loss": total_loss / max(1, total_n),
        "auc": _safe_auc(all_labels, all_probs),
        "labels": all_labels,
        "probs": all_probs,
    }


def train(
    *,
    data_dir: Path,
    artifacts_dir: Path,
    splits_path: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device_name: str,
    freeze_backbone: bool,
    target_shape: tuple[int, int, int],
) -> None:
    ensure_output_dirs()
    _set_seed(SEED)
    device = _resolve_device(device_name)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics").mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")
    print(f"Artifacts dir: {artifacts_dir}")

    # 1) Datasets — same wrapper used by everything; transform=True only for train.
    full = MultiModalALSDataset(rootDirectory=data_dir, transform=False, targetShape=target_shape)
    if len(full) < 3:
        print(f"Error: fewer than 3 samples found in {data_dir}.")
        return
    train_aug = MultiModalALSDataset(rootDirectory=data_dir, transform=True, targetShape=target_shape)

    # 2) Splits — write splits.json once, reuse everywhere.
    splits = _ensure_splits(full.to_sample_meta(), splits_path, SEED)
    meta = full.to_sample_meta()
    train_idx = indices_from_split(meta, splits, "train")
    val_idx = indices_from_split(meta, splits, "val")
    test_idx = indices_from_split(meta, splits, "test")
    print(f"Splits: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print(f"  class counts: {splits['class_counts']}")

    if len(train_idx) == 0 or len(val_idx) == 0:
        print("Error: train or val split is empty. Add more data and rebuild splits.json.")
        return

    train_set = Subset(train_aug, train_idx)
    val_set = Subset(full, val_idx)

    pin_memory = device.type == "cuda"
    persistent = num_workers > 0
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=persistent, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=max(1, batch_size),
                            shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, persistent_workers=persistent)

    # 3) Model.
    model = ALSTriStreamClassifier(freeze_backbone=freeze_backbone).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: total={n_total:,}  trainable={n_trainable:,}")

    # 4) Loss + optimizer + scheduler.
    n_pos = sum(1 for i in train_idx if meta[i].label == 1.0)
    n_neg = sum(1 for i in train_idx if meta[i].label == 0.0)
    pos_weight_val = max(0.1, min(10.0, n_neg / max(1, n_pos))) if n_pos > 0 else 1.0
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=device)
    print(f"pos_weight = {pos_weight_val:.3f}  (train: {n_neg} neg, {n_pos} pos)")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = _build_optimizer(model, LR_BACKBONE, LR_HEAD, WEIGHT_DECAY)
    scheduler = _build_scheduler(optimizer, total_epochs=epochs, warmup_epochs=WARMUP_EPOCHS)

    # 5) AMP dtype.
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    if amp_dtype is not None:
        print(f"AMP enabled: dtype={amp_dtype}")

    history: list[dict] = []
    best_val_auc = -float("inf")
    epochs_no_improve = 0
    ckpt_path = artifacts_dir / "checkpoints" / "encoder_weights.pth"

    for epoch in range(epochs):
        t0 = time.time()
        train_metrics = _epoch(
            train_loader, model, criterion, device,
            train=True, optimizer=optimizer, amp_dtype=amp_dtype, grad_clip=GRAD_CLIP,
        )
        val_metrics = _epoch(
            val_loader, model, criterion, device, train=False, amp_dtype=amp_dtype,
        )
        scheduler.step()
        dt = time.time() - t0

        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        record = {
            "epoch": epoch + 1,
            "lr_backbone": current_lrs[0],
            "lr_head": current_lrs[-1],
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "seconds": round(dt, 2),
        }
        history.append(record)

        improved = (not np.isnan(val_metrics["auc"])) and val_metrics["auc"] > best_val_auc
        tag = ""
        if improved:
            best_val_auc = val_metrics["auc"]
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_auc": best_val_auc,
                    "pos_weight": float(pos_weight_val),
                    "config": {
                        "freeze_backbone": freeze_backbone,
                        "target_shape": list(target_shape),
                        "batch_size": batch_size,
                        "lr_backbone": LR_BACKBONE,
                        "lr_head": LR_HEAD,
                        "weight_decay": WEIGHT_DECAY,
                        "warmup_epochs": WARMUP_EPOCHS,
                        "epochs": epochs,
                    },
                    "rng": {
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "numpy": np.random.get_state(),
                        "python": random.getstate(),
                    },
                    "splits_path": str(splits_path),
                },
                ckpt_path,
            )
            tag = " *best"
        else:
            epochs_no_improve += 1

        print(
            f"E{epoch + 1:02d}/{epochs} "
            f"lr=[{current_lrs[0]:.2e}/{current_lrs[-1]:.2e}] "
            f"tr_loss={train_metrics['loss']:.4f} tr_auc={train_metrics['auc']:.3f} | "
            f"va_loss={val_metrics['loss']:.4f} va_auc={val_metrics['auc']:.3f} "
            f"({dt:.1f}s){tag}"
        )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stop @ epoch {epoch + 1}: no val AUC improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    hist_path = artifacts_dir / "metrics" / "cnn_train_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"Training done. best_val_auc={best_val_auc:.4f}")
    print(f"Best checkpoint: {ckpt_path}")
    print(f"History       : {hist_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the tri-stream CNN.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Processed-data root.")
    p.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR, help="Artifacts root.")
    p.add_argument("--splits-path", type=Path, default=ARTIFACTS_DIR / "splits.json", help="splits.json location.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Keep MedicalNet ResNet50 frozen (default fine-tunes end-to-end).")
    p.add_argument("--target-shape", type=int, nargs=3, default=list(TARGET_SHAPE),
                   help="Spatial size for resampled volumes (D H W). Default 128 128 128.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        splits_path=args.splits_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        freeze_backbone=args.freeze_backbone,
        target_shape=tuple(args.target_shape),
    )
