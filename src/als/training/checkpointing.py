"""Checkpoint save / load / resume shared by every training stage.

Two files per stage live in ``runs/<model>/checkpoints/``:

  * ``<prefix>_latest.pt`` — written every epoch, used by ``--resume``.
  * ``<prefix>_best.pt``   — the best-validation snapshot, used for evaluation.

A checkpoint stores everything needed to continue *bit-for-bit* after a crash:
model + optimizer + scheduler + AMP scaler state, epoch, best metric (and its
name), the val-tuned threshold, the resolved config, the splits path, and full
RNG state. ``maybe_resume`` reads ``latest`` and restores all of it, so a killed
run continues from the next epoch instead of restarting from zero.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def _rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(rng: dict) -> None:
    try:
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    except (KeyError, RuntimeError, TypeError):
        # A best-effort restore; never let RNG bookkeeping abort a resume.
        pass


def save_checkpoint(
    ckpt_dir: Path | str,
    prefix: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    scaler: Any = None,
    epoch: int,
    best_metric: float,
    best_metric_name: str,
    threshold: float = 0.5,
    config: Optional[dict] = None,
    splits_path: Optional[str] = None,
    extra: Optional[dict] = None,
    is_best: bool = False,
) -> Path:
    """Write ``<prefix>_latest.pt``; copy it to ``<prefix>_best.pt`` when ``is_best``."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "best_metric_name": best_metric_name,
        "threshold": float(threshold),
        "config": config or {},
        "splits_path": splits_path,
        "rng": _rng_state(),
    }
    if extra:
        state.update(extra)

    latest = ckpt_dir / f"{prefix}_latest.pt"
    tmp = latest.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(latest)

    if is_best:
        best = ckpt_dir / f"{prefix}_best.pt"
        tmp_b = best.with_suffix(".pt.tmp")
        torch.save(state, tmp_b)
        tmp_b.replace(best)
    return latest


def load_checkpoint(path: Path | str, map_location="cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def maybe_resume(
    ckpt_dir: Path | str,
    prefix: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Any = None,
    device="cpu",
    enabled: bool = True,
) -> tuple[int, float]:
    """Restore from ``<prefix>_latest.pt`` if present and ``enabled``.

    Returns ``(start_epoch, best_metric)``. Prints a loud one-liner stating
    whether training is starting fresh or resuming, so it is unambiguous in the
    logs which one happened.
    """
    latest = Path(ckpt_dir) / f"{prefix}_latest.pt"
    if not enabled or not latest.exists():
        if enabled and not latest.exists():
            print(f"[resume] no checkpoint at {latest} — starting fresh from epoch 0.")
        else:
            print("[resume] disabled — starting fresh from epoch 0.")
        return 0, -float("inf")

    blob = load_checkpoint(latest, map_location=device)
    model.load_state_dict(blob["model_state_dict"])
    if optimizer is not None and blob.get("optimizer_state_dict"):
        optimizer.load_state_dict(blob["optimizer_state_dict"])
    if scheduler is not None and blob.get("scheduler_state_dict"):
        scheduler.load_state_dict(blob["scheduler_state_dict"])
    if scaler is not None and blob.get("scaler_state_dict"):
        scaler.load_state_dict(blob["scaler_state_dict"])
    _restore_rng(blob.get("rng", {}))

    start_epoch = int(blob.get("epoch", 0))
    best_metric = float(blob.get("best_metric", -float("inf")))
    name = blob.get("best_metric_name", "metric")
    print(f"[resume] RESUMING from {latest}: next epoch={start_epoch}, "
          f"best {name}={best_metric:.4f}.")
    return start_epoch, best_metric
