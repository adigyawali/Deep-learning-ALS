"""Optimizer / scheduler / AMP helpers shared by the training stages."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothBCEWithLogitsLoss(nn.Module):
    """BCE-with-logits + optional label smoothing, keeping ``pos_weight`` support.

    Label smoothing pulls the hard targets {0,1} toward 0.5 by ``smoothing``
    (target ← target·(1−s) + 0.5·s). On a small dataset this discourages the
    over-confident, memorised logits that drive the train→test overfitting gap,
    and improves probability calibration (lower ECE). ``smoothing=0`` reduces to
    plain ``BCEWithLogitsLoss`` so it is a safe default drop-in.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None, smoothing: float = 0.0):
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0,1), got {smoothing}")
        self.smoothing = float(smoothing)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0.0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


def amp_dtype_from_str(name: str, device: torch.device) -> torch.dtype | None:
    """Map a config string to an autocast dtype, or None to disable AMP.

    AMP is only ever enabled on CUDA; on MPS/CPU we return None so the trainer
    runs in fp32 (autocast support there is uneven and not worth the risk).
    """
    if device.type != "cuda":
        return None
    name = (name or "none").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    return None


def warmup_cosine_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    """Linear warmup for ``warmup_epochs`` then cosine decay to ~0."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pos_weight_from_labels(labels) -> float:
    """N_neg/N_pos clamped to [0.1, 10] for BCEWithLogitsLoss on imbalanced data."""
    labels = [float(x) for x in labels]
    n_pos = sum(1 for l in labels if l == 1.0)
    n_neg = sum(1 for l in labels if l == 0.0)
    if n_pos == 0:
        return 1.0
    return max(0.1, min(10.0, n_neg / max(1, n_pos)))
