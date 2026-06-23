"""Helpers shared across stages: dataloaders and per-model forward adapters."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def make_loader(dataset, *, batch_size: int, shuffle: bool, dl_cfg: dict, device: torch.device) -> DataLoader:
    num_workers = int(dl_cfg.get("num_workers", 2))
    kwargs: dict = dict(
        batch_size=max(1, batch_size),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(dl_cfg.get("pin_memory", True)) and device.type == "cuda",
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(dl_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(dl_cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def smoke_trim(indices: list[int], cfg: dict) -> list[int]:
    """Cap an index list to the smoke subset size when in smoke mode."""
    if cfg.get("smoke"):
        return indices[: int(cfg.get("smoke_max_samples", 6))]
    return indices


# ── forward adapters: (model, batch, device) -> (logits (B,1), labels (B,1)) ──

def cnn_forward(model, batch, device):
    (t1, t2, flair), y = batch
    t1 = t1.to(device, non_blocking=True)
    t2 = t2.to(device, non_blocking=True)
    flair = flair.to(device, non_blocking=True)
    logits = model(t1, t2, flair)
    return logits, y.to(device, non_blocking=True).unsqueeze(1)


def vit_forward(model, batch, device):
    x, y, _ = batch
    x = x.to(device, non_blocking=True)
    logits = model(x)
    return logits, y.to(device, non_blocking=True).unsqueeze(1)


def volume_forward(model, batch, device):
    x, y = batch
    x = x.to(device, non_blocking=True)
    logits = model(x)
    return logits, y.to(device, non_blocking=True).unsqueeze(1)
