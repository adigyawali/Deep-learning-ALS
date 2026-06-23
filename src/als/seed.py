"""Reproducibility helpers shared by every training/eval entry point."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42, *, cudnn_benchmark: bool = True) -> None:
    """Seed Python / NumPy / Torch (CPU + all CUDA devices).

    ``cudnn_benchmark=True`` is the right default for fixed-shape 3D volumes
    (it lets cuDNN pick the fastest conv algorithm once); set it False if you
    need bit-exact determinism instead of speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = cudnn_benchmark


def resolve_device(prefer: str = "auto") -> torch.device:
    """Resolve a device string ('auto'|'cuda'|'mps'|'cpu') to a torch.device.

    'auto' prefers CUDA, then Apple MPS, then CPU. An explicit choice that is
    unavailable silently falls back through the same order, so the same command
    works on the lab GPU box and on a laptop.
    """
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
