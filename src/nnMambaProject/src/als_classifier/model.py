"""Wrapper around the upstream nnMamba classification model.

Imports from external/nnMamba/nnMamba4cls.py. That file requires `mamba-ssm`,
which is CUDA-only. This module will raise a clear error on macOS / CPU-only
machines; the rest of the pipeline (dataset, split) still works locally.
"""
from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2] / "external" / "nnMamba"
if REPO.exists() and str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


try:
    from nnMamba4cls import nnMambaEncoder  # type: ignore
except ImportError as e:
    raise ImportError(
        "Could not import nnMambaEncoder from external/nnMamba/nnMamba4cls.py. "
        "This requires:\n"
        "  1. external/nnMamba/ exists (run: git clone "
        "https://github.com/lhaof/nnMamba.git external/nnMamba)\n"
        "  2. mamba-ssm and causal-conv1d are installed (CUDA-only — lab machine).\n"
        "If you're on macOS or a CPU-only machine, this is expected; only the "
        "training/inference paths need the GPU build."
    ) from e


def build_model(
    in_channels: int = 3,
    num_classes: int = 2,
    channels: int = 32,
    blocks: int = 3,
):
    """Build nnMambaEncoder configured for ALS T1+T2+FLAIR classification.

    Translates conventional kwarg names (in_channels, num_classes) to the
    upstream names (in_ch, number_classes).
    """
    return nnMambaEncoder(
        in_ch=in_channels,
        channels=channels,
        blocks=blocks,
        number_classes=num_classes,
    )
