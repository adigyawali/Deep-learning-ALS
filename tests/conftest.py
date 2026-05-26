"""
Shared pytest fixtures and path setup.

Adds `src/` to sys.path so tests can import the project modules without an
editable install. Keeps tests independent of how the project is packaged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# `src/` for the canonical modules (e.g. splits.py).
sys.path.insert(0, str(SRC))
# Per-stage paths are injected by individual test modules to avoid `dataset`
# name collisions between cnnModelMultiModality and ViTModel.

# Tests must never reach the public internet (MedicalNet weight download).
os.environ.setdefault("TORCH_HOME", str(ROOT / ".pytest_cache" / "torch_home"))
