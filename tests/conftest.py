"""Shared pytest setup.

The package is normally installed editable (``pip install -e .``); we also add
``src/`` to ``sys.path`` so the suite runs straight from a checkout. Tests must
never hit the network (no MedicalNet download): TORCH_HOME points at a local
cache and ALS_SKIP_PRETRAINED keeps any backbone build offline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("TORCH_HOME", str(ROOT / ".pytest_cache" / "torch_home"))
os.environ.setdefault("ALS_SKIP_PRETRAINED", "1")
