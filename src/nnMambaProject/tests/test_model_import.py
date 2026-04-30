"""Document that model.py raises a clear error without mamba-ssm.

On Mac/CPU: the import fails, and the test confirms the error message is helpful.
On the lab GPU: mamba-ssm is installed, the import succeeds, and the test is skipped.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_model_import_fails_gracefully_without_mamba_ssm():
    try:
        import mamba_ssm  # noqa: F401
        pytest.skip("mamba_ssm is installed; this test only runs without it.")
    except ImportError:
        pass

    # Force a fresh import attempt
    if "als_classifier.model" in sys.modules:
        del sys.modules["als_classifier.model"]

    with pytest.raises(ImportError, match=r"mamba-ssm|nnMambaEncoder"):
        importlib.import_module("als_classifier.model")
