"""
Tests for the statistical helpers in evaluate_vit.py — DeLong CI, bootstrap
CI, ECE — without needing a trained model.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC / "ViTModel"))
_spec = importlib.util.spec_from_file_location("vit_eval_module", _SRC / "ViTModel" / "evaluate_vit.py")
_vit_eval = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["vit_eval_module"] = _vit_eval
_spec.loader.exec_module(_vit_eval)
_bootstrap_auc_ci = _vit_eval._bootstrap_auc_ci
_delong_ci = _vit_eval._delong_ci
_expected_calibration_error = _vit_eval._expected_calibration_error
_sens_spec = _vit_eval._sens_spec


def test_bootstrap_ci_brackets_auc() -> None:
    rng = np.random.default_rng(0)
    # Perfectly separable case: AUC = 1, narrow CI around 1.
    labels = [0] * 30 + [1] * 30
    probs = list(rng.uniform(0.0, 0.4, size=30)) + list(rng.uniform(0.6, 1.0, size=30))
    lo, hi = _bootstrap_auc_ci(labels, probs, n_boot=400, seed=0)
    assert 0.8 < lo <= hi <= 1.0


def test_delong_ci_is_finite_for_well_separated_classes() -> None:
    rng = np.random.default_rng(0)
    labels = [0] * 50 + [1] * 50
    probs = list(rng.normal(0.2, 0.1, 50)) + list(rng.normal(0.8, 0.1, 50))
    lo, hi = _delong_ci(labels, probs)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert 0.0 <= lo <= hi <= 1.0


def test_delong_ci_handles_single_class() -> None:
    lo, hi = _delong_ci([0, 0, 0], [0.1, 0.2, 0.3])
    assert math.isnan(lo) and math.isnan(hi)


def test_ece_zero_for_perfect_calibration() -> None:
    # Perfectly calibrated 50/50 binary problem.
    labels = [0, 1] * 50
    probs = [0.0, 1.0] * 50
    assert _expected_calibration_error(labels, probs, n_bins=10) < 1e-6


def test_sens_spec_basic() -> None:
    # 10 positive predicted correctly (sens=1), 8 negatives correct, 2 false positives → spec=0.8
    labels = [1] * 10 + [0] * 10
    preds = [1] * 10 + [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    sens, spec = _sens_spec(labels, preds)
    assert math.isclose(sens, 1.0)
    assert math.isclose(spec, 0.8)
