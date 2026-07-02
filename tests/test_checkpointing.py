"""Best-weights save / load round-trip (no resume, no per-epoch checkpoint)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from als.training.checkpointing import load_best_weights, save_best_weights


def _model():
    return nn.Linear(4, 1)


def test_only_best_written(tmp_path: Path):
    m = _model()
    save_best_weights(tmp_path, "vit", model=m, best_metric=0.7,
                      best_metric_name="roc_auc", threshold=0.55)
    assert (tmp_path / "vit_best.pt").exists()
    # No per-epoch / latest / resume file is ever produced.
    assert not (tmp_path / "vit_latest.pt").exists()


def test_best_file_holds_only_inference_essentials(tmp_path: Path):
    m = _model()
    save_best_weights(tmp_path, "nnmamba", model=m, best_metric=0.83,
                      best_metric_name="roc_auc", threshold=0.6, config={"a": 1})
    blob = load_best_weights(tmp_path / "nnmamba_best.pt")
    assert set(blob) == {"model_state_dict", "best_metric", "best_metric_name",
                         "threshold", "config"}
    # Deliberately absent: optimizer / scheduler / scaler / rng / epoch state.
    for forbidden in ("optimizer_state_dict", "scheduler_state_dict",
                      "scaler_state_dict", "rng", "epoch"):
        assert forbidden not in blob
    assert abs(blob["best_metric"] - 0.83) < 1e-9
    assert abs(blob["threshold"] - 0.6) < 1e-9


def test_weights_reload_matches(tmp_path: Path):
    m = _model()
    save_best_weights(tmp_path, "vit", model=m, best_metric=0.9,
                      best_metric_name="roc_auc")
    m2 = _model()
    m2.load_state_dict(load_best_weights(tmp_path / "vit_best.pt")["model_state_dict"])
    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)
