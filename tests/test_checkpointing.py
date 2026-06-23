"""Checkpoint save / best / resume round-trip."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from als.training.checkpointing import maybe_resume, save_checkpoint


def _model_opt():
    m = nn.Linear(4, 1)
    return m, torch.optim.AdamW(m.parameters(), lr=1e-3)


def test_latest_and_best_written(tmp_path: Path):
    m, opt = _model_opt()
    save_checkpoint(tmp_path, "vit", model=m, optimizer=opt, epoch=3, best_metric=0.7,
                    best_metric_name="roc_auc", threshold=0.55, is_best=True)
    assert (tmp_path / "vit_latest.pt").exists()
    assert (tmp_path / "vit_best.pt").exists()


def test_resume_restores_epoch_metric_and_weights(tmp_path: Path):
    m, opt = _model_opt()
    # take a step so optimizer state is non-trivial
    out = m(torch.randn(2, 4)).sum(); out.backward(); opt.step()
    save_checkpoint(tmp_path, "nnmamba", model=m, optimizer=opt, epoch=5, best_metric=0.83,
                    best_metric_name="roc_auc", is_best=True)

    m2, opt2 = _model_opt()
    start_epoch, best = maybe_resume(tmp_path, "nnmamba", model=m2, optimizer=opt2,
                                     device="cpu", enabled=True)
    assert start_epoch == 5
    assert abs(best - 0.83) < 1e-9
    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


def test_fresh_when_no_checkpoint(tmp_path: Path):
    m, opt = _model_opt()
    start_epoch, best = maybe_resume(tmp_path, "vit", model=m, optimizer=opt, enabled=True)
    assert start_epoch == 0 and best == -float("inf")


def test_resume_disabled_starts_fresh(tmp_path: Path):
    m, opt = _model_opt()
    save_checkpoint(tmp_path, "vit", model=m, optimizer=opt, epoch=9, best_metric=0.9,
                    best_metric_name="roc_auc", is_best=True)
    start_epoch, best = maybe_resume(tmp_path, "vit", model=m, optimizer=opt, enabled=False)
    assert start_epoch == 0 and best == -float("inf")
