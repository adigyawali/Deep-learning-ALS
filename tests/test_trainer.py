"""Trainer integration: loop runs, writes best weights and history (no resume)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from als.training import trainer


def _loaders(n=12, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    # learnable signal: label depends on first feature
    y = (x[:, 0] > 0).float()
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4), DataLoader(ds, batch_size=4)


def _forward(model, batch, device):
    x, y = batch
    return model(x.to(device)), y.to(device).unsqueeze(1)


def test_fit_writes_best_and_history(tmp_path: Path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    tr, va = _loaders()
    out = trainer.fit(
        model=model, train_loader=tr, val_loader=va, forward_fn=_forward,
        criterion=nn.BCEWithLogitsLoss(), optimizer=opt, scheduler=None,
        device=torch.device("cpu"), epochs=3, ckpt_dir=tmp_path, ckpt_prefix="t",
        amp_dtype=None, grad_accum_steps=2, best_metric_name="roc_auc",
        early_stop_patience=99, history_path=tmp_path / "hist.json",
    )
    assert (tmp_path / "t_best.pt").exists()
    # No per-epoch / latest / resume checkpoint is written.
    assert not (tmp_path / "t_latest.pt").exists()
    assert (tmp_path / "hist.json").exists()
    assert len(out["history"]) == 3
    assert out["best_metric_name"] == "roc_auc"


def test_best_file_is_inference_only(tmp_path: Path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    tr, va = _loaders()
    trainer.fit(model=model, train_loader=tr, val_loader=va, forward_fn=_forward,
                criterion=nn.BCEWithLogitsLoss(), optimizer=opt, scheduler=None,
                device=torch.device("cpu"), epochs=2, ckpt_dir=tmp_path, ckpt_prefix="t",
                amp_dtype=None, early_stop_patience=99)
    blob = torch.load(tmp_path / "t_best.pt", weights_only=False)
    for forbidden in ("optimizer_state_dict", "scheduler_state_dict",
                      "scaler_state_dict", "rng", "epoch"):
        assert forbidden not in blob
