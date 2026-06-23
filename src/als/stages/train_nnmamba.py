"""Stage: train the end-to-end CNN→nnMamba model on raw volumes."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Subset

from .. import sanity
from ..config import get
from ..data.volume_dataset import VolumeDataset
from ..models.cnn_nnmamba import CNNnnMamba
from ..models.components.mamba_block import MAMBA_BACKEND
from ..paths import DEFAULT_DATA_DIR, RunPaths
from ..splits import indices_from_split, load_or_build_splits
from ..training import trainer
from ..training.optim import amp_dtype_from_str, pos_weight_from_labels, warmup_cosine_scheduler
from ._common import make_loader, smoke_trim, volume_forward


def run(cfg: dict, paths: RunPaths, device: torch.device, *, resume: bool = False) -> None:
    data_dir = get(cfg, "data", "data_dir") or DEFAULT_DATA_DIR
    target_shape = tuple(get(cfg, "data", "target_shape", default=[128, 128, 128]))
    use_frequency = bool(get(cfg, "data", "use_frequency", default=True))
    aug_level = get(cfg, "data", "aug_level", default="medium")
    m = cfg["nnmamba"]
    print(f"[nnmamba] Mamba backend: {MAMBA_BACKEND}  use_frequency={use_frequency}")

    full = VolumeDataset(data_dir, return_mode="stack", target_shape=target_shape,
                         transform=False, use_frequency=use_frequency)
    if len(full) < 3:
        print(f"[nnmamba] Error: fewer than 3 samples in {data_dir}.")
        return
    train_aug = VolumeDataset(data_dir, return_mode="stack", target_shape=target_shape,
                              transform=True, use_frequency=use_frequency, aug_level=aug_level)

    splits = load_or_build_splits(
        full.to_sample_meta(), paths.splits_path,
        train_ratio=get(cfg, "split", "train_ratio", default=0.8),
        val_ratio=get(cfg, "split", "val_ratio", default=0.1),
        seed=cfg.get("seed", 42),
    )
    meta = full.to_sample_meta()
    train_idx = smoke_trim(indices_from_split(meta, splits, "train"), cfg)
    val_idx = smoke_trim(indices_from_split(meta, splits, "val"), cfg)
    if not train_idx or not val_idx:
        print("[nnmamba] Error: empty train or val split.")
        return

    dl = cfg.get("dataloader", {})
    train_loader = make_loader(Subset(train_aug, train_idx), batch_size=m["batch_size"],
                               shuffle=True, dl_cfg=dl, device=device)
    val_loader = make_loader(Subset(full, val_idx), batch_size=m["batch_size"],
                             shuffle=False, dl_cfg=dl, device=device)

    model = CNNnnMamba(
        use_frequency=use_frequency, base=m.get("base", 32), blocks=m.get("blocks", 3),
        token_grid=m.get("token_grid", 4), mamba_layers=m.get("mamba_layers", 2),
        d_state=m.get("d_state", 16), dropout=m.get("dropout", 0.1),
    ).to(device)

    pw = pos_weight_from_labels([meta[i].label for i in train_idx])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=m["lr"], weight_decay=m["weight_decay"])
    scheduler = warmup_cosine_scheduler(optimizer, m["epochs"], m.get("warmup_epochs", 5))

    sanity.preflight(stage="train_nnmamba", model=model, dataset=full, splits=splits,
                     train_loader=train_loader, forward_fn=volume_forward, device=device,
                     ckpt_dir=paths.checkpoints, ckpt_prefix="nnmamba")

    trainer.fit(
        model=model, train_loader=train_loader, val_loader=val_loader,
        forward_fn=volume_forward, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, epochs=m["epochs"], ckpt_dir=paths.checkpoints, ckpt_prefix="nnmamba",
        config=cfg, splits_path=str(paths.splits_path),
        amp_dtype=amp_dtype_from_str(get(cfg, "train", "amp", default="bf16"), device),
        grad_accum_steps=m.get("grad_accum_steps", 1),
        clip_grad=get(cfg, "train", "clip_grad", default=1.0),
        best_metric_name=get(cfg, "train", "best_metric", default="roc_auc"),
        early_stop_patience=get(cfg, "train", "early_stop_patience", default=20),
        resume=resume, history_path=paths.metrics / "nnmamba_train_history.json",
    )
