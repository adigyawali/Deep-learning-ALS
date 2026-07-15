"""Stage: train the end-to-end CNN→nnMamba model on raw volumes."""

from __future__ import annotations

import torch
from torch.utils.data import Subset

from .. import sanity
from ..config import get
from ..data.volume_dataset import VolumeDataset
from ..models.cnn_nnmamba import CNNnnMamba
from ..models.components.mamba_block import MAMBA_BACKEND
from ..paths import DEFAULT_DATA_DIR, RunPaths
from ..splits import indices_from_split, n_folds_in, resolve_splits
from ..training import trainer
from ..training.optim import (
    SmoothBCEWithLogitsLoss, amp_dtype_from_str, pos_weight_from_labels, warmup_cosine_scheduler,
)
from ._common import make_loader, smoke_trim, volume_forward


def run(cfg: dict, paths: RunPaths, device: torch.device) -> None:
    data_dir = get(cfg, "data", "data_dir") or DEFAULT_DATA_DIR
    target_shape = tuple(get(cfg, "data", "target_shape", default=[128, 128, 128]))
    use_frequency = bool(get(cfg, "data", "use_frequency", default=True))
    aug_level = get(cfg, "data", "aug_level", default="medium")
    aug_config = cfg.get("augmentations")   # from root config.yaml (source of truth)
    m = cfg["nnmamba"]
    spatial_encoder = m.get("spatial_encoder", "scratch")
    print(f"[nnmamba] Mamba backend: {MAMBA_BACKEND}  use_frequency={use_frequency}  "
          f"spatial_encoder={spatial_encoder}"
          + (f" (backbone={m.get('backbone', 'resnet18')}, "
             f"freeze={m.get('freeze_backbone', True)})" if spatial_encoder == "pretrained" else ""))

    full = VolumeDataset(data_dir, return_mode="stack", target_shape=target_shape,
                         transform=False, use_frequency=use_frequency)
    if len(full) < 3:
        print(f"[nnmamba] Error: fewer than 3 samples in {data_dir}.")
        return
    train_aug = VolumeDataset(data_dir, return_mode="stack", target_shape=target_shape,
                              transform=True, use_frequency=use_frequency,
                              aug_level=aug_level, aug_config=aug_config)

    splits = resolve_splits(
        full.to_sample_meta(), paths.splits_path,
        cv_cfg=cfg.get("cross_validation"), split_cfg=cfg.get("split"),
        seed=cfg.get("seed", 42),
    )
    meta = full.to_sample_meta()
    n_folds = n_folds_in(splits)
    dl = cfg.get("dataloader", {})

    # Train one independent nnMamba per CV fold, each into runs/cnn_nnmamba/fold{k}/.
    for fold in range(n_folds):
        fpaths = paths.fold(fold).ensure()
        train_idx = smoke_trim(indices_from_split(meta, splits, "train", fold), cfg)
        val_idx = smoke_trim(indices_from_split(meta, splits, "val", fold), cfg)
        if not train_idx or not val_idx:
            print(f"[nnmamba] fold {fold}: empty train or val split — skipping.")
            continue
        print(f"\n[nnmamba] ===== fold {fold + 1}/{n_folds} "
              f"(train={len(train_idx)} val={len(val_idx)}) =====")

        train_loader = make_loader(Subset(train_aug, train_idx), batch_size=m["batch_size"],
                                   shuffle=True, dl_cfg=dl, device=device)
        val_loader = make_loader(Subset(full, val_idx), batch_size=m["batch_size"],
                                 shuffle=False, dl_cfg=dl, device=device)

        model = CNNnnMamba(
            use_frequency=use_frequency, base=m.get("base", 32), blocks=m.get("blocks", 3),
            token_grid=m.get("token_grid", 4), mamba_layers=m.get("mamba_layers", 2),
            d_state=m.get("d_state", 16), dropout=m.get("dropout", 0.1),
            spatial_encoder=spatial_encoder, backbone=m.get("backbone", "resnet18"),
            freeze_backbone=m.get("freeze_backbone", True),
            pretrained_d_model=m.get("pretrained_d_model", 256),
        ).to(device)

        pw = pos_weight_from_labels([meta[i].label for i in train_idx])
        criterion = SmoothBCEWithLogitsLoss(
            pos_weight=torch.tensor(pw, dtype=torch.float32, device=device),
            smoothing=get(cfg, "train", "label_smoothing", default=0.0),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=m["lr"], weight_decay=m["weight_decay"])
        scheduler = warmup_cosine_scheduler(optimizer, m["epochs"], m.get("warmup_epochs", 5))

        sanity.preflight(stage=f"train_nnmamba[fold{fold}]", model=model, dataset=full, splits=splits,
                         train_loader=train_loader, forward_fn=volume_forward, device=device,
                         ckpt_dir=fpaths.checkpoints, ckpt_prefix="nnmamba")

        trainer.fit(
            model=model, train_loader=train_loader, val_loader=val_loader,
            forward_fn=volume_forward, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            device=device, epochs=m["epochs"], ckpt_dir=fpaths.checkpoints, ckpt_prefix="nnmamba",
            config=cfg,
            amp_dtype=amp_dtype_from_str(get(cfg, "train", "amp", default="bf16"), device),
            grad_accum_steps=m.get("grad_accum_steps", 1),
            clip_grad=get(cfg, "train", "clip_grad", default=1.0),
            best_metric_name=get(cfg, "train", "best_metric", default="roc_auc"),
            early_stop_patience=get(cfg, "train", "early_stop_patience", default=20),
            history_path=fpaths.metrics / "nnmamba_train_history.json",
        )
