"""Stage: fine-tune the tri-stream 3D CNN (CNN→ViT stage 1)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Subset

from .. import sanity
from ..config import get
from ..data.volume_dataset import VolumeDataset
from ..models.cnn_encoder import ALSTriStreamClassifier
from ..paths import DEFAULT_DATA_DIR, RunPaths
from ..splits import indices_from_split, load_or_build_splits
from ..training import trainer
from ..training.optim import amp_dtype_from_str, pos_weight_from_labels, warmup_cosine_scheduler
from ._common import cnn_forward, make_loader, smoke_trim


def _build_optimizer(model, lr_backbone, lr_head, wd):
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if "_backbone" in name else head).append(p)
    print(f"[cnn] optimizer: {len(backbone)} backbone params, {len(head)} head params")
    return torch.optim.AdamW(
        [{"params": backbone, "lr": lr_backbone, "weight_decay": wd},
         {"params": head, "lr": lr_head, "weight_decay": wd}],
        betas=(0.9, 0.999), eps=1e-8,
    )


def run(cfg: dict, paths: RunPaths, device: torch.device, *, resume: bool = False) -> None:
    data_dir = get(cfg, "data", "data_dir") or DEFAULT_DATA_DIR
    target_shape = tuple(get(cfg, "data", "target_shape", default=[128, 128, 128]))
    aug_level = get(cfg, "data", "aug_level", default="medium")
    c = cfg["cnn"]

    full = VolumeDataset(data_dir, return_mode="tuple", target_shape=target_shape, transform=False)
    if len(full) < 3:
        print(f"[cnn] Error: fewer than 3 samples in {data_dir}.")
        return
    train_aug = VolumeDataset(data_dir, return_mode="tuple", target_shape=target_shape,
                              transform=True, aug_level=aug_level)

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
        print("[cnn] Error: empty train or val split.")
        return

    dl = cfg.get("dataloader", {})
    train_loader = make_loader(Subset(train_aug, train_idx), batch_size=c["batch_size"],
                               shuffle=True, dl_cfg=dl, device=device)
    val_loader = make_loader(Subset(full, val_idx), batch_size=c["batch_size"],
                             shuffle=False, dl_cfg=dl, device=device)

    model = ALSTriStreamClassifier(backbone=c.get("backbone", "resnet50"),
                                   freeze_backbone=c.get("freeze_backbone", False)).to(device)

    pw = pos_weight_from_labels([meta[i].label for i in train_idx])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32, device=device))
    optimizer = _build_optimizer(model, c["lr_backbone"], c["lr_head"], c["weight_decay"])
    scheduler = warmup_cosine_scheduler(optimizer, c["epochs"], c.get("warmup_epochs", 5))

    sanity.preflight(stage="train_cnn", model=model, dataset=full, splits=splits,
                     train_loader=train_loader, forward_fn=cnn_forward, device=device,
                     ckpt_dir=paths.checkpoints, ckpt_prefix="cnn")

    trainer.fit(
        model=model, train_loader=train_loader, val_loader=val_loader,
        forward_fn=cnn_forward, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, epochs=c["epochs"], ckpt_dir=paths.checkpoints, ckpt_prefix="cnn",
        config=cfg, splits_path=str(paths.splits_path),
        amp_dtype=amp_dtype_from_str(get(cfg, "train", "amp", default="bf16"), device),
        grad_accum_steps=c.get("grad_accum_steps", 1),
        clip_grad=get(cfg, "train", "clip_grad", default=1.0),
        best_metric_name=get(cfg, "train", "best_metric", default="roc_auc"),
        early_stop_patience=get(cfg, "train", "early_stop_patience", default=15),
        resume=resume, history_path=paths.metrics / "cnn_train_history.json",
    )
