"""Stage: train a raw-volume Mamba classifier — either variant.

Serves both ``--model cnn_nnmamba`` (two-stage: CNN stem → Mamba) and
``--model nnmamba`` (one-stage: patch embedding → Mamba, no CNN). The two differ
only in the module built from the ``nnmamba:`` config block; the dataset, folds,
loss, and training loop are identical, which is what makes them comparable.
"""

from __future__ import annotations

import torch
from torch.utils.data import Subset

from .. import sanity
from ..config import get, resolve_streams
from ..data.mixup import build_mixup
from ..data.volume_dataset import VolumeDataset
from ..models import build_volume_model
from ..models.components.mamba_block import MAMBA_BACKEND
from ..paths import DEFAULT_DATA_DIR, RunPaths
from ..splits import indices_from_split, n_folds_in, resolve_splits
from ..training import trainer
from ..training.optim import (
    SmoothBCEWithLogitsLoss, amp_dtype_from_str, pos_weight_from_labels, warmup_cosine_scheduler,
)
from ._common import make_loader, smoke_trim, volume_forward


# Above this many tokens the pure-PyTorch Mamba fallback stops being "adequate":
# its scan is an O(L) Python loop, so wall-clock grows faster than linearly in L.
# Measured on CPU at 128³ / d_model 192 / 4 bidirectional layers, one fwd+bwd at
# batch 2: 128 tokens 1.5s, 512 tokens 16s, 1024 tokens 68s.
_SLOW_FALLBACK_TOKENS = 256


def _warn_if_sequence_is_slow(model, model_name: str) -> None:
    """Say so loudly when the token sequence is long AND the backend is the slow one."""
    seq = getattr(model, "sequence_length", None)
    if seq is None or MAMBA_BACKEND != "pytorch-fallback" or seq <= _SLOW_FALLBACK_TOKENS:
        return
    lever = ("nnmamba.patch_size (16 -> 32 cuts the sequence 8x)" if model_name == "nnmamba"
             else "nnmamba.token_grid (4 -> 3 cuts the sequence ~2.4x)")
    print(f"[{model_name}] WARNING: Mamba sequence is {seq} tokens and the active backend is "
          f"'pytorch-fallback', whose scan is an O(L) Python loop — this will be very slow.\n"
          f"[{model_name}]   Fix, best first: install the mamba-ssm CUDA kernel; "
          f"raise {lever}; or use data.streams: spatial (halves the sequence).")


def run(cfg: dict, paths: RunPaths, device: torch.device) -> None:
    model_name = cfg["model"]
    data_dir = get(cfg, "data", "data_dir") or DEFAULT_DATA_DIR
    target_shape = tuple(get(cfg, "data", "target_shape", default=[128, 128, 128]))
    streams = resolve_streams(cfg)
    aug_level = get(cfg, "data", "aug_level", default="medium")
    aug_config = cfg.get("augmentations")   # from root config.yaml (source of truth)
    mixup = build_mixup(aug_config)         # batch-level, so applied by the trainer
    m = cfg["nnmamba"]
    if model_name == "cnn_nnmamba":
        spatial_encoder = m.get("spatial_encoder", "scratch")
        arch = f"two-stage (CNN→Mamba)  spatial_encoder={spatial_encoder}" + (
            f" (backbone={m.get('backbone', 'resnet10')}, "
            f"freeze={m.get('freeze_backbone', True)})" if spatial_encoder == "pretrained" else "")
    else:
        arch = (f"one-stage (patch→Mamba, no CNN)  patch_size={m.get('patch_size', 16)} "
                f"d_model={m.get('d_model', 192)} bidirectional={m.get('bidirectional', True)}")
    print(f"[{model_name}] Mamba backend: {MAMBA_BACKEND}  streams={streams}  {arch}")

    full = VolumeDataset(data_dir, return_mode="stack", target_shape=target_shape,
                         transform=False)
    if len(full) < 3:
        print(f"[{model_name}] Error: fewer than 3 samples in {data_dir}.")
        return
    train_aug = VolumeDataset(data_dir, return_mode="stack", target_shape=target_shape,
                              transform=True, aug_level=aug_level, aug_config=aug_config)

    splits = resolve_splits(
        full.to_sample_meta(), paths.splits_path,
        cv_cfg=cfg.get("cross_validation"), split_cfg=cfg.get("split"),
        seed=cfg.get("seed", 42),
    )
    meta = full.to_sample_meta()
    n_folds = n_folds_in(splits)
    dl = cfg.get("dataloader", {})

    # Train one independent Mamba model per CV fold, each into runs/<model>/fold{k}/.
    for fold in range(n_folds):
        fpaths = paths.fold(fold).ensure()
        train_idx = smoke_trim(indices_from_split(meta, splits, "train", fold), cfg)
        val_idx = smoke_trim(indices_from_split(meta, splits, "val", fold), cfg)
        if not train_idx or not val_idx:
            print(f"[{model_name}] fold {fold}: empty train or val split — skipping.")
            continue
        print(f"\n[{model_name}] ===== fold {fold + 1}/{n_folds} "
              f"(train={len(train_idx)} val={len(val_idx)}) =====")

        train_loader = make_loader(Subset(train_aug, train_idx), batch_size=m["batch_size"],
                                   shuffle=True, dl_cfg=dl, device=device)
        val_loader = make_loader(Subset(full, val_idx), batch_size=m["batch_size"],
                                 shuffle=False, dl_cfg=dl, device=device)

        model = build_volume_model(
            model_name, m, streams=streams, input_shape=target_shape,
        ).to(device)
        if fold == 0:
            _warn_if_sequence_is_slow(model, model_name)

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
            mixup=mixup,
            grad_accum_steps=m.get("grad_accum_steps", 1),
            clip_grad=get(cfg, "train", "clip_grad", default=1.0),
            best_metric_name=get(cfg, "train", "best_metric", default="roc_auc"),
            early_stop_patience=get(cfg, "train", "early_stop_patience", default=20),
            history_path=fpaths.metrics / "nnmamba_train_history.json",
        )
