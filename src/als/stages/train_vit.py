"""Stage: train the spatial multi-modal ViT on dumped CNN features (CNN→ViT stage 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Subset

from .. import sanity
from ..config import get
from ..data.feature_dataset import ALSSpatialFeatureDataset, compute_pos_weight, indices_from
from ..models.cnn_vit import SpatialMultiModalViT
from ..paths import RunPaths
from ..splits import load_or_build_splits, n_folds_in
from ..training import trainer
from ..training.optim import amp_dtype_from_str, warmup_cosine_scheduler
from ._common import make_loader, smoke_trim, vit_forward


def run(cfg: dict, paths: RunPaths, device: torch.device) -> None:
    v = cfg["vit"]
    dl = cfg.get("dataloader", {})

    # Splits are shared across folds; build (or read) them once. Any fold's
    # feature metadata is enough — subjects are keyed by id, not by fold.
    splits = None
    n_folds = get(cfg, "split", "n_folds", default=5)

    for fold in range(n_folds):
        fpaths = paths.fold(fold).ensure()
        dataset = ALSSpatialFeatureDataset(features_dir=fpaths.features)
        if len(dataset) < 3:
            print(f"[vit] fold {fold}: need >=3 *_spatial.pt files in {fpaths.features}. "
                  f"Run extract_features first — skipping.")
            continue

        if splits is None:
            splits = load_or_build_splits(
                dataset.to_sample_meta(), paths.splits_path,
                n_folds=n_folds,
                test_ratio=get(cfg, "split", "test_ratio", default=0.2),
                seed=cfg.get("seed", 42),
            )

        train_idx = smoke_trim(indices_from(dataset.samples, splits, "train", fold), cfg)
        val_idx = smoke_trim(indices_from(dataset.samples, splits, "val", fold), cfg)
        if not train_idx or not val_idx:
            print(f"[vit] fold {fold}: empty train or val split — skipping.")
            continue
        print(f"\n[vit] ===== fold {fold + 1}/{n_folds} "
              f"(train={len(train_idx)} val={len(val_idx)}) =====")

        train_loader = make_loader(Subset(dataset, train_idx), batch_size=v["batch_size"],
                                   shuffle=True, dl_cfg=dl, device=device)
        val_loader = make_loader(Subset(dataset, val_idx), batch_size=v["batch_size"],
                                 shuffle=False, dl_cfg=dl, device=device)

        model = SpatialMultiModalViT(
            in_channels=dataset.in_channels, spatial_shape=dataset.spatial_shape,
            embed_dim=v["embed_dim"], depth=v["depth"], num_heads=v["num_heads"],
            mlp_ratio=v.get("mlp_ratio", 4.0), dropout=v.get("dropout", 0.15),
            modality_dropout_prob=v.get("modality_dropout_prob", 0.0),
        ).to(device)

        pos_weight = compute_pos_weight(dataset.samples, train_idx).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=v["lr"], weight_decay=v["weight_decay"])
        scheduler = warmup_cosine_scheduler(optimizer, v["epochs"], v.get("warmup_epochs", 5))

        sanity.preflight(stage=f"train_vit[fold{fold}]", model=model, dataset=dataset, splits=splits,
                         train_loader=train_loader, forward_fn=vit_forward, device=device,
                         ckpt_dir=fpaths.checkpoints, ckpt_prefix="vit")

        trainer.fit(
            model=model, train_loader=train_loader, val_loader=val_loader,
            forward_fn=vit_forward, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            device=device, epochs=v["epochs"], ckpt_dir=fpaths.checkpoints, ckpt_prefix="vit",
            config=cfg,
            amp_dtype=amp_dtype_from_str(get(cfg, "train", "amp", default="bf16"), device),
            clip_grad=get(cfg, "train", "clip_grad", default=1.0),
            best_metric_name=get(cfg, "train", "best_metric", default="roc_auc"),
            early_stop_patience=get(cfg, "train", "early_stop_patience", default=15),
            history_path=fpaths.metrics / "vit_train_history.json",
        )
