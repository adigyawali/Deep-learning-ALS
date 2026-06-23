"""Stage: evaluate either model on the shared held-out TEST split.

Writes ``runs/<model>/metrics/evaluation.json`` and ``predictions.json`` in one
schema for both models, so CNN→ViT and CNN→nnMamba are compared apples-to-apples
(Goal 8). The operating threshold is the val-tuned one stored in the checkpoint,
never chosen on the test data it scores.
"""

from __future__ import annotations

import json
from collections import defaultdict

import torch
from torch.utils.data import Subset

from ..config import get
from ..data.feature_dataset import ALSSpatialFeatureDataset, indices_from
from ..data.volume_dataset import VolumeDataset
from ..models.cnn_nnmamba import CNNnnMamba
from ..models.cnn_vit import SpatialMultiModalViT
from ..paths import DEFAULT_DATA_DIR, RunPaths
from ..splits import indices_from_split, load_or_build_splits
from ..training import metrics as M
from ._common import make_loader, vit_forward, volume_forward


def _build_eval_target(cfg: dict, paths: RunPaths, device: torch.device):
    """Return (dataset, samples, forward_fn, model, ckpt_blob) for the active model."""
    if cfg["model"] == "cnn_vit":
        ds = ALSSpatialFeatureDataset(features_dir=paths.features)
        ckpt = paths.checkpoints / "vit_best.pt"
        if not ckpt.exists() or len(ds) == 0:
            return None
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        mc = blob.get("config", {}).get("vit", {}) or cfg.get("vit", {})
        model = SpatialMultiModalViT(
            in_channels=ds.in_channels, spatial_shape=ds.spatial_shape,
            embed_dim=mc.get("embed_dim", 384), depth=mc.get("depth", 6),
            num_heads=mc.get("num_heads", 6), mlp_ratio=mc.get("mlp_ratio", 4.0),
            dropout=mc.get("dropout", 0.15), modality_dropout_prob=0.0,
        ).to(device)
        model.load_state_dict(blob["model_state_dict"])
        return ds, ds.samples, vit_forward, model, blob

    # cnn_nnmamba
    data_dir = get(cfg, "data", "data_dir") or DEFAULT_DATA_DIR
    ds = VolumeDataset(data_dir, return_mode="stack",
                       target_shape=tuple(get(cfg, "data", "target_shape", default=[128, 128, 128])),
                       transform=False, use_frequency=bool(get(cfg, "data", "use_frequency", default=True)))
    ckpt = paths.checkpoints / "nnmamba_best.pt"
    if not ckpt.exists() or len(ds) == 0:
        return None
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    mc = blob.get("config", {}).get("nnmamba", {}) or cfg.get("nnmamba", {})
    model = CNNnnMamba(
        use_frequency=bool(get(cfg, "data", "use_frequency", default=True)),
        base=mc.get("base", 32), blocks=mc.get("blocks", 3), token_grid=mc.get("token_grid", 4),
        mamba_layers=mc.get("mamba_layers", 2), d_state=mc.get("d_state", 16), dropout=mc.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(blob["model_state_dict"])
    return ds, ds.samples, volume_forward, model, blob


def run(cfg: dict, paths: RunPaths, device: torch.device) -> None:
    target = _build_eval_target(cfg, paths, device)
    if target is None:
        print(f"[eval] missing best checkpoint or data for {cfg['model']}. Train it first.")
        return
    dataset, samples, forward_fn, model, blob = target
    model.eval()

    meta = dataset.to_sample_meta()
    splits = load_or_build_splits(meta, paths.splits_path, seed=cfg.get("seed", 42))
    test_idx = indices_from_split(meta, splits, "test")
    if not test_idx:
        print("[eval] test split is empty.")
        return

    loader = make_loader(Subset(dataset, test_idx), batch_size=8, shuffle=False,
                         dl_cfg=cfg.get("dataloader", {}), device=device)
    threshold = float(blob.get("threshold", blob.get("best_val_threshold", 0.5)))

    labels: list[int] = []
    probs: list[float] = []
    ids: list[str] = []
    test_samples = [samples[i] for i in test_idx]
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            logits, y = forward_fn(model, batch, device)
            p = torch.sigmoid(logits.float()).reshape(-1).cpu().tolist()
            yy = y.reshape(-1).cpu().tolist()
            probs.extend(p)
            labels.extend(int(v) for v in yy)
            for _ in range(len(p)):
                s = test_samples[cursor]
                ids.append(s["id"] if isinstance(s, dict) else s.sample_id)
                cursor += 1

    bootstrap_n = int(get(cfg, "eval", "bootstrap_n", default=2000))
    result = M.binary_metrics(labels, probs, threshold=threshold)
    result.update({
        "model": cfg["model"],
        "roc_auc_bootstrap_95ci": list(M.bootstrap_auc_ci(labels, probs, n_boot=bootstrap_n, seed=cfg.get("seed", 42))),
        "roc_auc_delong_95ci": list(M.delong_ci(labels, probs)),
        "brier_score": M.brier(labels, probs),
        "ece_10bin": M.expected_calibration_error(labels, probs, n_bins=10),
        "num_test_samples": len(test_idx),
        "best_val_metric_name": blob.get("best_metric_name"),
        "best_val_metric": blob.get("best_metric"),
        "epoch_of_best": blob.get("epoch"),
    })

    # Per-site metrics when more than one site is present.
    by_id = {(s["id"] if isinstance(s, dict) else s.sample_id): (s.get("site", "UNK") if isinstance(s, dict) else s.site)
             for s in test_samples}
    sites = defaultdict(lambda: {"labels": [], "probs": []})
    preds = [1 if p >= threshold else 0 for p in probs]
    for sid, l, p in zip(ids, labels, probs):
        sites[by_id.get(sid, "UNK")]["labels"].append(l)
        sites[by_id.get(sid, "UNK")]["probs"].append(p)
    if len(sites) > 1:
        result["per_site"] = {
            site: M.binary_metrics(d["labels"], d["probs"], threshold=threshold)
            for site, d in sites.items()
        }

    out_metrics = paths.metrics / "evaluation.json"
    out_preds = paths.metrics / "predictions.json"
    out_metrics.write_text(json.dumps(result, indent=2))
    out_preds.write_text(json.dumps(
        [{"id": i, "label": int(l), "prob_als": p, "pred": int(pr)}
         for i, l, p, pr in zip(ids, labels, probs, preds)], indent=2,
    ))
    print(f"--- {cfg['model']} evaluation (test) ---")
    print(json.dumps({k: v for k, v in result.items() if k != "per_site"}, indent=2))
    print(f"[eval] saved {out_metrics}")
    print(f"[eval] saved {out_preds}")
