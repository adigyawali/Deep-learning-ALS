"""Stage: cross-validated evaluation of either model.

With 5-fold CV there are ``n_folds`` trained models (one per fold), plus one
fixed, subject-level held-out TEST set shared by every fold. This stage:

  * scores each fold's best model on **its own validation fold** and aggregates
    those into ``cv_summary.json`` (mean ± std over folds) — the cross-validated
    estimate of generalization;
  * scores each fold's model on the **shared TEST set** and reports both the
    per-fold spread (mean ± std) and a **mean-probability ensemble** of the five
    models in ``test_evaluation.json`` — the headline held-out number.

Every operating threshold is the val-tuned one stored in each fold's checkpoint
(the ensemble uses the mean of the five), so no threshold is ever chosen on the
test data it scores. Both models write the same schema so CNN→ViT and
CNN→nnMamba are compared apples-to-apples (Goal 8).
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict

import torch
from torch.utils.data import Subset

from ..config import get
from ..data.feature_dataset import ALSSpatialFeatureDataset
from ..data.volume_dataset import VolumeDataset
from ..models.cnn_nnmamba import CNNnnMamba
from ..models.cnn_vit import SpatialMultiModalViT
from ..paths import DEFAULT_DATA_DIR, RunPaths
from ..splits import indices_from_split, n_folds_in, read_splits
from ..training import metrics as M
from ._common import make_loader, vit_forward, volume_forward


def _sid(sample) -> str:
    return sample["id"] if isinstance(sample, dict) else sample.sample_id


def _site(sample) -> str:
    return (sample.get("site", "UNK") if isinstance(sample, dict) else sample.site) or "UNK"


def _load_fold_target(cfg: dict, paths: RunPaths, fold: int, device, shared_ds):
    """Return (dataset, samples, forward_fn, model, blob) for one fold, or None."""
    fpaths = paths.fold(fold)
    if cfg["model"] == "cnn_vit":
        ds = ALSSpatialFeatureDataset(features_dir=fpaths.features)
        ckpt = fpaths.checkpoints / "vit_best.pt"
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
        model.eval()
        return ds, ds.samples, vit_forward, model, blob

    # cnn_nnmamba — the raw-volume dataset is shared across folds; only the
    # checkpoint changes per fold.
    ckpt = fpaths.checkpoints / "nnmamba_best.pt"
    if not ckpt.exists() or len(shared_ds) == 0:
        return None
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    mc = blob.get("config", {}).get("nnmamba", {}) or cfg.get("nnmamba", {})
    model = CNNnnMamba(
        use_frequency=bool(get(cfg, "data", "use_frequency", default=True)),
        base=mc.get("base", 32), blocks=mc.get("blocks", 3), token_grid=mc.get("token_grid", 4),
        mamba_layers=mc.get("mamba_layers", 2), d_state=mc.get("d_state", 16), dropout=mc.get("dropout", 0.1),
        spatial_encoder=mc.get("spatial_encoder", "scratch"), backbone=mc.get("backbone", "resnet18"),
        freeze_backbone=mc.get("freeze_backbone", True), pretrained_d_model=mc.get("pretrained_d_model", 256),
        load_pretrained=False,  # the checkpoint carries the backbone weights; no download needed
    ).to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return shared_ds, shared_ds.samples, volume_forward, model, blob


def _infer(dataset, indices, forward_fn, model, device, cfg):
    """Return (labels, probs) aligned to `indices` order (loader is unshuffled)."""
    loader = make_loader(Subset(dataset, indices), batch_size=8, shuffle=False,
                         dl_cfg=cfg.get("dataloader", {}), device=device)
    labels: list[int] = []
    probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits, y = forward_fn(model, batch, device)
            probs.extend(torch.sigmoid(logits.float()).reshape(-1).cpu().tolist())
            labels.extend(int(v) for v in y.reshape(-1).cpu().tolist())
    return labels, probs


def _aggregate(metric_dicts: list[dict]) -> dict:
    """Mean/std across folds for every numeric metric (skips confusion_matrix)."""
    if not metric_dicts:
        return {}
    keys = [k for k, v in metric_dicts[0].items() if isinstance(v, (int, float))]
    out: dict[str, dict] = {}
    for k in keys:
        vals = [d[k] for d in metric_dicts
                if isinstance(d.get(k), (int, float)) and d[k] == d[k]]  # drop NaN
        if not vals:
            out[k] = {"mean": float("nan"), "std": float("nan"), "n_folds": 0}
            continue
        out[k] = {
            "mean": float(statistics.fmean(vals)),
            "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
            "n_folds": len(vals),
        }
    return out


def run(cfg: dict, paths: RunPaths, device: torch.device) -> None:
    if not paths.splits_path.exists():
        print(f"[eval] {paths.splits_path} not found. Train first.")
        return
    splits = read_splits(paths.splits_path)
    n_folds = n_folds_in(splits)

    # nnMamba reuses one raw-volume dataset across folds; ViT rebuilds per fold.
    shared_ds = None
    if cfg["model"] == "cnn_nnmamba":
        data_dir = get(cfg, "data", "data_dir") or DEFAULT_DATA_DIR
        shared_ds = VolumeDataset(
            data_dir, return_mode="stack",
            target_shape=tuple(get(cfg, "data", "target_shape", default=[128, 128, 128])),
            transform=False, use_frequency=bool(get(cfg, "data", "use_frequency", default=True)),
        )

    val_metrics_per_fold: list[dict] = []
    test_metrics_per_fold: list[dict] = []
    fold_thresholds: list[float] = []
    test_prob_by_fold: list[dict[str, float]] = []      # id -> prob, per fold
    test_label_by_id: dict[str, int] = {}
    test_site_by_id: dict[str, str] = {}
    best_val_metric_name = None
    best_val_metrics: list[float] = []
    # Pooled out-of-fold (OOF) validation predictions: each CV-pool subject lands
    # in exactly one validation fold, so concatenating them yields one ~N_cv-sized
    # held-out prediction set. It gives a single, low-variance threshold and CV
    # estimate — far more stable than averaging five thresholds tuned on ~70
    # samples each (which previously swung 0.12–0.65 across folds).
    oof_labels: list[int] = []
    oof_probs: list[float] = []

    for fold in range(n_folds):
        target = _load_fold_target(cfg, paths, fold, device, shared_ds)
        if target is None:
            print(f"[eval] fold {fold}: no best checkpoint / features — skipping.")
            continue
        dataset, samples, forward_fn, model, blob = target
        meta = dataset.to_sample_meta()
        thr = float(blob.get("threshold", blob.get("best_val_threshold", 0.5)))
        fold_thresholds.append(thr)
        best_val_metric_name = blob.get("best_metric_name", best_val_metric_name)
        if isinstance(blob.get("best_metric"), (int, float)):
            best_val_metrics.append(float(blob["best_metric"]))

        # Validation-fold metrics (the CV estimate).
        val_idx = indices_from_split(meta, splits, "val", fold)
        if val_idx:
            v_labels, v_probs = _infer(dataset, val_idx, forward_fn, model, device, cfg)
            val_metrics_per_fold.append(M.binary_metrics(v_labels, v_probs, threshold=thr))
            oof_labels.extend(int(v) for v in v_labels)
            oof_probs.extend(v_probs)

        # Fixed held-out TEST metrics for this fold's model.
        test_idx = indices_from_split(meta, splits, "test")
        if test_idx:
            t_labels, t_probs = _infer(dataset, test_idx, forward_fn, model, device, cfg)
            test_metrics_per_fold.append({"fold": fold,
                                          **M.binary_metrics(t_labels, t_probs, threshold=thr)})
            prob_map = {}
            for i, p, l in zip(test_idx, t_probs, t_labels):
                sid = _sid(samples[i])
                prob_map[sid] = p
                test_label_by_id[sid] = int(l)
                test_site_by_id[sid] = _site(samples[i])
            test_prob_by_fold.append(prob_map)
        print(f"[eval] fold {fold}: thr={thr:.2f}  val_n={len(val_idx)}  test_n={len(test_idx)}")

    if not fold_thresholds:
        print("[eval] no trained folds found. Train the model first.")
        return

    # ── Pooled out-of-fold estimate (the primary CV number + shared threshold) ─
    bootstrap_n = int(get(cfg, "eval", "bootstrap_n", default=2000))
    oof = None
    oof_threshold = float(statistics.fmean(fold_thresholds))  # fallback if OOF is empty
    if oof_labels and (0 in oof_labels and 1 in oof_labels):
        oof_threshold = float(M.youden_threshold(oof_labels, oof_probs))
        oof = M.binary_metrics(oof_labels, oof_probs, threshold=oof_threshold)
        oof.update({
            "roc_auc_bootstrap_95ci": list(M.bootstrap_auc_ci(
                oof_labels, oof_probs, n_boot=bootstrap_n, seed=cfg.get("seed", 42))),
            "roc_auc_delong_95ci": list(M.delong_ci(oof_labels, oof_probs)),
            "brier_score": M.brier(oof_labels, oof_probs),
            "ece_10bin": M.expected_calibration_error(oof_labels, oof_probs, n_bins=10),
        })

    # ── CV summary (validation folds) ──────────────────────────────────────
    cv_summary = {
        "model": cfg["model"],
        "n_folds_evaluated": len(val_metrics_per_fold),
        "best_val_metric_name": best_val_metric_name,
        "best_val_metric_mean": float(statistics.fmean(best_val_metrics)) if best_val_metrics else None,
        "best_val_metric_std": float(statistics.pstdev(best_val_metrics)) if len(best_val_metrics) > 1 else 0.0,
        # Pooled OOF is the headline CV estimate; per-fold mean±std is kept for the
        # spread. Both are reported so the two are never confused.
        "oof_threshold": oof_threshold,
        "oof_num_samples": len(oof_labels),
        "oof_metrics": oof,
        "per_fold_thresholds": fold_thresholds,
        "val_metrics_mean_std": _aggregate(val_metrics_per_fold),
        "val_metrics_per_fold": val_metrics_per_fold,
    }

    # ── Held-out test: per-fold spread + mean-probability ensemble ──────────
    # Use the single pooled-OOF threshold (stable) rather than the mean of five
    # per-fold thresholds tuned on ~70 samples each. Never tuned on the test set.
    ens_threshold = oof_threshold
    common_ids = sorted(set.intersection(*[set(d) for d in test_prob_by_fold])) if test_prob_by_fold else []
    ens_labels = [test_label_by_id[i] for i in common_ids]
    ens_probs = [float(statistics.fmean(d[i] for d in test_prob_by_fold)) for i in common_ids]

    ensemble = M.binary_metrics(ens_labels, ens_probs, threshold=ens_threshold)
    ensemble.update({
        "roc_auc_bootstrap_95ci": list(M.bootstrap_auc_ci(ens_labels, ens_probs, n_boot=bootstrap_n, seed=cfg.get("seed", 42))),
        "roc_auc_delong_95ci": list(M.delong_ci(ens_labels, ens_probs)),
        "brier_score": M.brier(ens_labels, ens_probs),
        "ece_10bin": M.expected_calibration_error(ens_labels, ens_probs, n_bins=10),
    })

    test_evaluation = {
        "model": cfg["model"],
        "n_folds": len(test_prob_by_fold),
        "num_test_samples": len(common_ids),
        "ensemble_threshold": ens_threshold,
        "ensemble": ensemble,
        "per_fold_mean_std": _aggregate([{k: v for k, v in m.items() if k != "fold"}
                                         for m in test_metrics_per_fold]),
        "per_fold": test_metrics_per_fold,
    }

    # Per-site ensemble metrics when more than one site is present in the test set.
    ens_preds = [1 if p >= ens_threshold else 0 for p in ens_probs]
    sites = defaultdict(lambda: {"labels": [], "probs": []})
    for sid, l, p in zip(common_ids, ens_labels, ens_probs):
        sites[test_site_by_id.get(sid, "UNK")]["labels"].append(l)
        sites[test_site_by_id.get(sid, "UNK")]["probs"].append(p)
    if len(sites) > 1:
        test_evaluation["ensemble_per_site"] = {
            site: M.binary_metrics(d["labels"], d["probs"], threshold=ens_threshold)
            for site, d in sites.items()
        }

    (paths.metrics / "cv_summary.json").write_text(json.dumps(cv_summary, indent=2))
    (paths.metrics / "test_evaluation.json").write_text(json.dumps(test_evaluation, indent=2))
    (paths.metrics / "predictions.json").write_text(json.dumps(
        [{"id": i, "label": int(l), "prob_als": p, "pred": int(pr)}
         for i, l, p, pr in zip(common_ids, ens_labels, ens_probs, ens_preds)], indent=2,
    ))

    print(f"--- {cfg['model']} cross-validated evaluation ---")
    print("[eval] CV (per-fold mean±std):",
          json.dumps(cv_summary["val_metrics_mean_std"].get("roc_auc", {}), indent=2))
    if oof is not None:
        print(f"[eval] CV (pooled OOF, n={len(oof_labels)}): "
              f"roc_auc={oof['roc_auc']:.3f}  bal_acc={oof['balanced_accuracy']:.3f}  "
              f"thr={oof_threshold:.2f}  ece={oof['ece_10bin']:.3f}")
    print("[eval] TEST ensemble:", json.dumps({k: v for k, v in ensemble.items()
                                               if k not in ("confusion_matrix",)}, indent=2))
    print(f"[eval] saved {paths.metrics / 'cv_summary.json'}")
    print(f"[eval] saved {paths.metrics / 'test_evaluation.json'}")
    print(f"[eval] saved {paths.metrics / 'predictions.json'}")
