"""End-to-end (tiny, synthetic) check of the CNN→nnMamba train + evaluate path,
focused on the pooled out-of-fold (OOF) threshold fix.

Builds a small fake ``Data/processed`` (both classes, enough subjects for a
2-fold CV + held-out test), trains one epoch per fold on the from-scratch stem,
then runs evaluation and asserts:

  * a pooled-OOF block is produced in ``cv_summary.json``;
  * the OOF threshold is a single number derived from the pooled val predictions
    (not the mean of the per-fold thresholds); and
  * the held-out TEST ensemble is decided with *that* OOF threshold — i.e. one
    stable operating point, never tuned on the test data it scores.

Runs on CPU with tiny volumes; no network (conftest sets ALS_SKIP_PRETRAINED).
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from als.paths import build_run_paths
from als.stages import evaluate, train_nnmamba


def _nifti(path: Path, shape=(8, 8, 8), seed=0):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(rng.normal(size=shape).astype(np.float32).clip(min=0), np.eye(4)), str(path))


def _subject(root: Path, sample_id: str, seed=0):
    for i, m in enumerate(("T1", "T2", "FLAIR")):
        _nifti(root / sample_id / f"{sample_id}_{m}.nii.gz", seed=seed + i)


def _make_cfg(data_dir: Path) -> dict:
    return {
        "model": "cnn_nnmamba",
        "seed": 0,
        "data": {"data_dir": str(data_dir), "target_shape": [8, 8, 8],
                 "use_frequency": False, "aug_level": "light"},
        "split": {"n_folds": 2, "test_ratio": 0.2},
        "dataloader": {"num_workers": 0, "pin_memory": False, "persistent_workers": False},
        "nnmamba": {"spatial_encoder": "scratch", "base": 8, "blocks": 2, "token_grid": 2,
                    "mamba_layers": 1, "d_state": 8, "dropout": 0.1, "epochs": 1,
                    "batch_size": 2, "grad_accum_steps": 1, "lr": 1e-3,
                    "weight_decay": 1e-4, "warmup_epochs": 0},
        "train": {"best_metric": "roc_auc", "label_smoothing": 0.1,
                  "early_stop_patience": 99, "clip_grad": 1.0, "amp": "none"},
        "eval": {"bootstrap_n": 25},
    }


def test_train_and_evaluate_uses_pooled_oof_threshold(tmp_path: Path):
    data_dir = tmp_path / "processed"
    # 10 controls + 10 patients → after 20% test, a 2-fold CV with both classes per fold.
    for i in range(10):
        _subject(data_dir, f"C{i + 100:03d}_V1", seed=i)
        _subject(data_dir, f"P{i + 100:03d}_V1", seed=100 + i)

    cfg = _make_cfg(data_dir)
    paths = build_run_paths("cnn_nnmamba", tmp_path / "runs").ensure()
    device = torch.device("cpu")

    train_nnmamba.run(cfg, paths, device)
    evaluate.run(cfg, paths, device)

    cv = json.loads((paths.metrics / "cv_summary.json").read_text())
    te = json.loads((paths.metrics / "test_evaluation.json").read_text())

    # Pooled-OOF block exists and covers every CV-pool sample (each once).
    assert cv["oof_metrics"] is not None
    assert "roc_auc" in cv["oof_metrics"]
    assert cv["oof_num_samples"] > 0
    assert isinstance(cv["per_fold_thresholds"], list) and len(cv["per_fold_thresholds"]) == 2

    # The held-out test ensemble is decided with the single pooled-OOF threshold.
    assert te["ensemble_threshold"] == cv["oof_threshold"]
    assert te["ensemble"]["threshold"] == cv["oof_threshold"]
