"""Root config.yaml is the single source of truth: it is merged into every
model config, and its explicit folds + augmentations drive the real pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from als import config as cfgmod
from als.paths import build_run_paths
from als.stages import evaluate, train_nnmamba


def _nifti(path: Path, shape=(8, 8, 8), seed=0):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(rng.normal(size=shape).astype(np.float32).clip(min=0), np.eye(4)), str(path))


def _subject(root: Path, sample_id: str, seed=0):
    for i, m in enumerate(("T1", "T2", "FLAIR")):
        _nifti(root / sample_id / f"{sample_id}_{m}.nii.gz", seed=seed + i)


def test_load_config_merges_root_augmentations_and_cv(tmp_path: Path, monkeypatch):
    root = tmp_path / "config.yaml"
    root.write_text(
        "augmentations:\n"
        "  enabled: true\n"
        "  transforms:\n"
        "    - {name: RandFlip, group: geometric, params: {prob: 0.5, spatial_axis: 0}}\n"
        "cross_validation:\n"
        "  mode: explicit\n"
        "  test_subjects: [C001, P001]\n"
        "  folds:\n"
        "    - [C002, P002]\n"
        "    - [C003, P003]\n"
    )
    monkeypatch.setattr(cfgmod, "ROOT_CONFIG", root)
    cfg = cfgmod.load_config("cnn_nnmamba")
    assert cfg["augmentations"]["enabled"] is True
    assert cfg["augmentations"]["transforms"][0]["name"] == "RandFlip"
    assert cfg["cross_validation"]["mode"] == "explicit"
    assert cfg["cross_validation"]["folds"] == [["C002", "P002"], ["C003", "P003"]]


def _explicit_cfg(data_dir: Path) -> dict:
    return {
        "model": "cnn_nnmamba",
        "seed": 0,
        "data": {"data_dir": str(data_dir), "target_shape": [8, 8, 8], "use_frequency": False},
        "dataloader": {"num_workers": 0, "pin_memory": False, "persistent_workers": False},
        # Custom augmentation policy straight from "config" (no aug_level anywhere).
        "augmentations": {"enabled": True, "transforms": [
            {"name": "RandFlip", "group": "geometric", "params": {"prob": 0.5, "spatial_axis": 0}},
            {"name": "RandGaussianNoise", "group": "intensity", "params": {"prob": 0.3, "std": 0.05}},
        ]},
        # Supervisor-approved explicit folds.
        "cross_validation": {"mode": "explicit",
                             "test_subjects": ["C001", "P001"],
                             "folds": [["C002", "C003", "P002", "P003"],
                                       ["C004", "C005", "P004", "P005"]]},
        "nnmamba": {"spatial_encoder": "scratch", "base": 8, "blocks": 2, "token_grid": 2,
                    "mamba_layers": 1, "d_state": 8, "dropout": 0.1, "epochs": 1,
                    "batch_size": 2, "grad_accum_steps": 1, "lr": 1e-3,
                    "weight_decay": 1e-4, "warmup_epochs": 0},
        "train": {"best_metric": "roc_auc", "label_smoothing": 0.0,
                  "early_stop_patience": 99, "clip_grad": 1.0, "amp": "none"},
        "eval": {"bootstrap_n": 25},
    }


def test_explicit_folds_drive_the_training_pipeline(tmp_path: Path):
    data_dir = tmp_path / "processed"
    for i in range(1, 6):
        _subject(data_dir, f"C{i:03d}_V1", seed=i)
        _subject(data_dir, f"P{i:03d}_V1", seed=100 + i)

    cfg = _explicit_cfg(data_dir)
    paths = build_run_paths("cnn_nnmamba", tmp_path / "runs").ensure()
    device = torch.device("cpu")

    train_nnmamba.run(cfg, paths, device)
    evaluate.run(cfg, paths, device)

    # The written splits.json is exactly the configured explicit split.
    sp = json.loads(paths.splits_path.read_text())
    assert sp["mode"] == "explicit" and sp["n_folds"] == 2
    assert set(sp["test_subjects"]) == {"C001", "P001"}
    assert set(sp["folds"][0]["val_subjects"]) == {"C002", "C003", "P002", "P003"}
    assert set(sp["folds"][1]["val_subjects"]) == {"C004", "C005", "P004", "P005"}

    # Two fold checkpoints were trained on those explicit folds, and eval ran.
    assert (paths.fold(0).checkpoints / "nnmamba_best.pt").exists()
    assert (paths.fold(1).checkpoints / "nnmamba_best.pt").exists()
    assert (paths.metrics / "test_evaluation.json").exists()
