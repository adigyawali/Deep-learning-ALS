"""Config loading, CLI overrides, smoke-mode shrinking, and snapshotting.

Configs live in ``configs/<model>.yaml`` and are plain nested dicts. The driver
loads one, applies a few CLI overrides, optionally shrinks it for ``--smoke``,
and saves the resolved result next to the run's outputs so every result is
reproducible from the config that produced it (Goal 8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs"
# Single, shared, root-level source of truth for data augmentations and K-fold
# splits (see config.yaml). Merged into every model's config below.
ROOT_CONFIG = PROJECT_ROOT / "config.yaml"
# Sections owned by the root config; if present there, they win over the model YAML.
_ROOT_SECTIONS = ("augmentations", "cross_validation")


def load_config(model: str, path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else CONFIG_DIR / f"{model}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg["model"] = model
    cfg["_config_path"] = str(cfg_path)
    _merge_root_config(cfg)
    return cfg


def _merge_root_config(cfg: dict) -> None:
    """Overlay the root ``config.yaml``'s augmentation + CV sections onto ``cfg``.

    Keeping augmentations and folds in one shared file (not duplicated per model)
    makes it the single place to edit, and guarantees both models train on the
    identical augmentation policy and the identical folds. Absent file/section →
    the model YAML's own ``data.aug_level`` / ``split`` fallbacks are used, so
    older setups keep working.
    """
    if not ROOT_CONFIG.exists():
        return
    root = yaml.safe_load(ROOT_CONFIG.read_text()) or {}
    for section in _ROOT_SECTIONS:
        if section in root and root[section] is not None:
            cfg[section] = root[section]
    cfg["_root_config_path"] = str(ROOT_CONFIG)


def _train_section(cfg: dict) -> dict:
    """The model-specific hyperparameter block ('vit'+'cnn' or 'nnmamba')."""
    return cfg["cnn" if cfg["model"] == "cnn_vit" else "nnmamba"]


def apply_overrides(cfg: dict, *, batch_size=None, epochs=None, lr=None,
                    num_workers=None, target_shape=None) -> dict:
    """Apply the handful of CLI flags that override config values."""
    sec = _train_section(cfg)
    if cfg["model"] == "cnn_vit":
        # batch/epochs/lr apply to BOTH cnn and vit sub-stages for convenience.
        for k, v in (("batch_size", batch_size), ("epochs", epochs)):
            if v is not None:
                cfg["cnn"][k] = v
                cfg["vit"][k] = v
        if lr is not None:
            cfg["cnn"]["lr_head"] = lr
            cfg["vit"]["lr"] = lr
    else:
        if batch_size is not None:
            sec["batch_size"] = batch_size
        if epochs is not None:
            sec["epochs"] = epochs
        if lr is not None:
            sec["lr"] = lr
    if num_workers is not None:
        cfg.setdefault("dataloader", {})["num_workers"] = num_workers
    if target_shape is not None:
        cfg.setdefault("data", {})["target_shape"] = list(target_shape)
    return cfg


def apply_smoke(cfg: dict) -> dict:
    """Shrink a config so the whole pipeline runs in seconds on a tiny subset.

    Used by ``--smoke`` to confirm the wiring (load → forward → loss → backward
    → checkpoint) before committing GPU hours.
    """
    cfg.setdefault("data", {})["target_shape"] = [32, 32, 32]
    cfg["smoke"] = True
    cfg["smoke_max_samples"] = 6
    cfg.setdefault("dataloader", {})["num_workers"] = 0
    cfg.setdefault("eval", {})["bootstrap_n"] = 50
    cfg.setdefault("train", {})["early_stop_patience"] = 99
    # A wiring check must not depend on the real config.yaml: disable augmentation
    # (fast, deterministic) and force auto 2-fold CV so any explicit patient-ID
    # folds (which name real subjects) don't apply to the tiny smoke subset.
    cfg["augmentations"] = {"enabled": False}
    cfg["cross_validation"] = {"mode": "auto", "n_folds": 2, "test_ratio": 0.2}
    cfg.setdefault("split", {})["n_folds"] = 2
    if cfg["model"] == "cnn_vit":
        cfg["cnn"].update({"backbone": "resnet10", "epochs": 2, "batch_size": 2, "freeze_backbone": False})
        cfg["vit"].update({"embed_dim": 32, "depth": 2, "num_heads": 2, "epochs": 2, "batch_size": 2})
    else:
        # Force the from-scratch stem: a wiring check must stay fast and must not
        # depend on a MedicalNet download (the pretrained path has its own test).
        cfg["nnmamba"].update({"spatial_encoder": "scratch",
                               "base": 8, "blocks": 2, "token_grid": 2, "mamba_layers": 1,
                               "epochs": 2, "batch_size": 2, "grad_accum_steps": 1})
    return cfg


def save_snapshot(cfg: dict, dest: Path | str, *, extra: dict | None = None) -> None:
    """Write the resolved config (+ optional env metadata) to ``dest`` as JSON."""
    blob = dict(cfg)
    if extra:
        blob["_run_meta"] = extra
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text(json.dumps(blob, indent=2, default=str))


def get(cfg: dict, *keys: str, default: Any = None) -> Any:
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node
