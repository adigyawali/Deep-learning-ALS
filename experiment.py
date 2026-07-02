"""
Unified ALS pipeline driver.

Pick a model with ``--model`` and run the whole pipeline (or named stages):

    python experiment.py --model cnn_vit                 # full CNN→ViT pipeline
    python experiment.py --model cnn_nnmamba             # full CNN→nnMamba pipeline
    python experiment.py --model cnn_nnmamba --smoke     # tiny end-to-end wiring check
    python experiment.py --model cnn_vit train_vit evaluate   # just these stages

Stages:
  cnn_vit     : preprocess train_cnn extract_features train_vit evaluate gradcam
  cnn_nnmamba : preprocess train_nnmamba evaluate

``all`` (the default) runs every stage for the chosen model except gradcam.
Every stage is idempotent where outputs already exist. Training saves only the
best-validation weights (``runs/<model>/checkpoints/<prefix>_best.pt``); there
is no per-epoch checkpoint and no resume, so a crashed run must be restarted.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Make ``als`` importable when run straight from a checkout (no install needed),
# in addition to the editable install documented in Instructions.md.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from als import config as cfgmod  # noqa: E402
from als.paths import build_run_paths  # noqa: E402
from als.seed import resolve_device, set_seed  # noqa: E402

STAGE_ORDER = {
    "cnn_vit": ["preprocess", "train_cnn", "extract_features", "train_vit", "evaluate"],
    "cnn_nnmamba": ["preprocess", "train_nnmamba", "evaluate"],
}
ALL_STAGES = {
    "cnn_vit": STAGE_ORDER["cnn_vit"] + ["gradcam"],
    "cnn_nnmamba": STAGE_ORDER["cnn_nnmamba"],
}


def _run_preprocess(args) -> int:
    # Preprocessing is a standalone subprocess (heavy ANTs work, no GPU).
    from als.data import preprocessing
    argv: list[str] = []
    if args.nonlinear:
        argv.append("--nonlinear")
    if args.limit > 0:
        argv += ["--limit", str(args.limit)]
    return preprocessing.main(argv)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified ALS pipeline driver.")
    p.add_argument("--model", required=True, choices=["cnn_vit", "cnn_nnmamba"])
    p.add_argument("stages", nargs="*", default=["all"],
                   help="Stage names or 'all' (default). See module docstring.")
    p.add_argument("--config", type=str, default=None, help="Override configs/<model>.yaml.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--smoke", action="store_true", help="Tiny-subset wiring check (fast, no GPU needed).")
    p.add_argument("--output-dir", type=str, default=None, help="Override runs/ root.")
    p.add_argument("--data-dir", type=str, default=None, help="Override Data/processed.")
    # common overrides
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    # preprocess passthrough
    p.add_argument("--nonlinear", action="store_true", help="SyN T1→MNI registration (preprocess).")
    p.add_argument("--limit", type=int, default=0, help="Limit preprocessed triplets (debug).")
    # extract / gradcam
    p.add_argument("--allow-missing-checkpoint", action="store_true",
                   help="Allow feature extraction without a fine-tuned CNN checkpoint (ablation).")
    p.add_argument("--subject", type=str, default=None, help="Grad-CAM subject (default: random test subject).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model = args.model

    requested = args.stages if args.stages and args.stages != ["all"] else None
    stages = requested if requested else ALL_STAGES[model]
    valid = set(STAGE_ORDER[model] + ["gradcam"])
    bad = [s for s in stages if s not in valid]
    if bad:
        print(f"Unknown stage(s) for {model}: {bad}. Valid: {sorted(valid)}", file=sys.stderr)
        return 2

    cfg = cfgmod.load_config(model, args.config)
    cfg = cfgmod.apply_overrides(cfg, batch_size=args.batch_size, epochs=args.epochs,
                                 lr=args.lr, num_workers=args.num_workers)
    if args.data_dir:
        cfg.setdefault("data", {})["data_dir"] = args.data_dir
    if args.smoke:
        cfg = cfgmod.apply_smoke(cfg)
        # Smoke must never block on a network weight download.
        os.environ.setdefault("ALS_SKIP_PRETRAINED", "1")
        # Preprocessing needs raw data + ANTs; not part of a wiring check.
        stages = [s for s in stages if s != "preprocess"]

    set_seed(cfg.get("seed", 42))
    device = resolve_device(args.device)
    paths = build_run_paths(model, args.output_dir).ensure()

    from als.models.components.mamba_block import MAMBA_BACKEND
    git = ""
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=Path(__file__).resolve().parent, text=True).strip()
    except Exception:
        pass
    cfgmod.save_snapshot(cfg, paths.config_json, extra={
        "device": str(device), "git_commit": git, "mamba_backend": MAMBA_BACKEND,
        "argv": sys.argv[1:], "smoke": bool(args.smoke),
    })

    print("=" * 70)
    print(f"model={model}  device={device}  stages={stages}  smoke={args.smoke}")
    print(f"outputs → {paths.root}")
    print("=" * 70)

    for stage in stages:
        print(f"\n{'=' * 70}\n== Stage: {stage}\n{'=' * 70}")
        if stage == "preprocess":
            rc = _run_preprocess(args)
            if rc not in (0, None):
                print(f"Stage preprocess failed (rc={rc}). Stopping.", file=sys.stderr)
                return rc
            continue
        if stage == "train_cnn":
            from als.stages import train_cnn
            train_cnn.run(cfg, paths, device)
        elif stage == "extract_features":
            from als.stages import extract_features
            extract_features.run(cfg, paths, device,
                                 allow_missing_checkpoint=args.allow_missing_checkpoint)
        elif stage == "train_vit":
            from als.stages import train_vit
            train_vit.run(cfg, paths, device)
        elif stage == "train_nnmamba":
            from als.stages import train_nnmamba
            train_nnmamba.run(cfg, paths, device)
        elif stage == "evaluate":
            from als.stages import evaluate
            evaluate.run(cfg, paths, device)
        elif stage == "gradcam":
            from als.stages import gradcam
            gradcam.run(cfg, paths, device, subject=args.subject)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
