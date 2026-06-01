"""
One-shot ALS pipeline driver.

Run the entire pipeline (or a subset) with a single command. Each step is
idempotent — already-done outputs are skipped.

Examples
--------
Full pipeline on the lab box::

    python src/experiment.py all

Just the stages that need the GPU::

    python src/experiment.py train_cnn extract_features train_vit evaluate_vit

Per stage::

    python src/experiment.py preprocess
    python src/experiment.py train_cnn
    python src/experiment.py extract_features
    python src/experiment.py train_vit
    python src/experiment.py evaluate_vit
    python src/experiment.py gradcam
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent       # .../src
PROJECT_ROOT = ROOT.parent

STAGES = ["preprocess", "train_cnn", "extract_features", "train_vit", "evaluate_vit", "gradcam"]


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    return subprocess.call([str(c) for c in cmd], cwd=cwd)


def run_preprocess(args: argparse.Namespace) -> int:
    cmd = [sys.executable, ROOT / "preprocessing" / "preprocessing.py"]
    if args.nonlinear:
        cmd.append("--nonlinear")
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    return _run(cmd)


def run_train_cnn(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, ROOT / "cnnModelMultiModality" / "train.py",
        "--epochs", str(args.cnn_epochs),
        "--batch-size", str(args.cnn_batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
    ]
    if args.freeze_backbone:
        cmd.append("--freeze-backbone")
    return _run(cmd)


def run_extract_features(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, ROOT / "cnnModelMultiModality" / "generate_spatial_features.py",
        "--device", args.device,
        "--batch-size", "1",
    ]
    # Require the fine-tuned CNN checkpoint by default: extracting features from
    # an un-fine-tuned (MedicalNet-only / random) backbone silently degrades the
    # ViT. Pass --allow-missing-checkpoint to opt out (debug/ablation only).
    if not args.allow_missing_checkpoint:
        cmd.append("--require-checkpoint")
    return _run(cmd)


def run_train_vit(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, ROOT / "ViTModel" / "train_vit.py",
        "--epochs", str(args.vit_epochs),
        "--batch-size", str(args.vit_batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--best-metric", args.best_metric,
    ]
    return _run(cmd)


def run_evaluate_vit(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, ROOT / "ViTModel" / "evaluate_vit.py",
        "--device", args.device,
        "--bootstrap-n", str(args.bootstrap_n),
    ]
    return _run(cmd)


def run_gradcam(args: argparse.Namespace) -> int:
    cmd = [sys.executable, ROOT / "cnnModelMultiModality" / "gradcam.py"]
    if args.subject:
        cmd += ["--subject", args.subject]
    return _run(cmd)


DISPATCH = {
    "preprocess": run_preprocess,
    "train_cnn": run_train_cnn,
    "extract_features": run_extract_features,
    "train_vit": run_train_vit,
    "evaluate_vit": run_evaluate_vit,
    "gradcam": run_gradcam,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end ALS pipeline driver.")
    p.add_argument("stages", nargs="+", help="Stage names (or 'all'). Choices: " + ", ".join(STAGES))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--num-workers", type=int, default=2)
    # preprocess
    p.add_argument("--nonlinear", action="store_true", help="Use SyN T1->MNI registration.")
    p.add_argument("--limit", type=int, default=0, help="Limit number of triplets (debug).")
    # CNN
    p.add_argument("--cnn-epochs", type=int, default=60)
    p.add_argument("--cnn-batch-size", type=int, default=4)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--allow-missing-checkpoint", action="store_true",
                   help="Allow feature extraction to proceed without a fine-tuned CNN "
                        "checkpoint (uses MedicalNet/random weights). Default requires it.")
    # ViT
    p.add_argument("--vit-epochs", type=int, default=60)
    p.add_argument("--vit-batch-size", type=int, default=8)
    p.add_argument("--best-metric", choices=["roc_auc", "pr_auc"], default="roc_auc")
    p.add_argument("--bootstrap-n", type=int, default=2000)
    # gradcam
    p.add_argument("--subject", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stages = STAGES if args.stages == ["all"] else args.stages
    bad = [s for s in stages if s not in DISPATCH]
    if bad:
        print(f"Unknown stage(s): {bad}. Valid: {STAGES}", file=sys.stderr)
        return 2
    for stage in stages:
        print("\n" + "=" * 64)
        print(f"== Stage: {stage}")
        print("=" * 64)
        rc = DISPATCH[stage](args)
        if rc != 0:
            print(f"Stage {stage} failed with rc={rc}. Stopping.", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
