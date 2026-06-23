"""Canonical filesystem layout for data, splits, checkpoints, and results.

Everything a run produces lives under one ``runs/<model>/`` tree so it is easy
to see where outputs, logs, and checkpoints went, and the ``splits.json`` is
shared at ``runs/splits.json`` so both models train/eval on the same subjects.

    Data/processed/                 # one folder per subject-visit (preprocessing output)
    runs/
      splits.json                   # SHARED canonical split (written once)
      cnn_vit/
        checkpoints/                # cnn_{latest,best}.pt, vit_{latest,best}.pt
        features/                   # <id>_spatial.pt  (CNN layer4 maps for the ViT)
        metrics/                    # *_history.json, evaluation.json, predictions.json
        logs/
        config.json                 # resolved config snapshot for this run
      cnn_nnmamba/
        checkpoints/                # {latest,best}.pt
        metrics/  logs/  config.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data" / "processed"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs"

MODELS = ("cnn_vit", "cnn_nnmamba")


@dataclass(frozen=True)
class RunPaths:
    model: str
    root: Path
    checkpoints: Path
    features: Path
    metrics: Path
    logs: Path
    config_json: Path
    splits_path: Path

    def ensure(self) -> "RunPaths":
        for d in (self.root, self.checkpoints, self.features, self.metrics, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        self.splits_path.parent.mkdir(parents=True, exist_ok=True)
        return self


def build_run_paths(model: str, output_root: Path | str | None = None) -> RunPaths:
    """Return the output paths for ``model`` under ``output_root`` (default runs/)."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}. Choices: {MODELS}")
    root_out = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
    run_root = root_out / model
    return RunPaths(
        model=model,
        root=run_root,
        checkpoints=run_root / "checkpoints",
        features=run_root / "features",
        metrics=run_root / "metrics",
        logs=run_root / "logs",
        config_json=run_root / "config.json",
        splits_path=root_out / "splits.json",
    )
