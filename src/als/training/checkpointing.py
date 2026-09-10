"""
Best-weights saving, shared by every training stage.

Exactly one file per stage per fold: ``<prefix>_best.pt``, overwritten only when
validation improves. It holds just what is needed to rebuild and run the best
model — weights, the resolved config, the validation-tuned threshold, and the
best metric.

It deliberately stores no optimizer / scheduler / scaler / RNG state and no epoch
counter, so there is no per-epoch checkpoint and no resume: if a run dies,
restart it. The best file written so far stays valid for evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def save_best_weights(
    ckpt_dir: Path | str,
    prefix: str,
    *,
    model: torch.nn.Module,
    best_metric: float,
    best_metric_name: str,
    threshold: float = 0.5,
    config: Optional[dict] = None,
) -> Path:
    """Overwrite ``<prefix>_best.pt`` with the current best-so-far weights.

    Written atomically (temp file + replace) so a crash mid-save cannot corrupt
    the last good best file.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "best_metric": float(best_metric),
        "best_metric_name": best_metric_name,
        "threshold": float(threshold),
        "config": config or {},
    }
    best = ckpt_dir / f"{prefix}_best.pt"
    tmp = best.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(best)
    return best


def load_best_weights(path: Path | str, map_location="cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)
