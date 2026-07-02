"""Best-weights saving shared by every training stage.

Exactly ONE file is written per stage, into ``runs/<model>/checkpoints/``:

  * ``<prefix>_best.pt`` — the best-validation snapshot so far, overwritten only
    when validation improves. It is what evaluation / feature extraction /
    Grad-CAM / inference load.

This is deliberately NOT a full training checkpoint. It stores only what is
needed to rebuild and run the best model: the model weights, the resolved config
(so the eval side can reconstruct the architecture), the validation-tuned
decision threshold, and the best validation metric (name + value) for the
report. It intentionally does NOT store optimizer / scheduler / AMP-scaler / RNG
state or an epoch counter, so there is no per-epoch checkpoint and no
resume-after-crash: if a run dies, restart it. The best weights saved so far
stay usable for evaluation.
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
