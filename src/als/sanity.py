"""Pre-training sanity checks and a preflight summary (Goal 7).

Before any real training the stages call ``preflight`` to print, in one place:
dataset size and ALS/non-ALS balance, the train/val/test split counts, the
device + GPU, the model's parameter count, the shape of one real batch and a
NaN/inf check on it, and whether a resume checkpoint was found. If the first
batch is malformed you find out in seconds instead of mid-epoch.
"""

from __future__ import annotations

from pathlib import Path

import torch

from . import gpu


def count_report(samples) -> dict:
    n = len(samples)
    labels = [float(s["label"]) if isinstance(s, dict) else float(s.label) for s in samples]
    n_pos = sum(1 for l in labels if l == 1.0)
    n_neg = n - n_pos
    ratio = (n_neg / n_pos) if n_pos else float("inf")
    print(f"[sanity] samples={n}  ALS(patient)={n_pos}  non-ALS(control)={n_neg}  "
          f"neg/pos={ratio:.2f}")
    if n_pos == 0 or n_neg == 0:
        print("[sanity] WARNING: only one class present — metrics will be degenerate.")
    return {"n": n, "patient": n_pos, "control": n_neg, "neg_over_pos": ratio}


def split_report(splits: dict) -> None:
    cc = splits.get("class_counts", {})
    for kind in ("train", "val", "test"):
        subj = len(splits.get(f"{kind}_subjects", []))
        c = cc.get(kind, {})
        print(f"[sanity] {kind:5s}: subjects={subj}  controls={c.get('control', '?')}  "
              f"patients={c.get('patient', '?')}")


def param_count(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[sanity] model params: total={total:,}  trainable={trainable:,}")
    return total, trainable


def check_one_batch(loader, forward_fn, model, device) -> None:
    """Pull a single batch, report its input shape, and flag NaN/inf early."""
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("[sanity] WARNING: train loader is empty.")
        return
    # Locate the input tensor(s) for a shape/NaN report (batch layouts differ per model).
    def _first_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (tuple, list)):
            for el in obj:
                t = _first_tensor(el)
                if t is not None:
                    return t
        return None
    x = _first_tensor(batch)
    if x is not None:
        finite = bool(torch.isfinite(x).all())
        print(f"[sanity] first batch input shape={tuple(x.shape)} dtype={x.dtype} "
              f"finite={finite}")
        if not finite:
            print("[sanity] WARNING: non-finite values in the input batch (check preprocessing).")
    model.eval()
    with torch.no_grad():
        logits, labels = forward_fn(model, batch, device)
    print(f"[sanity] forward OK: logits={tuple(logits.shape)} labels={tuple(labels.shape)} "
          f"logits_finite={bool(torch.isfinite(logits).all())}")


def resume_status(ckpt_dir: Path | str, prefix: str) -> bool:
    latest = Path(ckpt_dir) / f"{prefix}_latest.pt"
    found = latest.exists()
    print(f"[sanity] resume checkpoint {'FOUND' if found else 'not found'}: {latest}")
    return found


def preflight(*, stage: str, model, dataset, splits, train_loader, forward_fn,
              device, ckpt_dir, ckpt_prefix) -> None:
    print(f"\n[sanity] ===== preflight: {stage} =====")
    print(f"[sanity] {gpu.device_report(device)}")
    count_report(getattr(dataset, "samples", []))
    if splits:
        split_report(splits)
    param_count(model)
    resume_status(ckpt_dir, ckpt_prefix)
    check_one_batch(train_loader, forward_fn, model, device)
    print("[sanity] ===== preflight done =====\n")
