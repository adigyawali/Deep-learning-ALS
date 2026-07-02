"""Model-agnostic training loop shared by the ViT and nnMamba stages.

The loop knows nothing about a specific model's input shape. Each stage passes a
``forward_fn(model, batch, device) -> (logits, labels)`` adapter that moves its
own batch to the device and returns ``(B, 1)`` logits + ``(B, 1)`` float labels.
That single hook lets the tri-stream CNN (3 volume tensors), the ViT (one
stacked feature tensor), and end-to-end nnMamba (one volume tensor) all reuse:

  * AMP (bf16 default on CUDA; fp16 path uses a GradScaler),
  * gradient accumulation (raise it to keep the effective batch size while the
    per-step memory drops — the main OOM lever besides batch size),
  * a CUDA-OOM guard that prints concrete remedies instead of a bare traceback,
  * per-epoch GPU/host-RAM reporting,
  * best-weights saving (only when validation improves) and early stopping.

There is deliberately no per-epoch checkpoint and no resume: only
``<prefix>_best.pt`` is written, and only on a validation improvement.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import torch
from tqdm import tqdm

from .. import gpu
from . import metrics as M
from .checkpointing import save_best_weights

ForwardFn = Callable[[torch.nn.Module, object, torch.device], tuple[torch.Tensor, torch.Tensor]]

_OOM_HELP = (
    "CUDA out of memory.\n"
    "  Try, in rough order of effectiveness:\n"
    "   * lower batch_size (and raise grad_accum_steps to keep the effective batch)\n"
    "   * raise grad_accum_steps alone\n"
    "   * use a smaller cnn_backbone (resnet18/resnet34) or smaller target_shape\n"
    "   * reduce num_workers / prefetch_factor (frees host RAM, not VRAM)\n"
    "   * keep amp_dtype=bf16 (the default on CUDA)\n"
)


def _select_value(m: dict, name: str) -> float:
    return m.get(name, float("nan"))


def fit(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    forward_fn: ForwardFn,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    device: torch.device,
    epochs: int,
    ckpt_dir: Path | str,
    ckpt_prefix: str,
    config: Optional[dict] = None,
    amp_dtype: Optional[torch.dtype] = None,
    grad_accum_steps: int = 1,
    clip_grad: float = 1.0,
    best_metric_name: str = "roc_auc",
    early_stop_patience: int = 15,
    history_path: Optional[Path] = None,
) -> dict:
    grad_accum_steps = max(1, int(grad_accum_steps))
    use_amp = amp_dtype is not None and device.type == "cuda"
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_metric = -float("inf")

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[trainer] params total={n_params:,} trainable={n_train:,} "
          f"amp={'off' if not use_amp else amp_dtype} grad_accum={grad_accum_steps} "
          f"select={best_metric_name}")
    print(f"[trainer] {gpu.device_report(device)}")

    history: list[dict] = []
    epochs_no_improve = 0
    best_threshold = 0.5

    for epoch in range(epochs):
        t0 = time.time()
        gpu.reset_peak(device)

        # ── train ─────────────────────────────────────────────────────────
        model.train()
        running_loss, seen = 0.0, 0
        tr_labels: list[float] = []
        tr_probs: list[float] = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"E{epoch + 1:02d}/{epochs} train",
                    leave=False, dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            try:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits, labels = forward_fn(model, batch, device)
                    loss = criterion(logits, labels)
                loss_to_back = loss / grad_accum_steps
                scaler.scale(loss_to_back).backward()

                if (step + 1) % grad_accum_steps == 0:
                    if clip_grad and clip_grad > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"\n[trainer] {_OOM_HELP}  (peak {gpu.step_report(device)})")
                raise

            bs = labels.size(0)
            running_loss += float(loss.detach()) * bs
            seen += bs
            pbar.set_postfix(loss=f"{running_loss / max(1, seen):.4f}")
            with torch.no_grad():
                p = torch.sigmoid(logits.detach().float()).reshape(-1).cpu().tolist()
            tr_probs.extend(p)
            tr_labels.extend(labels.detach().reshape(-1).cpu().tolist())

        # Flush a trailing partial accumulation window.
        if len(train_loader) % grad_accum_steps != 0:
            if clip_grad and clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, seen)
        train_auc = M.safe_auc(tr_labels, tr_probs)

        # ── validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss_sum, val_seen = 0.0, 0
        va_labels: list[float] = []
        va_probs: list[float] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"E{epoch + 1:02d}/{epochs} val",
                              leave=False, dynamic_ncols=True):
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits, labels = forward_fn(model, batch, device)
                    loss = criterion(logits, labels)
                val_loss_sum += float(loss) * labels.size(0)
                val_seen += labels.size(0)
                va_probs.extend(torch.sigmoid(logits.float()).reshape(-1).cpu().tolist())
                va_labels.extend(labels.reshape(-1).cpu().tolist())

        if scheduler is not None:
            scheduler.step()

        val_loss = val_loss_sum / max(1, val_seen)
        threshold = M.youden_threshold(va_labels, va_probs)
        val_metrics = M.binary_metrics(va_labels, va_probs, threshold=threshold)
        select_value = _select_value(val_metrics, best_metric_name)
        current_lr = optimizer.param_groups[0]["lr"]

        improved = (select_value == select_value) and select_value > best_metric  # NaN-safe
        tag = ""
        if improved:
            best_metric = select_value
            best_threshold = threshold
            epochs_no_improve = 0
            tag = " *best"
            # Save ONLY the best weights so far — no per-epoch/latest checkpoint,
            # no optimizer/scheduler/scaler/RNG state, so there is no resume.
            save_best_weights(
                ckpt_dir, ckpt_prefix,
                model=model, best_metric=best_metric, best_metric_name=best_metric_name,
                threshold=best_threshold, config=config,
            )
        else:
            epochs_no_improve += 1

        dt = time.time() - t0
        record = {
            "epoch": epoch + 1, "lr": current_lr,
            "train_loss": train_loss, "train_auc": train_auc,
            "val_loss": val_loss, "val_threshold": threshold,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
            "seconds": round(dt, 2),
        }
        history.append(record)
        if history_path is not None:
            import json
            Path(history_path).write_text(json.dumps(history, indent=2))

        print(
            f"E{epoch + 1:02d}/{epochs} lr={current_lr:.2e} "
            f"tr_loss={train_loss:.4f} tr_auc={train_auc:.3f} | "
            f"va_loss={val_loss:.4f} va_auc={val_metrics['roc_auc']:.3f} "
            f"va_bal_acc={val_metrics['balanced_accuracy']:.3f} "
            f"va_f1={val_metrics['f1_score']:.3f} thr={threshold:.2f} "
            f"({dt:.1f}s) [{gpu.step_report(device)}]{tag}"
        )

        if epochs_no_improve >= early_stop_patience:
            print(f"[trainer] early stop @ epoch {epoch + 1}: "
                  f"no {best_metric_name} improvement for {early_stop_patience} epochs.")
            break

    print(f"[trainer] done. best {best_metric_name}={best_metric:.4f} "
          f"@thr={best_threshold:.2f}")
    return {
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "best_threshold": best_threshold,
        "history": history,
    }
