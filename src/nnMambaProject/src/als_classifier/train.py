"""Train nnMamba on the ALS classification task.

Run from project root:
    python src/als_classifier/train.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from als_classifier.dataset import ALSDataset, list_subject_folders
from als_classifier.model import build_model
from als_classifier.split import split_by_subject


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, loss_fn, opt=None, scaler=None, device="cuda") -> dict:
    train = opt is not None
    model.train(train)

    losses, ys, probs = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(dtype=torch.float16):
                logits = model(x)
                loss = loss_fn(logits, y)

            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            losses.append(loss.item())
            ys.extend(y.detach().cpu().tolist())
            probs.extend(
                torch.softmax(logits, dim=1)[:, 1].detach().float().cpu().tolist()
            )

    preds = [1 if p > 0.5 else 0 for p in probs]
    auc = roc_auc_score(ys, probs) if len(set(ys)) > 1 else float("nan")
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "acc": accuracy_score(ys, preds),
        "f1": f1_score(ys, preds, zero_division=0),
        "auc": auc,
        "n": len(ys),
    }


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    torch.manual_seed(cfg["split"]["seed"])

    ckpt_dir = ROOT / cfg["paths"]["ckpt_dir"]
    log_dir = ROOT / cfg["paths"]["log_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    folders = list_subject_folders(REPO_ROOT / cfg["data"]["root"])
    train_f, val_f, test_f = split_by_subject(
        folders,
        val_frac=cfg["split"]["val_frac"],
        test_frac=cfg["split"]["test_frac"],
        seed=cfg["split"]["seed"],
    )
    print(f"Splits — train: {len(train_f)}  val: {len(val_f)}  test: {len(test_f)}")

    ds_kwargs = dict(
        target_shape=cfg["data"]["target_shape"],
        target_spacing=cfg["data"]["target_spacing"],
    )
    train_ds = ALSDataset(train_f, train=True, **ds_kwargs)
    val_ds = ALSDataset(val_f, train=False, **ds_kwargs)
    test_ds = ALSDataset(test_f, train=False, **ds_kwargs)

    dl_kwargs = dict(
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_dl = DataLoader(test_ds, shuffle=False, **dl_kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("Training requires CUDA. Run on the lab machine.")

    model = build_model(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        channels=cfg["model"]["channels"],
        blocks=cfg["model"]["blocks"],
    ).to(device)

    opt = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    sched = CosineAnnealingLR(opt, T_max=cfg["training"]["epochs"])
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()

    history: list[dict] = []
    best_auc = -1.0

    for epoch in range(cfg["training"]["epochs"]):
        t0 = time.time()
        tr = run_epoch(model, train_dl, loss_fn, opt, scaler, device)
        va = run_epoch(model, val_dl, loss_fn, None, None, device)
        sched.step()
        dt = time.time() - t0

        log = {
            "epoch": epoch,
            "dt_sec": round(dt, 1),
            "lr": opt.param_groups[0]["lr"],
            "train": tr,
            "val": va,
        }
        history.append(log)
        print(
            f"E{epoch:03d} ({dt:5.1f}s) "
            f"tr_loss={tr['loss']:.3f} tr_auc={tr['auc']:.3f} | "
            f"va_loss={va['loss']:.3f} va_auc={va['auc']:.3f} va_acc={va['acc']:.3f}"
        )

        with open(log_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if va["auc"] > best_auc:
            best_auc = va["auc"]
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  → new best (val AUC {best_auc:.3f})")

    # Final evaluation on the test split using the best checkpoint
    model.load_state_dict(torch.load(ckpt_dir / "best.pt"))
    te = run_epoch(model, test_dl, loss_fn, None, None, device)
    print(f"\nTEST  acc={te['acc']:.3f}  f1={te['f1']:.3f}  auc={te['auc']:.3f}")

    with open(log_dir / "test_metrics.json", "w") as f:
        json.dump(te, f, indent=2)


if __name__ == "__main__":
    main()
