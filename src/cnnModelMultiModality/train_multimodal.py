import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from classifier import ALSTriStreamClassifier
from dataset import MultiModalALSDataset
from paths import CHECKPOINT_PATH, DATA_DIR, METRICS_DIR, ensure_output_dirs
from split_utils import split_indices_by_subject

# Training settings
SEED = 42
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed: int) -> None:
    """Make splits and training runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train() -> None:
    ensure_output_dirs()
    set_seed(SEED)

    if not DATA_DIR.exists():
        print(f"Error: data directory not found: {DATA_DIR}")
        return

    base_dataset = MultiModalALSDataset(rootDirectory=str(DATA_DIR), transform=False)
    if len(base_dataset) < 2:
        print("Error: not enough samples to run train/validation split.")
        return

    train_indices, val_indices, _ = split_indices_by_subject(
        base_dataset.samples, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED
    )
    if not train_indices or not val_indices:
        print("Error: subject-level split produced an empty train or validation set.")
        return

    # Build two dataset views so only training gets augmentation.
    train_dataset = MultiModalALSDataset(rootDirectory=str(DATA_DIR), transform=True)
    val_dataset = MultiModalALSDataset(rootDirectory=str(DATA_DIR), transform=False)
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = ALSTriStreamClassifier().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    use_amp = DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    history = []

    print(f"Starting training on {DEVICE}.")
    print(f"Train samples: {len(train_set)} | Validation samples: {len(val_set)}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for (t1, t2, flair), label in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"):
            t1, t2, flair = t1.to(DEVICE), t2.to(DEVICE), flair.to(DEVICE)
            label = label.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(t1, t2, flair)
                loss = criterion(logits, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * t1.size(0)

        train_loss = train_loss_sum / len(train_set)

        # Validation loop mirrors training without gradients.
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for (t1, t2, flair), label in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]"):
                t1, t2, flair = t1.to(DEVICE), t2.to(DEVICE), flair.to(DEVICE)
                label = label.to(DEVICE).unsqueeze(1)

                logits = model(t1, t2, flair)
                loss = criterion(logits, label)
                val_loss_sum += loss.item() * t1.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == label).sum().item()
                total += label.numel()

        val_loss = val_loss_sum / len(val_set)
        val_acc = correct / max(1, total)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            best_tag = " (saved best)"
        else:
            best_tag = ""

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch + 1:02d}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}{best_tag}"
        )

    metrics_path = METRICS_DIR / "train_history.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training finished. Best checkpoint: {CHECKPOINT_PATH}")
    print(f"Training history saved to: {metrics_path}")


if __name__ == "__main__":
    train()
