import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from .dataset import ALSFeatureDataset, split_indices_by_subject
    from .model import SimpleMultiModalViT
    from .paths import (
        build_runtime_paths,
        ensure_output_dirs,
    )
except ImportError:
    from dataset import ALSFeatureDataset, split_indices_by_subject
    from model import SimpleMultiModalViT
    from paths import (
        build_runtime_paths,
        ensure_output_dirs,
    )

# Training settings kept intentionally simple.
SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(features_dir: str | None = None, artifacts_dir: str | None = None, batch_size: int = BATCH_SIZE, num_workers: int = 2) -> None:
    ensure_output_dirs()
    set_seed(SEED)
    runtime_features_dir, runtime_artifacts_dir, runtime_checkpoint_dir, runtime_metrics_dir, runtime_checkpoint_path = (
        build_runtime_paths(features_dir=features_dir, artifacts_dir=artifacts_dir)
    )
    runtime_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runtime_metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = ALSFeatureDataset(features_dir=str(runtime_features_dir))
    if len(dataset) < 3:
        print(f"Error: need at least 3 feature files in {runtime_features_dir}")
        return

    train_idx, val_idx, _ = split_indices_by_subject(
        dataset.samples, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED
    )
    if not train_idx or not val_idx:
        print("Error: empty train or validation split after subject-level split.")
        return

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = SimpleMultiModalViT(input_dim=dataset.feature_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    history = []

    print(f"Starting ViT training on {DEVICE}")
    print(f"Features dir: {runtime_features_dir}")
    print(f"Artifacts dir: {runtime_artifacts_dir}")
    print(f"Train samples: {len(train_set)} | Validation samples: {len(val_set)}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for features, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"):
            features = features.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * features.size(0)

        train_loss = train_loss_sum / len(train_set)

        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]"):
                features = features.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)

                logits = model(features)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * features.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()

        val_loss = val_loss_sum / len(val_set)
        val_acc = correct / max(total, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "input_dim": dataset.feature_dim,
                        "embed_dim": model.embed_dim,
                        "num_heads": 4,
                        "num_layers": 2,
                        "mlp_dim": 512,
                        "dropout": 0.1,
                    },
                },
                runtime_checkpoint_path,
            )
            saved_tag = " (saved best)"
        else:
            saved_tag = ""

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        print(
            f"Epoch {epoch + 1:02d}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}{saved_tag}"
        )

    history_path = runtime_metrics_dir / "vit_train_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best checkpoint: {runtime_checkpoint_path}")
    print(f"Training history saved to: {history_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train simple multimodal ViT on CNN 3D features.")
    parser.add_argument("--features-dir", type=str, default=None, help="Path to CNN feature files (*.pt).")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Path to save ViT checkpoints/metrics.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (Colab: 2 is a safe default).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
