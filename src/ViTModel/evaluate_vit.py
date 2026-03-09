import json
import argparse

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset

try:
    from .dataset import ALSFeatureDataset, split_indices_by_subject
    from .model import SimpleMultiModalViT
    from .paths import build_runtime_paths, ensure_output_dirs
except ImportError:
    from dataset import ALSFeatureDataset, split_indices_by_subject
    from model import SimpleMultiModalViT
    from paths import build_runtime_paths, ensure_output_dirs

SEED = 42
BATCH_SIZE = 8
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def evaluate(features_dir: str | None = None, artifacts_dir: str | None = None, batch_size: int = BATCH_SIZE, num_workers: int = 2) -> None:
    ensure_output_dirs()
    runtime_features_dir, runtime_artifacts_dir, runtime_checkpoint_dir, runtime_metrics_dir, runtime_checkpoint_path = (
        build_runtime_paths(features_dir=features_dir, artifacts_dir=artifacts_dir)
    )
    runtime_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runtime_metrics_dir.mkdir(parents=True, exist_ok=True)

    if not runtime_checkpoint_path.exists():
        print(f"Error: checkpoint not found: {runtime_checkpoint_path}")
        print("Run train_vit.py first.")
        return

    dataset = ALSFeatureDataset(features_dir=str(runtime_features_dir))
    if len(dataset) == 0:
        print(f"Error: no feature files found in {runtime_features_dir}")
        return

    _, _, test_idx = split_indices_by_subject(dataset.samples, train_ratio=0.8, val_ratio=0.1, seed=SEED)
    if not test_idx:
        print("Error: empty test split. Add more subjects.")
        return

    test_set = Subset(dataset, test_idx)
    pin_memory = DEVICE.type == "cuda"
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    checkpoint = torch.load(runtime_checkpoint_path, map_location=DEVICE)
    config = checkpoint["config"]
    model = SimpleMultiModalViT(
        input_dim=config["input_dim"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        mlp_dim=config["mlp_dim"],
        dropout=config["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for features, labels, _ in test_loader:
            features = features.to(DEVICE)
            logits = model(features).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().numpy().reshape(-1)

            all_labels.extend(labels.long().numpy().reshape(-1).tolist())
            all_preds.extend(preds.tolist())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "num_test_samples": len(test_set),
    }

    metrics_path = runtime_metrics_dir / "vit_evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("--- ViT Evaluation ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"Saved metrics JSON to: {metrics_path}")
    print(f"Artifacts dir: {runtime_artifacts_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate simple multimodal ViT on CNN 3D features.")
    parser.add_argument("--features-dir", type=str, default=None, help="Path to CNN feature files (*.pt).")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Path where ViT checkpoint/metrics are stored.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (Colab: 2 is a safe default).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
