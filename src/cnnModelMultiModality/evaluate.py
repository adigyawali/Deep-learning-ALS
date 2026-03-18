"""
evaluate.py

Evaluates the trained ALSTriStreamClassifier on the held-out test split.

Ensures the sigmoid + threshold logic is consistent with the BCEWithLogitsLoss
contract used in training (single logit output, no softmax).
"""

import json

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset

from classifier import ALSTriStreamClassifier
from dataset import MultiModalALSDataset
from paths import CHECKPOINT_PATH, DATA_DIR, METRICS_DIR, ensure_output_dirs
from split_utils import split_indices_by_subject

DEVICE     = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
BATCH_SIZE = 1
SEED       = 42


def evaluateModel() -> None:
    ensure_output_dirs()
    print(f"--- Starting Evaluation on {DEVICE} ---")

    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    # Rebuild the full dataset (no augmentation for evaluation)
    fullDataset = MultiModalALSDataset(rootDirectory=str(DATA_DIR), transform=False)
    _, _, test_indices = split_indices_by_subject(
        fullDataset.samples, train_ratio=0.8, val_ratio=0.1, seed=SEED
    )

    if not test_indices:
        print("Test set is empty.  Add more data before running evaluation.")
        return

    testSet    = Subset(fullDataset, test_indices)
    testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False)

    # Load checkpoint
    print(f"-> Loading model from {CHECKPOINT_PATH}...")
    model = ALSTriStreamClassifier().to(DEVICE)

    if not CHECKPOINT_PATH.exists():
        print("Error: Checkpoint not found.  Train the model first.")
        return

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # Inference
    allLabels = []
    allProbs  = []

    print("-> Running inference...")
    with torch.no_grad():
        for inputs, labels in testLoader:
            t1, t2, flair = [v.to(DEVICE) for v in inputs]

            logits = model(t1, t2, flair).squeeze(1)  # (B,)
            probs  = torch.sigmoid(logits)             # (B,) ∈ [0, 1]

            allProbs.extend(probs.cpu().tolist())
            allLabels.extend(labels.cpu().tolist())

    # Convert probabilities → binary predictions at threshold 0.5
    allPreds = [1 if p >= 0.5 else 0 for p in allProbs]
    intLabels = [int(l) for l in allLabels]

    acc     = accuracy_score(intLabels, allPreds)
    prec    = precision_score(intLabels, allPreds, zero_division=0)
    rec     = recall_score(intLabels, allPreds, zero_division=0)
    f1      = f1_score(intLabels, allPreds, zero_division=0)
    confMat = confusion_matrix(intLabels, allPreds)

    # AUC requires at least two classes in the test set
    try:
        auc = roc_auc_score(intLabels, allProbs)
    except ValueError:
        auc = float("nan")

    metrics = {
        "accuracy":          acc,
        "precision":         prec,
        "recall":            rec,
        "f1_score":          f1,
        "roc_auc":           auc,
        "confusion_matrix":  confMat.tolist(),
        "num_test_samples":  len(testSet),
    }

    metrics_path = METRICS_DIR / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(f"  TN={confMat[0,0]}  FP={confMat[0,1]}")
    print(f"  FN={confMat[1,0]}  TP={confMat[1,1]}")
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    evaluateModel()