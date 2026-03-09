import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

from dataset import MultiModalALSDataset
from classifier import ALSTriStreamClassifier
from paths import CHECKPOINT_PATH, DATA_DIR, METRICS_DIR, ensure_output_dirs
from split_utils import split_indices_by_subject

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
SEED = 42

def evaluateModel():
    ensure_output_dirs()
    print(f"--- Starting Evaluation on {DEVICE} ---")

    # Rebuild the same dataset then split into train/val/test for holdout testing.
    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    fullDataset = MultiModalALSDataset(rootDirectory=str(DATA_DIR))
    _, _, test_indices = split_indices_by_subject(fullDataset.samples, train_ratio=0.8, val_ratio=0.1, seed=SEED)
    if not test_indices:
        print("Test set is empty. Add more data before running evaluation.")
        return

    testSet = Subset(fullDataset, test_indices)
    testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False)

    # Load checkpoint from the shared artifacts folder.
    print(f"-> Loading model from {CHECKPOINT_PATH}...")
    model = ALSTriStreamClassifier().to(DEVICE)

    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("Error: Checkpoint not found. Train the model first.")
        return

    model.eval()

    # Run predictions on the holdout split.
    allLabels = []
    allPreds = []

    print("-> Running inference...")
    with torch.no_grad():
        for inputs, labels in testLoader:
            t1, t2, flair = inputs
            t1 = t1.to(DEVICE)
            t2 = t2.to(DEVICE)
            flair = flair.to(DEVICE)
            
            outputs = model(t1, t2, flair).squeeze(1)

            # Convert logits to binary predictions with threshold 0.5.
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long().cpu().numpy().reshape(-1)

            allLabels.extend(labels.long().numpy().reshape(-1))
            allPreds.extend(preds.tolist())

    # Compute core binary-classification metrics.
    acc = accuracy_score(allLabels, allPreds)
    prec = precision_score(allLabels, allPreds, zero_division=0)
    rec = recall_score(allLabels, allPreds, zero_division=0)
    f1 = f1_score(allLabels, allPreds, zero_division=0)
    confMat = confusion_matrix(allLabels, allPreds)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": confMat.tolist(),
        "num_test_samples": len(testSet),
    }
    metrics_path = METRICS_DIR / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confMat)
    print(f"\nSaved metrics JSON to: {metrics_path}")

if __name__ == "__main__":
    evaluateModel()
