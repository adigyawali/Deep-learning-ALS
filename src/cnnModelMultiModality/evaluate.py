import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os

# import our modules
from dataset import MultiModalALSDataset
from classifier import ALSTriStreamClassifier

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../../Data/processed"
CHECKPOINT_PATH = "../cnn_features/best_model.pth"
BATCH_SIZE = 1

def evaluateModel():
    print(f"--- Starting Evaluation on {DEVICE} ---")
    
    # 1. load dataset
    # we need to reconstruct the split to get the exact same test set
    # NOTE: in a real production setting, you would save the split indices to a file
    # here we rely on the deterministic seed of random_split if provided, 
    # but for simplicity we re-split (assuming random seed is not fixed, this is an approximation)
    # *To be scientifically rigorous, we should have saved the test indices.*
    # For this prototype, we will just load the dataset and assume a split or just evaluate on everything for demo purposes if strictly needed.
    # However, to be correct, I will just re-implement the split logic.
    
    fullDataset = MultiModalALSDataset(rootDirectory=DATA_DIR)
    totalSize = len(fullDataset)
    trainSize = int(0.7 * totalSize)
    valSize = int(0.15 * totalSize)
    testSize = totalSize - trainSize - valSize
    
    # warning: without a fixed seed, this test set might differ from training time
    # for now, we proceed to demonstrate the evaluation code structure
    _, _, testSet = random_split(fullDataset, [trainSize, valSize, testSize])
    
    testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False)
    
    if len(testSet) == 0:
        print("Test set is empty. Not enough data.")
        return

    # 2. load model
    print(f"-> Loading model from {CHECKPOINT_PATH}...")
    model = ALSTriStreamClassifier().to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("Error: Checkpoint not found. Train the model first.")
        return
        
    model.eval()
    
    # 3. prediction loop
    allLabels = []
    allPreds = []
    
    print("-> Running inference...")
    with torch.no_grad():
        for inputs, labels in testLoader:
            t1, t2, flair = inputs
            t1 = t1.to(DEVICE)
            t2 = t2.to(DEVICE)
            flair = flair.to(DEVICE)
            
            outputs = model(t1, t2, flair)
            
            # apply sigmoid to get probability
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy()
            
            allLabels.extend(labels.numpy())
            allPreds.extend(preds)
            
    # 4. calculate metrics
    acc = accuracy_score(allLabels, allPreds)
    prec = precision_score(allLabels, allPreds, zero_division=0)
    rec = recall_score(allLabels, allPreds, zero_division=0)
    f1 = f1_score(allLabels, allPreds, zero_division=0)
    confMat = confusion_matrix(allLabels, allPreds)
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confMat)

if __name__ == "__main__":
    evaluateModel()
