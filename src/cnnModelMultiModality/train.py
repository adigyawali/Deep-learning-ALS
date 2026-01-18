import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import time
import copy

# import our custom modules
from dataset import MultiModalALSDataset
from classifier import ALSTriStreamClassifier

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# set device (mps for mac, cuda for nvidia, cpu otherwise)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2 # small batch size due to large 3d volumes
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
DATA_DIR = "../../Data/processed" # relative to this script
CHECKPOINT_DIR = "./checkpoints"

# create checkpoint directory if it does not exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Training Routine
# ---------------------------------------------------------
def trainModel():
    print(f"--- Starting Training on {DEVICE} ---")
    
    # 1. prepare dataset
    print("-> Loading dataset...")
    fullDataset = MultiModalALSDataset(rootDirectory=DATA_DIR)
    
    totalSize = len(fullDataset)
    if totalSize == 0:
        print("Error: No processed data found. Run preprocessing first.")
        return
        
    print(f"-> Found {totalSize} samples.")
    
    # split into train (70%), validation (15%), test (15%)
    trainSize = int(0.7 * totalSize)
    valSize = int(0.15 * totalSize)
    testSize = totalSize - trainSize - valSize
    
    trainSet, valSet, testSet = random_split(fullDataset, [trainSize, valSize, testSize])
    
    # create dataloaders
    trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)
    valLoader = DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. initialize model
    print("-> Initializing Tri-Stream Network...")
    model = ALSTriStreamClassifier().to(DEVICE)
    
    # 3. define loss and optimizer
    # bce with logits includes sigmoid for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # variables to track best performance
    bestValAcc = 0.0
    bestModelWeights = copy.deepcopy(model.state_dict())
    
    # 4. training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 10)
        
        # --- training phase ---
        model.train()
        runningLoss = 0.0
        runningCorrects = 0
        
        for batchIdx, (inputs, labels) in enumerate(trainLoader):
            t1, t2, flair = inputs
            
            # move data to device
            t1 = t1.to(DEVICE)
            t2 = t2.to(DEVICE)
            flair = flair.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1) # match shape (Batch, 1)
            
            # zero gradients
            optimizer.zero_grad()
            
            # forward
            outputs = model(t1, t2, flair)
            loss = criterion(outputs, labels)
            
            # backward
            loss.backward()
            optimizer.step()
            
            # statistics
            runningLoss += loss.item() * t1.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            runningCorrects += torch.sum(preds == labels.data)
            
        epochLoss = runningLoss / len(trainSet)
        epochAcc = runningCorrects.double() / len(trainSet)
        
        print(f"Train Loss: {epochLoss:.4f} Acc: {epochAcc:.4f}")
        
        # --- validation phase ---
        model.eval()
        valLoss = 0.0
        valCorrects = 0
        
        with torch.no_grad():
            for inputs, labels in valLoader:
                t1, t2, flair = inputs
                t1 = t1.to(DEVICE)
                t2 = t2.to(DEVICE)
                flair = flair.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1)
                
                outputs = model(t1, t2, flair)
                loss = criterion(outputs, labels)
                
                valLoss += loss.item() * t1.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                valCorrects += torch.sum(preds == labels.data)
                
        valLoss = valLoss / len(valSet)
        valAcc = valCorrects.double() / len(valSet)
        
        print(f"Val Loss: {valLoss:.4f} Acc: {valAcc:.4f}")
        
        # deep copy the model if it's the best one so far
        if valAcc > bestValAcc:
            bestValAcc = valAcc
            bestModelWeights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print("-> Found new best model. Saved.")
            
    print("\n--- Training Complete ---")
    print(f"Best Val Acc: {bestValAcc:.4f}")

if __name__ == "__main__":
    trainModel()
