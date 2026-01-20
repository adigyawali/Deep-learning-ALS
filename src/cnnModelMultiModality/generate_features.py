import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure we can import local modules
sys.path.append(os.getcwd())

from dataset import MultiModalALSDataset
from classifier import ALSTriStreamClassifier

# Configuration
DATA_DIR = "../../Data/processed"
# Updated to match train.py output
CHECKPOINT_PATH = "../cnn_features/best_model.pth" 
OUTPUT_DIR = "../cnn_features"
BATCH_SIZE = 2
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_features():
    print(f"--- Starting Feature Extraction on {DEVICE} ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found.")
        return
        
    dataset = MultiModalALSDataset(rootDirectory=DATA_DIR)
    # Using a larger batch size for inference if possible, but keeping consistency
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Found {len(dataset)} samples.")
    
    # 2. Load Model
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint {CHECKPOINT_PATH} not found. Please run train.py first.")
        return

    print("-> Loading model architecture...")
    model = ALSTriStreamClassifier().to(DEVICE)
    
    print(f"-> Loading weights from {CHECKPOINT_PATH}...")
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Tip: Ensure train.py was run with the same model architecture.")
        return

    model.eval()
    print("Model weights loaded successfully.")
    
    # 3. Extract Features
    print(f"Extracting features to {OUTPUT_DIR}...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            t1, t2, flair = inputs
            t1, t2, flair = t1.to(DEVICE), t2.to(DEVICE), flair.to(DEVICE)
            
            # Access encoders inside the wrapper
            # ALSTriStreamClassifier -> CascadedMixingTransformer (self.model) -> Encoders
            t1_encoder = model.model.t1Encoder
            t2_encoder = model.model.t2Encoder
            flair_encoder = model.model.flairEncoder
            
            # Get embeddings individually
            feat_t1 = t1_encoder(t1)
            feat_t2 = t2_encoder(t2)
            feat_flair = flair_encoder(flair)
            
            # Save individually per subject
            for i in range(t1.size(0)):
                global_idx = batch_idx * BATCH_SIZE + i
                if global_idx >= len(dataset): break
                
                sample_id = dataset.samples[global_idx]['id']
                label_val = dataset.samples[global_idx]['label']
                
                save_path = os.path.join(OUTPUT_DIR, f"{sample_id}.pt")
                torch.save({
                    't1_feat': feat_t1[i].cpu(),
                    't2_feat': feat_t2[i].cpu(),
                    'flair_feat': feat_flair[i].cpu(),
                    'label': label_val
                }, save_path)
                
    print("--- Feature Extraction Complete ---")

if __name__ == "__main__":
    generate_features()
