import torch
from torch.utils.data import DataLoader

from dataset import MultiModalALSDataset
from classifier import ALSTriStreamClassifier
from paths import CHECKPOINT_PATH, DATA_DIR, FEATURES_DIR, ensure_output_dirs

# Configuration
BATCH_SIZE = 2
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def generate_features():
    ensure_output_dirs()
    print(f"--- Starting Feature Extraction on {DEVICE} ---")

    # 1) Build dataset/loader from processed MRI folders.
    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    dataset = MultiModalALSDataset(rootDirectory=str(DATA_DIR))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Found {len(dataset)} samples.")

    # 2) Restore trained CNN weights.
    if not CHECKPOINT_PATH.exists():
        print(f"Error: Checkpoint {CHECKPOINT_PATH} not found. Please run train_multimodal.py first.")
        return

    print("-> Loading model architecture...")
    model = ALSTriStreamClassifier().to(DEVICE)

    print(f"-> Loading weights from {CHECKPOINT_PATH}...")
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Tip: Ensure train_multimodal.py was run with the same model architecture.")
        return

    model.eval()
    print("Model weights loaded successfully.")
    
    # 3) Export one feature file per subject to the dedicated features folder.
    print(f"Extracting features to {FEATURES_DIR}...")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            t1, t2, flair = inputs
            t1, t2, flair = t1.to(DEVICE), t2.to(DEVICE), flair.to(DEVICE)

            # Pull features from each encoder branch before transformer fusion.
            t1_encoder = model.model.t1Encoder
            t2_encoder = model.model.t2Encoder
            flair_encoder = model.model.flairEncoder

            feat_t1 = t1_encoder(t1)
            feat_t2 = t2_encoder(t2)
            feat_flair = flair_encoder(flair)

            for i in range(t1.size(0)):
                global_idx = batch_idx * BATCH_SIZE + i
                if global_idx >= len(dataset):
                    break

                sample_id = dataset.samples[global_idx]['id']
                label_val = dataset.samples[global_idx]['label']

                save_path = FEATURES_DIR / f"{sample_id}_features.pt"
                torch.save({
                    'id': sample_id,
                    't1_feat': feat_t1[i].cpu(),
                    't2_feat': feat_t2[i].cpu(),
                    'flair_feat': feat_flair[i].cpu(),
                    'label': label_val
                }, save_path)

    print("--- Feature Extraction Complete ---")
    print(f"Saved feature tensors in: {FEATURES_DIR}")

if __name__ == "__main__":
    generate_features()
