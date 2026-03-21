"""
generate_features.py

Extracts per-subject feature tensors from the trained CNN backbone and saves
them to FEATURES_DIR for downstream use by the ViT classifier.

Each output .pt file contains:
  {
    'id'        : str    — processed sample identifier (e.g. "P010_V2" or "P010_V2_run-02")
    't1_feat'   : Tensor — shape (FEATURE_DIM,)
    't2_feat'   : Tensor — shape (FEATURE_DIM,)
    'flair_feat': Tensor — shape (FEATURE_DIM,)
    'label'     : float  — 0.0 (control) or 1.0 (ALS)
  }

The ViT classifier should load these and treat the three modality vectors as
a sequence of tokens (same approach as the CascadedMixingTransformer, but now
the ViT can add positional encodings and deeper attention across them).
"""

import torch
from torch.utils.data import DataLoader

from classifier import ALSTriStreamClassifier
from dataset import MultiModalALSDataset
from paths import CHECKPOINT_PATH, DATA_DIR, FEATURES_DIR, ensure_output_dirs

BATCH_SIZE = 2
DEVICE     = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)


def generate_features() -> None:
    ensure_output_dirs()
    print(f"--- Feature Extraction on {DEVICE} ---")

    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    # No augmentation during feature extraction — we want deterministic features
    dataset = MultiModalALSDataset(rootDirectory=str(DATA_DIR), transform=False, targetShape=(128, 128, 128))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Found {len(dataset)} samples.")

    if not CHECKPOINT_PATH.exists():
        print(f"Error: Checkpoint {CHECKPOINT_PATH} not found.  Run train.py first.")
        return

    model = ALSTriStreamClassifier().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    print("Checkpoint loaded.")

    print(f"Extracting features to {FEATURES_DIR} ...")
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(loader):
            t1, t2, flair = [v.to(DEVICE) for v in inputs]

            # Extract directly from each modality encoder (before fusion)
            feat_t1    = model.model.t1Encoder(t1)       # (B, FEATURE_DIM)
            feat_t2    = model.model.t2Encoder(t2)
            feat_flair = model.model.flairEncoder(flair)

            for i in range(t1.size(0)):
                global_idx = batch_idx * BATCH_SIZE + i
                if global_idx >= len(dataset):
                    break

                sample_id = dataset.samples[global_idx]["id"]
                label_val = dataset.samples[global_idx]["label"]

                save_path = FEATURES_DIR / f"{sample_id}_features.pt"
                torch.save(
                    {
                        "id":         sample_id,
                        "t1_feat":    feat_t1[i].cpu(),
                        "t2_feat":    feat_t2[i].cpu(),
                        "flair_feat": feat_flair[i].cpu(),
                        "label":      label_val,
                    },
                    save_path,
                )

    print("--- Feature Extraction Complete ---")
    print(f"Features saved in: {FEATURES_DIR}")


if __name__ == "__main__":
    generate_features()
