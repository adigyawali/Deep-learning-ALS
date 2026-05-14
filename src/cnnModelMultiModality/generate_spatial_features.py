"""
generate_spatial_features.py

Extract per-subject *spatial* feature maps from the CNN backbone (ResNet50
layer4) and save them to SPATIAL_FEATURES_DIR for the spatial ViT.

Each output .pt file contains:
  {
    'id'         : str    -- e.g. "P010_V2_run-02"
    'subject_id' : str    -- e.g. "P010"
    't1_feat'    : Tensor -- shape (C, D', H', W')
    't2_feat'    : Tensor -- shape (C, D', H', W')
    'flair_feat' : Tensor -- shape (C, D', H', W')
    'label'      : float  -- 0.0 (control) or 1.0 (ALS)
    'shape'      : tuple  -- (C, D', H', W') for sanity checks
  }

For the default 128^3 input through MONAI ResNet50:
  C = 2048, D' = H' = W' = 4  ->  64 spatial tokens per modality.

If a trained CNN checkpoint exists, its backbone weights are loaded (so any
backbone fine-tuning is preserved). If not, MedicalNet pretrained weights are
used as initialised in featureExtractor.py.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from classifier import ALSTriStreamClassifier
from dataset import MultiModalALSDataset
from paths import ARTIFACTS_DIR, CHECKPOINT_PATH, DATA_DIR, ensure_output_dirs

BATCH_SIZE = 1
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

SPATIAL_FEATURES_DIR = ARTIFACTS_DIR / "spatial_features"


def generate_spatial_features() -> None:
    SPATIAL_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    ensure_output_dirs()
    print(f"--- Spatial feature extraction on {DEVICE} ---")

    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    dataset = MultiModalALSDataset(
        rootDirectory=str(DATA_DIR),
        transform=False,
        targetShape=(128, 128, 128),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Found {len(dataset)} subjects.")

    model = ALSTriStreamClassifier().to(DEVICE)
    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Loaded CNN checkpoint: {CHECKPOINT_PATH}")
    else:
        print("No CNN checkpoint found -- using MedicalNet pretrained backbone only.")
    model.eval()

    t1_enc = model.model.t1Encoder
    t2_enc = model.model.t2Encoder
    flair_enc = model.model.flairEncoder

    print(f"Writing spatial features to: {SPATIAL_FEATURES_DIR}")
    written = 0
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(loader):
            t1, t2, flair = [v.to(DEVICE) for v in inputs]

            f_t1 = t1_enc.forward_features(t1)        # (B, C, D', H', W')
            f_t2 = t2_enc.forward_features(t2)
            f_fl = flair_enc.forward_features(flair)

            for i in range(t1.size(0)):
                global_idx = batch_idx * BATCH_SIZE + i
                if global_idx >= len(dataset):
                    break

                meta = dataset.samples[global_idx]
                sample_id = meta["id"]
                save_path = SPATIAL_FEATURES_DIR / f"{sample_id}_spatial.pt"

                payload = {
                    "id":         sample_id,
                    "subject_id": meta["subject_id"],
                    "t1_feat":    f_t1[i].cpu().contiguous(),
                    "t2_feat":    f_t2[i].cpu().contiguous(),
                    "flair_feat": f_fl[i].cpu().contiguous(),
                    "label":      float(meta["label"]),
                    "shape":      tuple(f_t1[i].shape),
                }
                torch.save(payload, save_path)
                written += 1

    print(f"--- Done. Wrote {written} files. ---")
    if written:
        first = next(SPATIAL_FEATURES_DIR.glob("*_spatial.pt"))
        sample = torch.load(first, map_location="cpu")
        print(f"First file: {first.name}")
        print(f"  t1_feat shape: {tuple(sample['t1_feat'].shape)}")
        print(f"  label:         {sample['label']}")


if __name__ == "__main__":
    generate_spatial_features()
