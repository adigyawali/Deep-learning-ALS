"""
Extract pooled per-modality embeddings (FEATURE_DIM each) for every subject.

These are the lightweight pooled features. The downstream ViT consumes spatial
features (see generate_spatial_features.py); this script is here for ablation
experiments and back-compat.

Each output `.pt` file contains:
  {
    'id'         : str
    'subject_id' : str
    'site'       : str
    't1_feat'    : Tensor (FEATURE_DIM,)
    't2_feat'    : Tensor (FEATURE_DIM,)
    'flair_feat' : Tensor (FEATURE_DIM,)
    'label'      : float
  }
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from classifier import ALSTriStreamClassifier  # noqa: E402
from dataset import MultiModalALSDataset       # noqa: E402
from paths import (                             # noqa: E402
    CHECKPOINT_PATH,
    DATA_DIR,
    FEATURES_DIR,
    ensure_output_dirs,
)


def _resolve_device(prefer: str) -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> bool:
    """Load `encoder_weights.pth`, tolerating both raw state-dicts and full checkpoints."""
    if not ckpt_path.exists():
        return False
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    return True


def generate_features(
    *,
    data_dir: Path,
    features_dir: Path,
    ckpt_path: Path,
    device_name: str,
    target_shape: tuple[int, int, int],
    batch_size: int,
    require_checkpoint: bool,
) -> None:
    ensure_output_dirs()
    device = _resolve_device(device_name)
    print(f"--- Pooled feature extraction on {device} ---")
    features_dir.mkdir(parents=True, exist_ok=True)

    dataset = MultiModalALSDataset(rootDirectory=data_dir, transform=False, targetShape=target_shape)
    if len(dataset) == 0:
        print(f"No samples found in {data_dir}.")
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Samples: {len(dataset)}")

    model = ALSTriStreamClassifier().to(device)
    loaded = _load_checkpoint(model, ckpt_path, device)
    if loaded:
        print(f"Loaded CNN checkpoint: {ckpt_path}")
    elif require_checkpoint:
        print(f"Error: --require-checkpoint set but {ckpt_path} not found.")
        return
    else:
        print(f"No checkpoint at {ckpt_path} -- using MedicalNet pretrained weights only.")
    model.eval()

    print(f"Writing features to: {features_dir}")
    written = 0
    with torch.no_grad():
        offset = 0
        for inputs, _ in loader:
            t1, t2, flair = (v.to(device) for v in inputs)
            f_t1 = model.model.t1Encoder(t1).cpu()
            f_t2 = model.model.t2Encoder(t2).cpu()
            f_fl = model.model.flairEncoder(flair).cpu()

            for i in range(t1.size(0)):
                global_idx = offset + i
                if global_idx >= len(dataset):
                    break
                meta = dataset.samples[global_idx]
                payload = {
                    "id": meta["id"],
                    "subject_id": meta["subject_id"],
                    "site": meta.get("site", "UNK"),
                    "t1_feat": f_t1[i].contiguous(),
                    "t2_feat": f_t2[i].contiguous(),
                    "flair_feat": f_fl[i].contiguous(),
                    "label": float(meta["label"]),
                }
                out_path = features_dir / f"{meta['id']}_features.pt"
                tmp = out_path.with_suffix(".pt.tmp")
                torch.save(payload, tmp)
                tmp.replace(out_path)
                written += 1
            offset += t1.size(0)

    print(f"--- Done. Wrote {written} files. ---")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract pooled CNN features for the ViT.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--target-shape", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--require-checkpoint", action="store_true",
                   help="Fail if no CNN checkpoint exists (default uses MedicalNet only).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_features(
        data_dir=args.data_dir,
        features_dir=args.features_dir,
        ckpt_path=args.checkpoint,
        device_name=args.device,
        target_shape=tuple(args.target_shape),
        batch_size=args.batch_size,
        require_checkpoint=args.require_checkpoint,
    )
