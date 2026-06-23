"""Stage: dump per-subject CNN layer4 feature maps for the ViT (CNN→ViT stage 1.5).

Runs the fine-tuned tri-stream CNN over every subject and writes
``<id>_spatial.pt`` (one ``(C, D', H', W')`` map per modality + metadata) into
``runs/cnn_vit/features/``. Requires the ``cnn_best.pt`` checkpoint by default:
extracting features from an un-fine-tuned backbone silently degrades the ViT.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..config import get
from ..data.volume_dataset import VolumeDataset
from ..models.cnn_encoder import ALSTriStreamClassifier
from ..paths import DEFAULT_DATA_DIR, RunPaths


def _load_cnn(model: torch.nn.Module, ckpt_path, device) -> bool:
    if not ckpt_path.exists():
        return False
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[extract] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    return True


def run(cfg: dict, paths: RunPaths, device: torch.device, *, allow_missing_checkpoint: bool = False) -> None:
    data_dir = get(cfg, "data", "data_dir") or DEFAULT_DATA_DIR
    target_shape = tuple(get(cfg, "data", "target_shape", default=[128, 128, 128]))
    paths.features.mkdir(parents=True, exist_ok=True)

    dataset = VolumeDataset(data_dir, return_mode="tuple", target_shape=target_shape, transform=False)
    if len(dataset) == 0:
        print(f"[extract] No samples in {data_dir}.")
        return
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = ALSTriStreamClassifier(backbone=get(cfg, "cnn", "backbone", default="resnet50")).to(device)
    ckpt = paths.checkpoints / "cnn_best.pt"
    if _load_cnn(model, ckpt, device):
        print(f"[extract] loaded CNN checkpoint: {ckpt}")
    elif allow_missing_checkpoint or cfg.get("smoke"):
        print(f"[extract] no checkpoint at {ckpt} — using backbone weights only (degraded).")
    else:
        print(f"[extract] Error: {ckpt} not found. Train the CNN first, or pass "
              f"--allow-missing-checkpoint (ablation/debug only).")
        return
    model.eval()
    t1e, t2e, fle = model.t1Encoder, model.t2Encoder, model.flairEncoder

    written = 0
    with torch.no_grad():
        for idx, ((t1, t2, flair), _) in enumerate(loader):
            t1, t2, flair = t1.to(device), t2.to(device), flair.to(device)
            f1 = t1e.forward_features(t1).cpu()
            f2 = t2e.forward_features(t2).cpu()
            ff = fle.forward_features(flair).cpu()
            meta = dataset.samples[idx]
            payload = {
                "id": meta["id"], "subject_id": meta["subject_id"], "site": meta.get("site", "UNK"),
                "t1_feat": f1[0].contiguous(), "t2_feat": f2[0].contiguous(), "flair_feat": ff[0].contiguous(),
                "label": float(meta["label"]), "shape": tuple(f1[0].shape),
            }
            out = paths.features / f"{meta['id']}_spatial.pt"
            tmp = out.with_suffix(".pt.tmp")
            torch.save(payload, tmp)
            tmp.replace(out)
            written += 1

    print(f"[extract] wrote {written} feature files to {paths.features}")
    if written:
        first = next(paths.features.glob("*_spatial.pt"))
        sample = torch.load(first, map_location="cpu", weights_only=False)
        print(f"[extract] e.g. {first.name}: t1_feat {tuple(sample['t1_feat'].shape)} label={sample['label']}")
