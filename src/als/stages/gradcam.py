"""Stage: 3D Grad-CAM for the tri-stream CNN (CNN→ViT only).

Hooks each modality encoder's ``layer4`` and saves a per-modality heatmap NIfTI
(upsampled to the original grid, original affine) for one held-out test subject.
"""

from __future__ import annotations

import json
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

from ..config import get
from ..data.volume_dataset import VolumeDataset
from ..models.cnn_encoder import ALSTriStreamClassifier
from ..paths import DEFAULT_DATA_DIR, RunPaths
from ..splits import indices_from_split, read_splits

MODALITIES = ("t1", "t2", "flair")


def _prepare_volume(path: str, target_shape) -> torch.Tensor:
    data = nib.load(path).get_fdata(dtype=np.float32)
    fg = data[data > 0]
    mu, std = (fg.mean(), max(fg.std(), 1e-8)) if fg.size else (data.mean(), max(data.std(), 1e-8))
    data = (data - mu) / std
    factors = [target_shape[i] / data.shape[i] for i in range(3)]
    data = zoom(data, factors, order=1)
    return torch.from_numpy(data[np.newaxis, np.newaxis]).float()


def _save_heatmap(cam: np.ndarray, ref_path: str, save_path) -> None:
    ref = nib.load(ref_path)
    factors = [ref.shape[i] / cam.shape[i] for i in range(3)]
    cam_nat = np.clip(zoom(cam, factors, order=1), 0.0, None)
    if cam_nat.max() > 0:
        cam_nat = cam_nat / cam_nat.max()
    nib.save(nib.Nifti1Image(cam_nat.astype(np.float32), ref.affine, ref.header), str(save_path))


def run(cfg: dict, paths: RunPaths, device: torch.device, *, subject: str | None = None) -> None:
    data_dir = get(cfg, "data", "data_dir") or DEFAULT_DATA_DIR
    target_shape = tuple(get(cfg, "data", "target_shape", default=[128, 128, 128]))
    # Grad-CAM is illustrative, not a metric — use fold 0's CNN on the shared
    # held-out test subjects.
    gradcam_fold = int(get(cfg, "gradcam", "fold", default=0))
    fpaths = paths.fold(gradcam_fold)
    ckpt = fpaths.checkpoints / "cnn_best.pt"
    if not ckpt.exists():
        print(f"[gradcam] checkpoint {ckpt} not found. Train the CNN first.")
        return
    if not paths.splits_path.exists():
        print(f"[gradcam] {paths.splits_path} not found. Run training first.")
        return

    dataset = VolumeDataset(data_dir, return_mode="tuple", target_shape=target_shape, transform=False)
    splits = read_splits(paths.splits_path)
    test_idx = indices_from_split(dataset.to_sample_meta(), splits, "test")
    if not test_idx:
        print("[gradcam] test split is empty.")
        return
    test_samples = [dataset.samples[i] for i in test_idx]

    if subject:
        chosen = next((s for s in test_samples if s["id"] == subject or s["subject_id"] == subject.upper()), None)
        if chosen is None:
            print(f"[gradcam] {subject!r} not in test split. Available: {[s['id'] for s in test_samples]}")
            return
    else:
        chosen = random.choice(test_samples)

    model = ALSTriStreamClassifier(backbone=get(cfg, "cnn", "backbone", default="resnet50")).to(device)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    model.load_state_dict(state)
    model.eval()

    encoders = {"t1": model.t1Encoder, "t2": model.t2Encoder, "flair": model.flairEncoder}
    activations: dict[str, torch.Tensor] = {}
    hooks = []
    for name, enc in encoders.items():
        def _make(n):
            def _hook(_m, _i, output):
                activations[n] = output
            return _hook
        hooks.append(enc._backbone.layer4.register_forward_hook(_make(name)))

    inputs = {}
    for m in MODALITIES:
        x = _prepare_volume(chosen[m], target_shape).to(device)
        x.requires_grad_(True)
        inputs[m] = x

    logit = model(inputs["t1"], inputs["t2"], inputs["flair"]).squeeze()
    prob = torch.sigmoid(logit).item()
    grads = dict(zip(MODALITIES, torch.autograd.grad(logit, [activations[m] for m in MODALITIES])))
    for h in hooks:
        h.remove()

    out_dir = paths.root / "gradcam" / chosen["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in MODALITIES:
        A = activations[m].detach()[0]
        G = grads[m].detach()[0]
        weights = G.mean(dim=(1, 2, 3))
        cam = torch.relu((weights[:, None, None, None] * A).sum(dim=0))
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = F.interpolate(cam[None, None], size=target_shape, mode="trilinear",
                            align_corners=False).squeeze().cpu().numpy()
        _save_heatmap(cam, chosen[m], out_dir / f"{chosen['id']}_{m.upper()}_gradcam.nii.gz")

    meta = {"sample_id": chosen["id"], "subject_id": chosen["subject_id"],
            "true_label": "ALS" if chosen["label"] == 1.0 else "Control",
            "pred_label": "ALS" if prob >= 0.5 else "Control", "prob_ALS": float(prob),
            "checkpoint": str(ckpt), "target_layer": "_backbone.layer4"}
    (out_dir / "gradcam_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[gradcam] {chosen['id']} (true={meta['true_label']} pred={meta['pred_label']} "
          f"p={prob:.3f}) → {out_dir}")
