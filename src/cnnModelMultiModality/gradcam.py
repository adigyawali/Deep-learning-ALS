"""
gradcam.py

Generates 3D Grad-CAM heatmaps (one per modality: T1, T2, FLAIR) for a single
subject drawn from the held-out test split, and saves them as NIfTI volumes.

Why this works cleanly here
---------------------------
The tri-stream architecture has independent ResNet50 encoders for each
modality.  Hooking each encoder's last conv block (`layer4`) yields a clean
modality-specific Grad-CAM — backprop from the ALS logit is naturally
partitioned across the three streams by the late-fusion design.

Output layout
-------------
    src/cnn_features/cnn_multimodal/gradcam/<sample_id>/
        <sample_id>_T1_gradcam.nii.gz
        <sample_id>_T2_gradcam.nii.gz
        <sample_id>_FLAIR_gradcam.nii.gz
        gradcam_meta.json

Each heatmap is upsampled from layer4 resolution (~4^3) to the *original*
NIfTI shape and saved with the original affine, so it overlays correctly on
the corresponding modality scan in any NIfTI viewer (ITK-SNAP, FSLeyes, ...).

Run
---
    cd src/cnnModelMultiModality
    python gradcam.py                       # random subject from test split
    python gradcam.py --subject P110_V1     # specific test-split sample
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

from classifier import ALSTriStreamClassifier
from dataset import MultiModalALSDataset
from paths import ARTIFACTS_DIR, CHECKPOINT_PATH, DATA_DIR, ensure_output_dirs
from split_utils import split_indices_by_subject

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)
SPLIT_SEED   = 42
TARGET_SHAPE = (128, 128, 128)
MODALITIES   = ("t1", "t2", "flair")
GRADCAM_DIR  = ARTIFACTS_DIR / "gradcam"


# ── Volume preparation (mirrors dataset._loadVolume) ──────────────────────

def _prepare_volume(path: str) -> torch.Tensor:
    """Load NIfTI, foreground z-score, resize to TARGET_SHAPE → (1, 1, D, H, W)."""
    data = nib.load(path).get_fdata(dtype=np.float32)
    foreground = data[data > 0]
    if foreground.size > 0:
        mu, std = foreground.mean(), max(foreground.std(), 1e-8)
    else:
        mu, std = data.mean(), max(data.std(), 1e-8)
    data = (data - mu) / std
    zoom_factors = [TARGET_SHAPE[i] / data.shape[i] for i in range(3)]
    data = zoom(data, zoom_factors, order=1)
    return torch.from_numpy(data[np.newaxis, np.newaxis]).float()


# ── Subject selection ─────────────────────────────────────────────────────

def _pick_subject(samples, test_indices, requested):
    test_samples = [samples[i] for i in test_indices]
    if requested:
        for s in test_samples:
            if s["id"] == requested or s["subject_id"] == requested.upper():
                return s
        ids = [s["id"] for s in test_samples]
        raise SystemExit(
            f"Subject {requested!r} is not in the test split.\n"
            f"Available test samples: {ids}"
        )
    return random.choice(test_samples)


# ── Heatmap save (resize to original NIfTI grid + reuse its affine) ───────

def _save_heatmap_nifti(cam_128: np.ndarray, reference_path: str, save_path: Path) -> None:
    ref     = nib.load(reference_path)
    factors = [ref.shape[i] / cam_128.shape[i] for i in range(3)]
    cam_nat = zoom(cam_128, factors, order=1)
    cam_nat = np.clip(cam_nat, 0.0, None)
    if cam_nat.max() > 0:
        cam_nat = cam_nat / cam_nat.max()
    nib.save(
        nib.Nifti1Image(cam_nat.astype(np.float32), ref.affine, ref.header),
        str(save_path),
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="3D Grad-CAM for the ALS tri-stream model.")
    parser.add_argument(
        "--subject", default=None,
        help="Optional sample id (e.g. 'CALSNIC2_EDM_P110_V1') or subject id "
             "(e.g. 'P110'). Must be in the test split. Default: random.",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory {DATA_DIR} not found.")
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"Checkpoint {CHECKPOINT_PATH} not found.  Train the model first.")

    ensure_output_dirs()
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"--- Grad-CAM on {DEVICE} ---")
    dataset = MultiModalALSDataset(
        rootDirectory=str(DATA_DIR), transform=False, targetShape=TARGET_SHAPE,
    )
    _, _, test_indices = split_indices_by_subject(dataset.samples, seed=SPLIT_SEED)
    if not test_indices:
        raise SystemExit("Test split is empty.")

    sample = _pick_subject(dataset.samples, test_indices, args.subject)
    true_lbl = "ALS" if sample["label"] == 1.0 else "Control"
    print(f"-> Sample : {sample['id']}  (true label = {true_lbl})")

    model = ALSTriStreamClassifier().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    encoders = {
        "t1":    model.model.t1Encoder,
        "t2":    model.model.t2Encoder,
        "flair": model.model.flairEncoder,
    }

    # Forward hooks capture the layer4 activation tensor for each encoder.
    activations: dict[str, torch.Tensor] = {}
    hooks = []
    for name, enc in encoders.items():
        def _make(n):
            def _hook(_module, _inp, output):
                activations[n] = output
            return _hook
        hooks.append(enc._backbone.layer4.register_forward_hook(_make(name)))

    # Inputs need requires_grad=True so the autograd graph is built end-to-end
    # (backbone params are frozen, but the input being a non-leaf-with-grad
    #  ensures intermediate activations are part of the graph).
    inputs = {}
    for m in MODALITIES:
        x = _prepare_volume(sample[m]).to(DEVICE)
        x.requires_grad_(True)
        inputs[m] = x

    logit = model(inputs["t1"], inputs["t2"], inputs["flair"]).squeeze()
    prob  = torch.sigmoid(logit).item()
    pred  = "ALS" if prob >= 0.5 else "Control"
    print(f"-> Pred   : {pred}  (sigmoid = {prob:.4f})")

    # Backprop the ALS logit to all three layer4 activations in one pass.
    grads = torch.autograd.grad(
        outputs=logit,
        inputs=[activations[m] for m in MODALITIES],
        retain_graph=False,
        create_graph=False,
    )
    grads = dict(zip(MODALITIES, grads))

    for h in hooks:
        h.remove()

    out_dir = GRADCAM_DIR / sample["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("-> Computing per-modality heatmaps...")
    for m in MODALITIES:
        A = activations[m].detach()[0]   # (C, d, h, w)
        G = grads[m].detach()[0]         # (C, d, h, w)
        weights = G.mean(dim=(1, 2, 3))  # (C,)
        cam = (weights[:, None, None, None] * A).sum(dim=0)  # (d, h, w)
        cam = torch.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_128 = F.interpolate(
            cam[None, None], size=TARGET_SHAPE,
            mode="trilinear", align_corners=False,
        ).squeeze().cpu().numpy()

        save_path = out_dir / f"{sample['id']}_{m.upper()}_gradcam.nii.gz"
        _save_heatmap_nifti(cam_128, sample[m], save_path)
        print(f"     {save_path.name}  (range [0, 1], grid matches {m.upper()} scan)")

    meta = {
        "sample_id":     sample["id"],
        "subject_id":    sample["subject_id"],
        "true_label":    true_lbl,
        "pred_label":    pred,
        "prob_ALS":      float(prob),
        "checkpoint":    str(CHECKPOINT_PATH),
        "split_seed":    SPLIT_SEED,
        "target_layer":  "_backbone.layer4",
    }
    (out_dir / "gradcam_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved to: {out_dir}")
    print("View: open the original modality in FSLeyes/ITK-SNAP, then overlay")
    print("      the matching *_gradcam.nii.gz on top.")


if __name__ == "__main__":
    main()
