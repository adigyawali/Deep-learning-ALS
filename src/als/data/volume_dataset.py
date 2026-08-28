"""
Unified per-subject 3D volume dataset.

Both models consume the *same* preprocessed MNI-space volumes from
``Data/processed/<sample_id>/`` (so preprocessing is comparable, Goal 8). One
class serves three consumers via ``return_mode``:

  * ``"tuple"``  → ``((t1, t2, flair), label)`` with each modality ``(1,D,H,W)``
                   — the tri-stream CNN encoder and feature extraction.
  * ``"stack"``  → ``(volume, label)`` with ``volume`` ``(3,D,H,W)``
                   — end-to-end nnMamba.

Both modes hand over the three spatial modalities and nothing else. The nnMamba
frequency stream is **not** built here: it is derived inside the model from the
CNN feature map (``models/cnn_nnmamba.py``) or the patch embedding
(``models/nnmamba.py``), so this dataset has no ``streams`` switch and never
returns 6 channels.

Multi-modal registration is sacred: T1/T2/FLAIR are co-registered, so any
geometric augmentation is sampled ONCE and applied identically to all three.
Augmentation is still split into a *geometric* group (flip/affine, shared across
channels) and an *intensity* group (noise/contrast), because that ordering keeps
the modalities aligned; both now apply to the same 3-channel volume.

Which augmentations run, and with what parameters, is **not** decided here: it
comes from the ``augmentations`` section of the root ``config.yaml`` (passed in
as ``aug_config``) and is turned into the two Compose pipelines by
``als.data.augment.build_transforms``. Pass ``aug_config=None`` to fall back to a
named preset via ``aug_level`` (the older behaviour).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from ..splits import SampleMeta, extract_site, extract_subject_id, label_from_subject_id
from .augment import build_transforms

ReturnMode = Literal["tuple", "stack"]


class VolumeDataset(Dataset):
    MODALITIES = ("t1", "t2", "flair")

    def __init__(
        self,
        root_dir: str | Path,
        *,
        return_mode: ReturnMode = "tuple",
        target_shape: tuple[int, int, int] = (128, 128, 128),
        transform: bool = False,
        use_mask: bool = True,
        aug_level: str = "medium",
        aug_config: dict | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.return_mode = return_mode
        self.target_shape = tuple(target_shape)
        self.transform = transform
        self.use_mask = use_mask
        self.aug_level = aug_level
        self.aug_config = aug_config
        self.samples: list[dict] = []
        # Augmentations come from config (single source of truth); aug_level is a
        # fallback preset only. Non-training datasets build nothing.
        self._geom, self._intensity = (
            build_transforms(aug_config, aug_level=aug_level) if transform else (None, None)
        )
        self._prepare()

    # ── scanning ────────────────────────────────────────────────────────
    def _prepare(self) -> None:
        if not self.root_dir.exists():
            print(f"  Dataset root not found: {self.root_dir}")
            return
        folders = sorted(f for f in self.root_dir.iterdir() if f.is_dir() and not f.name.startswith("_"))
        skipped: list[tuple[str, str]] = []
        for folder in folders:
            name = folder.name
            # subject_id is dataset-qualified (CALSNIC_C005 / CAPTURE_C178), so the
            # label must come from the C/P token, not the first character.
            subject_id = extract_subject_id(name)
            try:
                label = label_from_subject_id(subject_id)
            except ValueError as e:
                skipped.append((name, str(e)))
                continue
            t1 = folder / f"{name}_T1.nii.gz"
            t2 = folder / f"{name}_T2.nii.gz"
            fl = folder / f"{name}_FLAIR.nii.gz"
            mask = folder / f"{name}_mask.nii.gz"
            if not (t1.exists() and t2.exists() and fl.exists()):
                missing = [m for m, p in (("T1", t1), ("T2", t2), ("FLAIR", fl)) if not p.exists()]
                skipped.append((name, f"missing: {missing}"))
                continue
            self.samples.append({
                "id": name, "subject_id": subject_id,
                "site": extract_site(name) or "UNK",
                "t1": str(t1), "t2": str(t2), "flair": str(fl),
                "mask": str(mask) if mask.exists() else None,
                "label": label,
            })
        n_c = sum(1 for s in self.samples if s["label"] == 0.0)
        n_p = sum(1 for s in self.samples if s["label"] == 1.0)
        print(f"  Dataset ready: {len(self.samples)} samples ({n_c} controls, {n_p} patients)")
        if skipped:
            print(f"  Skipped {len(skipped)} folder(s); first few:")
            for nm, reason in skipped[:5]:
                print(f"    {nm}: {reason}")

    def to_sample_meta(self) -> list[SampleMeta]:
        return [SampleMeta(s["id"], s["subject_id"], s["label"], s.get("site")) for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    # ── loading ─────────────────────────────────────────────────────────
    def _load_volume(self, path: str, mask_path: str | None) -> torch.Tensor:
        """NIfTI → foreground/mask Z-score → trilinear resize → ``(1,D,H,W)``."""
        data = nib.load(path).get_fdata(dtype=np.float32)
        if self.use_mask and mask_path is not None:
            mask = nib.load(mask_path).get_fdata(dtype=np.float32) > 0.5
            if mask.shape != data.shape:
                mask = data > 0
            foreground = data[mask]
        else:
            foreground = data[data > 0]
        if foreground.size > 0:
            mu = float(foreground.mean())
            std = max(float(foreground.std()), 1e-8)
        else:
            mu = float(data.mean())
            std = max(float(data.std()), 1e-8)
        data = (data - mu) / std
        factors = [self.target_shape[i] / data.shape[i] for i in range(3)]
        data = zoom(data, factors, order=1)
        out = torch.from_numpy(data[np.newaxis]).float()
        del data  # free the resampled ndarray promptly (3 of these per item)
        return out

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        spatial = torch.cat([
            self._load_volume(s[m], s.get("mask")) for m in self.MODALITIES
        ], dim=0)  # (3, D, H, W) — channels share one geometry, so they stay registered

        if self.transform and self._geom is not None:
            spatial = self._as_tensor(self._geom(spatial))

        label = torch.tensor(s["label"], dtype=torch.float32)

        if self.transform and self._intensity is not None:
            spatial = self._as_tensor(self._intensity(spatial))

        if self.return_mode == "stack":
            return spatial, label                      # (3, D, H, W) — nnMamba

        # tuple mode (CNN): split the stacked volume back into the three streams.
        t1, t2, fl = spatial[0:1], spatial[1:2], spatial[2:3]
        return (t1.contiguous(), t2.contiguous(), fl.contiguous()), label

    # ── augmentation ────────────────────────────────────────────────────
    @staticmethod
    def _as_tensor(x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=torch.float32)
