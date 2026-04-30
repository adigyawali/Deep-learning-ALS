"""Dataset for 3-channel (T1, T2, FLAIR) 3D MRI volumes.

Each subject-visit folder is named like:
    CALSNIC2_EDM_P110_V1_run-02
where the third underscore-token starts with 'P' (ALS) or 'C' (Control).

Inside each folder we expect three files:
    <folder_name>_T1.nii.gz
    <folder_name>_T2.nii.gz
    <folder_name>_FLAIR.nii.gz
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    NormalizeIntensity,
    RandAffine,
    RandFlip,
    ResizeWithPadOrCrop,
    Spacing,
    ToTensor,
)


# Matches '_P110_' or '_C045_' inside the folder name; captures 'P' or 'C'.
LABEL_RE = re.compile(r"_([PC])\d+_")


def list_subject_folders(root: str | Path = "Data/processed") -> list[Path]:
    """Return all subject-visit folders under `root`, excluding hidden/QC entries."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )


class ALSDataset(Dataset):
    """Yields (volume, label) where volume is a (3, D, H, W) float tensor."""

    MODALITIES: tuple[str, ...] = ("T1", "T2", "FLAIR")

    def __init__(
        self,
        folders: Iterable[Path],
        train: bool = True,
        target_shape: Sequence[int] = (128, 128, 128),
        target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    ):
        self.folders = list(folders)
        self.train = train

        base = [
            EnsureChannelFirst(channel_dim="no_channel"),
            Spacing(pixdim=tuple(target_spacing), mode="bilinear"),
            ResizeWithPadOrCrop(spatial_size=tuple(target_shape)),
            NormalizeIntensity(nonzero=True, channel_wise=True),
        ]
        aug = []
        if train:
            aug = [
                RandFlip(prob=0.5, spatial_axis=0),
                RandAffine(
                    prob=0.5,
                    rotate_range=(0.1, 0.1, 0.1),
                    scale_range=(0.05, 0.05, 0.05),
                    padding_mode="zeros",
                ),
            ]
        self.transform = Compose(base + aug + [ToTensor()])

    def __len__(self) -> int:
        return len(self.folders)

    @staticmethod
    def parse_label(folder_name: str) -> int:
        """Return 1 if folder_name encodes an ALS patient (P###), 0 if control (C###)."""
        m = LABEL_RE.search(folder_name)
        if not m:
            raise ValueError(f"Cannot parse label from folder name: {folder_name!r}")
        return 1 if m.group(1) == "P" else 0

    def _load_modality(self, folder: Path, modality: str) -> torch.Tensor:
        path = folder / f"{folder.name}_{modality}.nii.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing modality file: {path}")
        arr = nib.load(str(path)).get_fdata().astype(np.float32)
        return self.transform(arr)  # -> (1, D, H, W) tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        folder = self.folders[idx]
        channels = [self._load_modality(folder, m) for m in self.MODALITIES]
        volume = torch.cat(channels, dim=0)  # (3, D, H, W)
        label = self.parse_label(folder.name)
        return volume, label
