"""Dataset for (T1, T2, FLAIR) 3D MRI volumes with optional FFT channels.

Each subject-visit folder is named like:
    CALSNIC2_EDM_P110_V1_run-02
where the third underscore-token starts with 'P' (ALS) or 'C' (Control).

Inside each folder we expect three files:
    <folder_name>_T1.nii.gz
    <folder_name>_T2.nii.gz
    <folder_name>_FLAIR.nii.gz

Output volume shape:
    use_frequency=False -> (3, D, H, W)  [T1, T2, FLAIR]
    use_frequency=True  -> (6, D, H, W)  [T1, T2, FLAIR, FFT(T1), FFT(T2), FFT(FLAIR)]

The FFT channels are the log-magnitude of the 3D Fourier transform, fftshifted
so zero-frequency sits at the center of the volume, and z-scored per channel.
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
    RandAdjustContrast,
    RandAffine,
    RandFlip,
    RandGaussianNoise,
    ResizeWithPadOrCrop,
    Spacing,
    ToTensor,
)


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


def compute_freq_magnitude(x: torch.Tensor) -> torch.Tensor:
    """3D FFT magnitude of a single-channel volume, log-scaled and z-scored.

    Input:  (1, D, H, W) real-valued spatial volume
    Output: (1, D, H, W) real-valued frequency-magnitude volume

    Steps:
      1. fftn -> complex spectrum (same shape)
      2. |F|  -> magnitude (real, non-negative)
      3. log(1+|F|) -> compress huge dynamic range (DC bin dominates otherwise)
      4. fftshift -> move zero-frequency to the center (purely visual/convention)
      5. z-score -> match the spatial channels' intensity scale
    """
    spectrum = torch.fft.fftn(x, dim=(-3, -2, -1))
    mag = torch.log1p(torch.abs(spectrum))
    mag = torch.fft.fftshift(mag, dim=(-3, -2, -1))
    mag = (mag - mag.mean()) / (mag.std() + 1e-6)
    return mag


class ALSDataset(Dataset):
    """Yields (volume, label) where volume is (C, D, H, W).

    C = 3 when use_frequency is False, 6 when True.
    """

    MODALITIES: tuple[str, ...] = ("T1", "T2", "FLAIR")

    def __init__(
        self,
        folders: Iterable[Path],
        train: bool = True,
        target_shape: Sequence[int] = (128, 128, 128),
        target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
        use_frequency: bool = True,
        aug_level: str = "medium",
    ):
        self.folders = list(folders)
        self.train = train
        self.use_frequency = use_frequency

        base = [
            EnsureChannelFirst(channel_dim="no_channel"),
            Spacing(pixdim=tuple(target_spacing), mode="bilinear"),
            ResizeWithPadOrCrop(spatial_size=tuple(target_shape)),
            NormalizeIntensity(nonzero=True, channel_wise=True),
        ]
        aug = _build_augmentations(aug_level) if train else []
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
        spatial = [self._load_modality(folder, m) for m in self.MODALITIES]
        channels = list(spatial)
        if self.use_frequency:
            channels.extend(compute_freq_magnitude(x) for x in spatial)
        volume = torch.cat(channels, dim=0)  # (C, D, H, W)
        label = self.parse_label(folder.name)
        return volume, label


def _build_augmentations(level: str) -> list:
    """Return a list of MONAI random transforms by intensity level.

    'light'  — current behavior: 1-axis flip + tiny affine
    'medium' — flips on all 3 axes, stronger affine, gaussian noise, gamma
    'heavy'  — medium + bias field, smoothing
    """
    if level == "light":
        return [
            RandFlip(prob=0.5, spatial_axis=0),
            RandAffine(
                prob=0.5,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.05, 0.05, 0.05),
                padding_mode="zeros",
            ),
        ]
    if level == "medium":
        return [
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            RandAffine(
                prob=0.7,
                rotate_range=(0.15, 0.15, 0.15),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="zeros",
            ),
            RandGaussianNoise(prob=0.3, mean=0.0, std=0.05),
            RandAdjustContrast(prob=0.3, gamma=(0.7, 1.5)),
        ]
    if level == "heavy":
        from monai.transforms import RandBiasField, RandGaussianSmooth
        return [
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandFlip(prob=0.5, spatial_axis=2),
            RandAffine(
                prob=0.8,
                rotate_range=(0.2, 0.2, 0.2),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="zeros",
            ),
            RandGaussianNoise(prob=0.4, mean=0.0, std=0.07),
            RandAdjustContrast(prob=0.4, gamma=(0.6, 1.6)),
            RandBiasField(prob=0.3, coeff_range=(0.0, 0.1)),
            RandGaussianSmooth(prob=0.2, sigma_x=(0.25, 1.0), sigma_y=(0.25, 1.0), sigma_z=(0.25, 1.0)),
        ]
    raise ValueError(f"Unknown aug_level: {level!r}. Use 'light', 'medium', or 'heavy'.")
