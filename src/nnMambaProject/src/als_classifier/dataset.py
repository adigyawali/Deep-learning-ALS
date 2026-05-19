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

The FFT channels are the per-channel log-magnitude of the 3D Fourier transform,
fftshifted so zero-frequency sits at the volume centre, and z-scored per channel.

Augmentation contract (critical for multi-modal MRI)
----------------------------------------------------
T1/T2/FLAIR for a subject are co-registered: voxel (i,j,k) refers to the same
anatomical location in all three. Any *geometric* augmentation (flip, affine)
MUST therefore be sampled ONCE and applied IDENTICALLY to all modalities, or
the channels stop describing the same brain and the model cannot learn.

This is enforced structurally: deterministic preprocessing runs per modality
(safe — it is reproducible), the modalities are stacked into one
(C, D, H, W) tensor, and the random transforms are applied a single time to
that stacked tensor so every channel shares one sampled random state. The FFT
channels are derived *after* augmentation so they stay registered to their
spatial counterparts.
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
    """Per-channel 3D FFT log-magnitude, fftshifted and z-scored.

    Input:  (C, D, H, W) real-valued spatial volume
    Output: (C, D, H, W) real-valued frequency-magnitude volume

    Steps (applied independently per channel):
      1. fftn over the spatial dims -> complex spectrum
      2. |F|                        -> magnitude (real, non-negative)
      3. log(1+|F|)                 -> compress the huge DC-dominated range
      4. fftshift                   -> zero-frequency to the centre
      5. per-channel z-score        -> match the spatial channels' scale
    """
    spectrum = torch.fft.fftn(x, dim=(-3, -2, -1))
    mag = torch.log1p(torch.abs(spectrum))
    mag = torch.fft.fftshift(mag, dim=(-3, -2, -1))
    # Per-channel statistics: reduce only over the spatial dims so each
    # modality's spectrum is normalised on its own, not pooled together.
    dims = (-3, -2, -1)
    mean = mag.mean(dim=dims, keepdim=True)
    std = mag.std(dim=dims, keepdim=True)
    mag = (mag - mean) / (std + 1e-6)
    return mag.float()


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

        # Deterministic, reproducible preprocessing — safe to run per modality
        # because it produces identical output every call (no sampled state),
        # so modalities stay mutually registered.
        self.pre = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            Spacing(pixdim=tuple(target_spacing), mode="bilinear"),
            ResizeWithPadOrCrop(spatial_size=tuple(target_shape)),
            NormalizeIntensity(nonzero=True, channel_wise=True),
        ])

        # Random augmentation — applied ONCE to the stacked (C, D, H, W)
        # tensor so a single sampled random state is shared across all
        # modalities. Identity (just a tensor cast) when not training.
        aug = _build_augmentations(aug_level) if train else []
        self.augment = Compose(aug + [ToTensor()])

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
        """Load one modality and apply deterministic preprocessing only.

        Returns a (1, D, H, W) tensor. No random augmentation happens here —
        that is deferred to the shared, stacked pass in __getitem__.
        """
        path = folder / f"{folder.name}_{modality}.nii.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing modality file: {path}")
        arr = nib.load(str(path)).get_fdata().astype(np.float32)
        return torch.as_tensor(np.asarray(self.pre(arr)), dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        folder = self.folders[idx]

        # 1. Deterministic per-modality preprocessing -> uniform shape.
        spatial = [self._load_modality(folder, m) for m in self.MODALITIES]

        # 2. Stack into one multi-channel volume so the random transforms
        #    below see all modalities together and sample a single shared
        #    geometric/intensity state for every channel.
        volume = torch.cat(spatial, dim=0)  # (3, D, H, W)

        # 3. One shared augmentation pass across all channels.
        volume = torch.as_tensor(
            np.asarray(self.augment(volume)), dtype=torch.float32
        )

        # 4. Frequency channels derived from the *augmented* spatial volume,
        #    so each FFT channel stays registered to its modality.
        if self.use_frequency:
            volume = torch.cat([volume, compute_freq_magnitude(volume)], dim=0)

        label = self.parse_label(folder.name)
        return volume, label


def _build_augmentations(level: str) -> list:
    """Return a list of MONAI random transforms by intensity level.

    These run on the stacked (C, D, H, W) volume, so each transform samples
    one random state and applies it identically to every modality channel —
    preserving inter-modality registration.

    'light'  — 1-axis flip + tiny affine
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
