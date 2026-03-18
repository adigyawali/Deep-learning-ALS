"""
dataset.py

Reads from the per-subject folder layout produced by preprocessing.py:

    Data/processed/
      C001_V1/
        C001_V1_T1.nii.gz
        C001_V1_T2.nii.gz
        C001_V1_FLAIR.nii.gz
      P010_V2/
        P010_V2_T1.nii.gz
        P010_V2_T2.nii.gz
        P010_V2_FLAIR.nii.gz
      _QC_Snapshots/   <- ignored automatically
      ...

This layout is written directly by preprocessing.py::process_subject(), which
saves files as {subj_id}_{visit_id}_T1/T2/FLAIR.nii.gz inside a folder named
{subj_id}_{visit_id}/.

Label convention: subject ID starting with 'C' -> control (0),
                  subject ID starting with 'P' -> ALS patient (1).

Normalisation: Z-score over brain-foreground voxels (intensity > 0).
  Min-max was replaced because it destroys inter-subject intensity
  relationships that carry clinical meaning in T1/T2/FLAIR.

Target shape: 96^3 by default. Change to (128,128,128) if VRAM allows.

Augmentation: flip + 90 degree rotation, only when transform=True.
  Pass transform=True for training loaders, False for val/test.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset


class MultiModalALSDataset(Dataset):
    """
    Dataset for the per-subject folder layout output by preprocessing.py.

    Parameters
    ----------
    rootDirectory : str | Path
        Path to Data/processed/ -- contains one sub-folder per subject/visit.
    transform : bool
        If True, applies random flip + rotation augmentation.
        Use True for training loaders only.
    targetShape : tuple
        Spatial dimensions to resize every volume to before passing to the CNN.
    """

    def __init__(
        self,
        rootDirectory,
        transform: bool = False,
        targetShape: tuple = (96, 96, 96),
    ):
        self.rootDirectory = Path(rootDirectory)
        self.transform     = transform
        self.targetShape   = targetShape
        self.samples: list = []
        self._prepareDataset()

    # ── Dataset construction ──────────────────────────────────────────────

    def _prepareDataset(self) -> None:
        """
        Walk each sub-folder of processed/, look for the three expected NIfTI
        files, and add the subject to self.samples if all three exist.
        Folders starting with '_' (e.g. _QC_Snapshots) are skipped.
        """
        subject_folders = sorted(
            f for f in self.rootDirectory.iterdir()
            if f.is_dir() and not f.name.startswith("_")
        )

        skipped = []
        for folder in subject_folders:
            folder_name = folder.name          # e.g. "C001_V1" or "P010_V2"
            parts       = folder_name.split("_")

            if len(parts) < 2:
                skipped.append((folder_name, "folder name has fewer than 2 parts"))
                continue

            subject_id = parts[0]              # "C001" or "P010"
            if subject_id.startswith("C"):
                label = 0.0
            elif subject_id.startswith("P"):
                label = 1.0
            else:
                skipped.append((folder_name, "subject ID does not start with C or P"))
                continue

            # Expected filenames match exactly what preprocessing.py writes
            t1_path    = folder / f"{folder_name}_T1.nii.gz"
            t2_path    = folder / f"{folder_name}_T2.nii.gz"
            flair_path = folder / f"{folder_name}_FLAIR.nii.gz"

            if not (t1_path.exists() and t2_path.exists() and flair_path.exists()):
                missing = [
                    m for m, p in [("T1", t1_path), ("T2", t2_path), ("FLAIR", flair_path)]
                    if not p.exists()
                ]
                skipped.append((folder_name, f"missing modalities: {missing}"))
                continue

            self.samples.append(
                {
                    "id":    folder_name,
                    "t1":    str(t1_path),
                    "t2":    str(t2_path),
                    "flair": str(flair_path),
                    "label": label,
                }
            )

        n_controls = sum(1 for s in self.samples if s["label"] == 0.0)
        n_patients = sum(1 for s in self.samples if s["label"] == 1.0)
        print(f"  Dataset ready: {len(self.samples)} subjects "
              f"({n_controls} controls, {n_patients} ALS patients)")

        if skipped:
            print(f"  Skipped {len(skipped)} folder(s):")
            for name, reason in skipped:
                print(f"    {name}: {reason}")

    # ── PyTorch Dataset interface ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        t1Volume    = self._loadVolume(sample["t1"])
        t2Volume    = self._loadVolume(sample["t2"])
        flairVolume = self._loadVolume(sample["flair"])

        if self.transform:
            t1Volume, t2Volume, flairVolume = self._augment(t1Volume, t2Volume, flairVolume)

        label = torch.tensor(sample["label"], dtype=torch.float32)
        return (t1Volume, t2Volume, flairVolume), label

    # ── Preprocessing ─────────────────────────────────────────────────────

    def _loadVolume(self, path: str) -> torch.Tensor:
        """Load a NIfTI volume, Z-score normalise, resize, return (1, D, H, W)."""
        data = nib.load(path).get_fdata(dtype=np.float32)

        # Z-score over foreground voxels only (brain mask: intensity > 0).
        # Using the full volume including background inflates std and
        # compresses the brain signal toward zero -- harmful for CNNs.
        foreground = data[data > 0]
        if foreground.size > 0:
            mu  = foreground.mean()
            std = max(foreground.std(), 1e-8)
        else:
            mu  = data.mean()
            std = max(data.std(), 1e-8)

        data = (data - mu) / std

        # Resize to common spatial shape (bilinear interpolation)
        zoom_factors = [self.targetShape[i] / data.shape[i] for i in range(3)]
        data = zoom(data, zoom_factors, order=1)

        # Add channel dimension -> (1, D, H, W)
        return torch.from_numpy(data[np.newaxis]).float()

    # ── Augmentation ──────────────────────────────────────────────────────

    @staticmethod
    def _augment(t1: torch.Tensor, t2: torch.Tensor, flair: torch.Tensor):
        """
        Synchronised augmentation applied identically across all three
        modalities to preserve spatial correspondence.

        Operations:
          - Random flip along one spatial axis (50% probability)
          - Random 90 degree rotation in a randomly chosen plane (50% probability)

        Both are valid for ALS because the disease is bilateral and diffuse --
        flipping or rotating the brain does not change the label.
        Do NOT jitter intensities per-modality independently; T1/T2/FLAIR
        intensity relationships are physically meaningful and must stay intact.
        """
        if np.random.rand() > 0.5:
            axis  = int(np.random.choice([1, 2, 3]))
            t1    = torch.flip(t1,    [axis])
            t2    = torch.flip(t2,    [axis])
            flair = torch.flip(flair, [axis])

        if np.random.rand() > 0.5:
            k    = int(np.random.randint(1, 4))
            dims = tuple(np.random.choice([1, 2, 3], size=2, replace=False).tolist())
            t1    = torch.rot90(t1,    k, dims)
            t2    = torch.rot90(t2,    k, dims)
            flair = torch.rot90(flair, k, dims)

        return t1, t2, flair