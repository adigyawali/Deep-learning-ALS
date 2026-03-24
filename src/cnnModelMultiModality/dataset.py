"""
dataset.py

Reads from the per-subject folder layout produced by preprocessing.py:

    Data/processed/
      C001_V1/
        C001_V1_T1.nii.gz
        C001_V1_T2.nii.gz
        C001_V1_FLAIR.nii.gz
      C001_V1_run-02/
        C001_V1_run-02_T1.nii.gz
        C001_V1_run-02_T2.nii.gz
        C001_V1_run-02_FLAIR.nii.gz
      P010_V2/
        P010_V2_T1.nii.gz
        P010_V2_T2.nii.gz
        P010_V2_FLAIR.nii.gz
      _QC_Snapshots/   <- ignored automatically
      ...

"""

from pathlib import Path
import re

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
        targetShape: tuple = (128, 128, 128),
    ):
        self.rootDirectory = Path(rootDirectory)
        self.transform     = transform
        self.targetShape   = targetShape
        self.samples: list = []
        self._prepareDataset()

    # ── Dataset construction ──────────────────────────────────────────────

    @staticmethod
    def _extract_subject_id(folder_name: str) -> str | None:
        # Folder names are e.g. CALSNIC2_EDM_C007_V1 or CALSNIC2_CAL_P110_V2_run-02
        # The subject ID (C### or P###) sits between two underscores after the site code
        match = re.search(r'_([CP]\d+)_', folder_name, flags=re.IGNORECASE)
        return match.group(1).upper() if match else None

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
            folder_name = folder.name          # e.g. "C001_V1" or "P010_V2_run-02"

            subject_id = self._extract_subject_id(folder_name)
            if subject_id is None:
                skipped.append((folder_name, "folder name does not start with C### or P###"))
                continue

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
                    "subject_id": subject_id,
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

        Flips and 90-degree rotations are intentionally NOT used.
        All scans are registered to MNI152 space by preprocessing.py, so every
        brain has standardised orientation and coordinate axes.  A left-right
        flip would swap hemispheres and destroy any subtle asymmetric ALS
        markers.  A 90-degree rotation would make the brain anatomically
        nonsensical in MNI space.

        Instead we apply a small random affine perturbation — translation up to
        ±3 voxels and rotation up to ±5 degrees per axis — which simulates the
        realistic residuals left by ANTs registration.  The same random seed is
        used for all three modalities so their voxel correspondence is preserved.
        """
        try:
            from monai.transforms import RandAffine
        except ImportError:
            # MONAI not available — skip augmentation rather than crash
            return t1, t2, flair

        rand_affine = RandAffine(
            prob=0.8,
            translate_range=(3, 3, 3),            # voxels
            rotate_range=(0.087, 0.087, 0.087),   # radians ≈ 5 degrees per axis
            padding_mode="border",
            mode="bilinear",
        )

        # Fix one shared seed so all three modalities receive the identical
        # spatial transform — voxel correspondence stays intact.
        shared_seed = int(torch.randint(0, 2 ** 31, (1,)).item())

        rand_affine.set_random_state(seed=shared_seed)
        t1    = rand_affine(t1)

        rand_affine.set_random_state(seed=shared_seed)
        t2    = rand_affine(t2)

        rand_affine.set_random_state(seed=shared_seed)
        flair = rand_affine(flair)

        return t1, t2, flair