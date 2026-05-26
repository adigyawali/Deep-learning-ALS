"""
Per-subject 3D volume dataset for the tri-stream CNN.

Reads the per-subject folder layout produced by `src/preprocessing/preprocessing.py`:

    Data/processed/
      C005_V1/
        C005_V1_T1.nii.gz
        C005_V1_T2.nii.gz
        C005_V1_FLAIR.nii.gz
        C005_V1_mask.nii.gz       (optional, used by Z-score if present)
      P110_V2/
        ...
      manifest.csv                <-- written by preprocessing
      _QC_Snapshots/              <-- ignored

Subject IDs are extracted via the canonical `src/splits.py` regex so this stage
agrees with every downstream stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

# Allow this file to import the project-level splits module regardless of
# whether it's loaded as a package or a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))  # adds .../src/

from splits import SampleMeta, extract_site, extract_subject_id, label_from_subject_id  # noqa: E402


class MultiModalALSDataset(Dataset):
    """
    3D multi-modal dataset.

    Parameters
    ----------
    rootDirectory : str | Path
        Path to `Data/processed/` — one folder per subject-visit.
    transform : bool
        Apply synchronized augmentation across modalities (training only).
    targetShape : tuple
        Spatial size every volume is resampled to (default 128**3).
    useMask : bool
        If True and a `<sample>_mask.nii.gz` exists, Z-score over masked
        foreground voxels instead of `data > 0`.
    """

    def __init__(
        self,
        rootDirectory,
        transform: bool = False,
        targetShape: tuple = (128, 128, 128),
        useMask: bool = True,
    ):
        self.rootDirectory = Path(rootDirectory)
        self.transform = transform
        self.targetShape = tuple(targetShape)
        self.useMask = useMask
        self.samples: list[dict] = []
        self._prepareDataset()

    def _prepareDataset(self) -> None:
        if not self.rootDirectory.exists():
            print(f"  Dataset root not found: {self.rootDirectory}")
            return

        folders = sorted(
            f for f in self.rootDirectory.iterdir()
            if f.is_dir() and not f.name.startswith("_")
        )

        skipped: list[tuple[str, str]] = []
        for folder in folders:
            name = folder.name
            subject_id = extract_subject_id(name)
            # extract_subject_id falls back to the first token if no C###/P### match.
            # Validate explicitly: the first character of the subject id must be C or P.
            if not (subject_id.startswith("C") or subject_id.startswith("P")):
                skipped.append((name, f"unrecognized subject id {subject_id!r}"))
                continue
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
                "id": name,
                "subject_id": subject_id,
                "site": extract_site(name) or "UNK",
                "t1": str(t1),
                "t2": str(t2),
                "flair": str(fl),
                "mask": str(mask) if mask.exists() else None,
                "label": label,
            })

        n_c = sum(1 for s in self.samples if s["label"] == 0.0)
        n_p = sum(1 for s in self.samples if s["label"] == 1.0)
        print(f"  Dataset ready: {len(self.samples)} samples ({n_c} controls, {n_p} patients)")
        if skipped:
            print(f"  Skipped {len(skipped)} folder(s); first few:")
            for name, reason in skipped[:5]:
                print(f"    {name}: {reason}")

    def to_sample_meta(self) -> list[SampleMeta]:
        """Project samples into SampleMeta records for the canonical splitter."""
        return [
            SampleMeta(
                sample_id=s["id"],
                subject_id=s["subject_id"],
                label=s["label"],
                site=s.get("site"),
            )
            for s in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        t1 = self._loadVolume(s["t1"], s.get("mask"))
        t2 = self._loadVolume(s["t2"], s.get("mask"))
        flair = self._loadVolume(s["flair"], s.get("mask"))

        if self.transform:
            t1, t2, flair = self._augment(t1, t2, flair)

        return (t1, t2, flair), torch.tensor(s["label"], dtype=torch.float32)

    def _loadVolume(self, path: str, mask_path: str | None) -> torch.Tensor:
        """Load NIfTI → foreground Z-score → resize → (1, D, H, W) float32 tensor."""
        data = nib.load(path).get_fdata(dtype=np.float32)

        # Choose foreground voxels for Z-score statistics.
        if self.useMask and mask_path is not None:
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

        # Resize to common spatial shape (trilinear).
        factors = [self.targetShape[i] / data.shape[i] for i in range(3)]
        data = zoom(data, factors, order=1)

        return torch.from_numpy(data[np.newaxis]).float()

    @staticmethod
    def _augment(t1: torch.Tensor, t2: torch.Tensor, flair: torch.Tensor):
        """Synchronized random affine across modalities (no flips, no 90° rotations)."""
        try:
            from monai.transforms import RandAffine
        except ImportError:
            return t1, t2, flair

        seed = int(torch.randint(0, 2 ** 31, (1,)).item())

        def _apply(vol):
            rand = RandAffine(
                prob=0.8,
                translate_range=(3, 3, 3),
                rotate_range=(0.087, 0.087, 0.087),  # ~5 degrees per axis
                padding_mode="border",
                mode="bilinear",
            )
            rand.set_random_state(seed=seed)
            return rand(vol)

        return _apply(t1), _apply(t2), _apply(flair)
