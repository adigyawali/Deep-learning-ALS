"""
Spatial CNN-feature dataset for the multi-modal ViT.

Reads the ``<id>_spatial.pt`` files written by the ``extract_features`` stage,
each containing ``(C, D', H', W')`` feature maps for T1/T2/FLAIR plus metadata.
Splits come from ``als.splits`` so this agrees with every other stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from ..splits import (
    SampleMeta,
    extract_site,
    extract_subject_id,
    indices_from_split,
    load_or_build_splits,
)


@dataclass
class FeatureSample:
    sample_id: str
    subject_id: str
    site: str
    path: Path
    label: float

    def to_meta(self) -> SampleMeta:
        return SampleMeta(sample_id=self.sample_id, subject_id=self.subject_id, label=self.label, site=self.site)


class ALSSpatialFeatureDataset(Dataset):
    """Lazy loader for per-subject ``*_spatial.pt`` files."""

    EXPECTED_KEYS = ("t1_feat", "t2_feat", "flair_feat", "label")

    def __init__(self, features_dir: str | Path):
        self.features_dir = Path(features_dir)
        self.samples: List[FeatureSample] = []
        self._channels: int | None = None
        self._spatial_shape: Tuple[int, int, int] | None = None
        self._scan()

    def _scan(self) -> None:
        if not self.features_dir.exists():
            return
        for file_path in sorted(self.features_dir.glob("*_spatial.pt")):
            payload = torch.load(file_path, map_location="cpu", weights_only=False)
            for key in self.EXPECTED_KEYS:
                if key not in payload:
                    raise ValueError(f"Missing key {key!r} in {file_path}")

            t1 = payload["t1_feat"]
            if t1.ndim != 4:
                raise ValueError(f"Expected (C, D, H, W) tensor, got shape {tuple(t1.shape)} in {file_path}")

            if self._channels is None:
                self._channels = int(t1.shape[0])
                self._spatial_shape = tuple(int(s) for s in t1.shape[1:])
            elif int(t1.shape[0]) != self._channels or tuple(int(s) for s in t1.shape[1:]) != self._spatial_shape:
                raise ValueError(
                    f"Inconsistent feature shape in {file_path}: "
                    f"{tuple(t1.shape)} vs expected ({self._channels}, *{self._spatial_shape})"
                )

            sample_id = payload.get("id", file_path.stem.replace("_spatial", ""))
            subject_id = payload.get("subject_id") or extract_subject_id(sample_id)
            site = payload.get("site") or (extract_site(sample_id) or "UNK")
            self.samples.append(FeatureSample(
                sample_id=sample_id,
                subject_id=subject_id,
                site=site,
                path=file_path,
                label=float(payload["label"]),
            ))

    @property
    def in_channels(self) -> int:
        return int(self._channels or 0)

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        return self._spatial_shape or (0, 0, 0)

    def to_sample_meta(self) -> list[SampleMeta]:
        return [s.to_meta() for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        payload = torch.load(sample.path, map_location="cpu", weights_only=False)
        t1 = payload["t1_feat"].float()
        t2 = payload["t2_feat"].float()
        flair = payload["flair_feat"].float()
        x = torch.stack([t1, t2, flair], dim=0)              # (3, C, D, H, W)
        y = torch.tensor(sample.label, dtype=torch.float32)
        return x, y, sample.sample_id


def compute_pos_weight(samples: List[FeatureSample], indices: List[int]) -> torch.Tensor:
    """pos_weight = N_neg/N_pos, clamped to [0.1, 10] to avoid runaway gradients."""
    n_pos = sum(1 for i in indices if samples[i].label == 1.0)
    n_neg = sum(1 for i in indices if samples[i].label == 0.0)
    if n_pos == 0:
        return torch.tensor(1.0)
    weight = max(0.1, min(10.0, n_neg / max(1, n_pos)))
    return torch.tensor(weight, dtype=torch.float32)


def indices_from(samples: List[FeatureSample], splits: dict, kind: str) -> list[int]:
    return indices_from_split([s.to_meta() for s in samples], splits, kind)


__all__ = [
    "ALSSpatialFeatureDataset",
    "FeatureSample",
    "compute_pos_weight",
    "indices_from",
    "load_or_build_splits",
]
