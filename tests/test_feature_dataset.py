"""Spatial CNN-feature dataset: scanning, shapes, shared-split hookup."""

from __future__ import annotations

from pathlib import Path

import torch

from als.data.feature_dataset import ALSSpatialFeatureDataset, indices_from, load_or_build_splits


def _write(path: Path, sample_id: str, label: float, *, C=4, shape=(2, 2, 2), subject_id=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "id": sample_id, "subject_id": subject_id or sample_id.split("_")[0], "site": "EDM",
        "t1_feat": torch.randn(C, *shape), "t2_feat": torch.randn(C, *shape),
        "flair_feat": torch.randn(C, *shape), "label": label, "shape": (C, *shape),
    }, path)


def test_scans_and_shapes(tmp_path: Path):
    _write(tmp_path / "C001_V1_spatial.pt", "C001_V1", 0.0)
    _write(tmp_path / "P096_V1_spatial.pt", "P096_V1", 1.0)
    ds = ALSSpatialFeatureDataset(features_dir=tmp_path)
    assert len(ds) == 2 and ds.in_channels == 4 and ds.spatial_shape == (2, 2, 2)
    x, y, sid = ds[0]
    assert x.shape == (3, 4, 2, 2, 2) and y.dtype == torch.float32 and sid in {"C001_V1", "P096_V1"}


def test_split_hookup_no_leakage(tmp_path: Path):
    for sid in ("C001", "C002", "C003", "P001", "P002", "P003"):
        for v in (1, 2):
            _write(tmp_path / f"{sid}_V{v}_spatial.pt", f"{sid}_V{v}",
                   0.0 if sid.startswith("C") else 1.0, subject_id=sid)
    ds = ALSSpatialFeatureDataset(features_dir=tmp_path)
    sp = load_or_build_splits(ds.samples, tmp_path / "splits.json", seed=42)
    subj = lambda kind: {ds.samples[i].subject_id for i in indices_from(ds.samples, sp, kind)}
    tr, va, te = subj("train"), subj("val"), subj("test")
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert len(tr) > 0
