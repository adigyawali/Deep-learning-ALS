"""
Tests for src/ViTModel/dataset.py: scanning *_spatial.pt files, shape
consistency, and shared-splits hookup.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC / "ViTModel"))
_spec = importlib.util.spec_from_file_location("vit_dataset_module", _SRC / "ViTModel" / "dataset.py")
_vit_dataset = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["vit_dataset_module"] = _vit_dataset
_spec.loader.exec_module(_vit_dataset)
ALSSpatialFeatureDataset = _vit_dataset.ALSSpatialFeatureDataset
indices_from = _vit_dataset.indices_from
load_or_build_splits = _vit_dataset.load_or_build_splits


def _write_spatial(path: Path, sample_id: str, label: float, *, C=4, shape=(2, 2, 2), subject_id=None) -> None:
    payload = {
        "id": sample_id,
        "subject_id": subject_id or sample_id.split("_")[0],
        "site": "EDM",
        "t1_feat": torch.randn(C, *shape),
        "t2_feat": torch.randn(C, *shape),
        "flair_feat": torch.randn(C, *shape),
        "label": label,
        "shape": (C, *shape),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_spatial_dataset_scans_files(tmp_path: Path) -> None:
    _write_spatial(tmp_path / "C001_V1_spatial.pt", "C001_V1", 0.0)
    _write_spatial(tmp_path / "P096_V1_spatial.pt", "P096_V1", 1.0)
    ds = ALSSpatialFeatureDataset(features_dir=tmp_path)
    assert len(ds) == 2
    assert ds.in_channels == 4
    assert ds.spatial_shape == (2, 2, 2)
    x, y, sid = ds[0]
    assert x.shape == (3, 4, 2, 2, 2)
    assert y.dtype == torch.float32
    assert sid in {"C001_V1", "P096_V1"}


def test_load_or_build_splits_writes_then_reads(tmp_path: Path) -> None:
    # 6 subjects, 2 visits each.
    for sid in ("C001", "C002", "C003", "P001", "P002", "P003"):
        for v in (1, 2):
            label = 0.0 if sid.startswith("C") else 1.0
            _write_spatial(tmp_path / f"{sid}_V{v}_spatial.pt", f"{sid}_V{v}", label, subject_id=sid)
    ds = ALSSpatialFeatureDataset(features_dir=tmp_path)
    splits_path = tmp_path / "splits.json"
    splits_a = load_or_build_splits(ds.samples, splits_path, seed=42)
    assert splits_path.exists()
    splits_b = load_or_build_splits(ds.samples, splits_path, seed=42)
    assert splits_a == splits_b

    train_idx = indices_from(ds.samples, splits_a, "train")
    assert len(train_idx) > 0
    # No subject appears in more than one split.
    subj = lambda idxs: {ds.samples[i].subject_id for i in idxs}
    train_s = subj(indices_from(ds.samples, splits_a, "train"))
    val_s = subj(indices_from(ds.samples, splits_a, "val"))
    test_s = subj(indices_from(ds.samples, splits_a, "test"))
    assert train_s.isdisjoint(val_s)
    assert train_s.isdisjoint(test_s)
    assert val_s.isdisjoint(test_s)
