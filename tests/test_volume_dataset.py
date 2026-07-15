"""Unified VolumeDataset: name parsing, return modes, frequency channels."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from als.data.volume_dataset import VolumeDataset, compute_freq_magnitude


def _nifti(path: Path, shape=(24, 24, 24), seed=0):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(rng.normal(size=shape).astype(np.float32).clip(min=0), np.eye(4)), str(path))


def _subject(root: Path, sample_id: str, seed=0):
    for i, m in enumerate(("T1", "T2", "FLAIR")):
        _nifti(root / sample_id / f"{sample_id}_{m}.nii.gz", seed=seed + i)


def test_short_and_long_names_and_labels(tmp_path: Path):
    _subject(tmp_path, "C005_V1", 0)
    _subject(tmp_path, "CALSNIC2_EDM_P096_V2_run-02", 3)
    (tmp_path / "_QC_Snapshots").mkdir()
    ds = VolumeDataset(tmp_path, return_mode="tuple", target_shape=(16, 16, 16))
    by_id = {s["id"]: s for s in ds.samples}
    assert by_id["C005_V1"]["label"] == 0.0
    assert by_id["CALSNIC2_EDM_P096_V2_run-02"]["subject_id"] == "P096"
    assert by_id["CALSNIC2_EDM_P096_V2_run-02"]["label"] == 1.0


def test_tuple_mode_shapes(tmp_path: Path):
    _subject(tmp_path, "C001_V1")
    ds = VolumeDataset(tmp_path, return_mode="tuple", target_shape=(16, 16, 16))
    (t1, t2, fl), y = ds[0]
    assert t1.shape == t2.shape == fl.shape == (1, 16, 16, 16)
    assert float(y) == 0.0


def test_stack_mode_spatial_only(tmp_path: Path):
    _subject(tmp_path, "P001_V1")
    ds = VolumeDataset(tmp_path, return_mode="stack", target_shape=(16, 16, 16), use_frequency=False)
    vol, y = ds[0]
    assert vol.shape == (3, 16, 16, 16) and float(y) == 1.0


def test_stack_mode_with_frequency(tmp_path: Path):
    _subject(tmp_path, "P002_V1")
    ds = VolumeDataset(tmp_path, return_mode="stack", target_shape=(16, 16, 16), use_frequency=True)
    vol, _ = ds[0]
    assert vol.shape == (6, 16, 16, 16)
    assert bool(torch.isfinite(vol).all())


def test_missing_modality_skipped(tmp_path: Path):
    _nifti(tmp_path / "C001_V1" / "C001_V1_T1.nii.gz")
    _nifti(tmp_path / "C001_V1" / "C001_V1_T2.nii.gz")  # FLAIR missing
    ds = VolumeDataset(tmp_path, return_mode="tuple", target_shape=(8, 8, 8))
    assert len(ds) == 0


def test_compute_freq_shape_and_finite():
    freq = compute_freq_magnitude(torch.randn(3, 12, 12, 12))
    assert freq.shape == (3, 12, 12, 12) and bool(torch.isfinite(freq).all())


def test_aug_config_disabled_builds_no_transforms(tmp_path: Path):
    _subject(tmp_path, "C001_V1")
    ds = VolumeDataset(tmp_path, return_mode="stack", target_shape=(16, 16, 16),
                       transform=True, aug_config={"enabled": False})
    assert ds._geom is None and ds._intensity is None


def test_aug_config_routes_transform_by_group(tmp_path: Path):
    _subject(tmp_path, "C001_V1")
    cfg = {"enabled": True, "transforms": [
        {"name": "RandFlip", "group": "geometric", "params": {"prob": 0.5, "spatial_axis": 0}},
    ]}
    ds = VolumeDataset(tmp_path, return_mode="stack", target_shape=(16, 16, 16),
                       transform=True, aug_config=cfg)
    (vol, _) = ds[0]
    assert vol.shape[0] == 3           # spatial-only stack still shaped correctly
    assert ds._geom is not None and ds._intensity is None
