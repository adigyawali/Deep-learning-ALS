"""
Smoke + regression tests for the MultiModalALSDataset.

Most important regression: dataset folders named like 'C005_V1' (no leading
underscore) used to be silently skipped because the subject-id regex required
underscores on both sides. The new code must include them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _load_cnn_dataset_module():
    """
    Load src/cnnModelMultiModality/dataset.py under a unique module name so
    it does not collide with src/ViTModel/dataset.py.
    """
    src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src / "cnnModelMultiModality"))
    spec = importlib.util.spec_from_file_location(
        "cnn_dataset_module", src / "cnnModelMultiModality" / "dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["cnn_dataset_module"] = module
    spec.loader.exec_module(module)
    return module


_cnn_dataset = _load_cnn_dataset_module()
MultiModalALSDataset = _cnn_dataset.MultiModalALSDataset


def _write_dummy_nifti(path: Path, shape=(40, 40, 40), seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=shape).astype(np.float32).clip(min=0)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))


def _make_subject_folder(root: Path, sample_id: str, seed_offset: int = 0) -> None:
    folder = root / sample_id
    _write_dummy_nifti(folder / f"{sample_id}_T1.nii.gz", seed=seed_offset)
    _write_dummy_nifti(folder / f"{sample_id}_T2.nii.gz", seed=seed_offset + 1)
    _write_dummy_nifti(folder / f"{sample_id}_FLAIR.nii.gz", seed=seed_offset + 2)


def test_dataset_picks_up_short_folder_names(tmp_path: Path) -> None:
    """Regression: 'C005_V1'-style names must be parsed correctly (audit P0)."""
    _make_subject_folder(tmp_path, "C005_V1", seed_offset=0)
    _make_subject_folder(tmp_path, "P110_V2", seed_offset=3)
    (tmp_path / "_QC_Snapshots").mkdir()
    (tmp_path / "_QC_Snapshots" / "ignored.png").write_bytes(b"")

    ds = MultiModalALSDataset(rootDirectory=tmp_path, transform=False, targetShape=(32, 32, 32))
    ids = sorted(s["id"] for s in ds.samples)
    labels = sorted((s["id"], s["label"]) for s in ds.samples)
    assert ids == ["C005_V1", "P110_V2"]
    assert labels == [("C005_V1", 0.0), ("P110_V2", 1.0)]


def test_dataset_picks_up_long_folder_names(tmp_path: Path) -> None:
    _make_subject_folder(tmp_path, "CALSNIC2_EDM_C005_V1", seed_offset=0)
    _make_subject_folder(tmp_path, "CALSNIC2_EDM_P096_V2_run-02", seed_offset=3)

    ds = MultiModalALSDataset(rootDirectory=tmp_path, transform=False, targetShape=(32, 32, 32))
    by_id = {s["id"]: s for s in ds.samples}
    assert by_id["CALSNIC2_EDM_C005_V1"]["subject_id"] == "C005"
    assert by_id["CALSNIC2_EDM_P096_V2_run-02"]["subject_id"] == "P096"
    assert by_id["CALSNIC2_EDM_C005_V1"]["label"] == 0.0
    assert by_id["CALSNIC2_EDM_P096_V2_run-02"]["label"] == 1.0
    assert by_id["CALSNIC2_EDM_C005_V1"]["site"] == "EDM"


def test_getitem_returns_correct_shapes(tmp_path: Path) -> None:
    _make_subject_folder(tmp_path, "C001_V1", seed_offset=0)
    ds = MultiModalALSDataset(rootDirectory=tmp_path, transform=False, targetShape=(16, 16, 16))
    (t1, t2, fl), label = ds[0]
    assert t1.shape == (1, 16, 16, 16)
    assert t2.shape == (1, 16, 16, 16)
    assert fl.shape == (1, 16, 16, 16)
    assert float(label) == 0.0


def test_skips_folders_with_missing_modality(tmp_path: Path) -> None:
    """Subjects missing a modality must be skipped, not raise."""
    folder = tmp_path / "C001_V1"
    _write_dummy_nifti(folder / "C001_V1_T1.nii.gz")
    _write_dummy_nifti(folder / "C001_V1_T2.nii.gz")
    # FLAIR missing intentionally.
    ds = MultiModalALSDataset(rootDirectory=tmp_path, transform=False, targetShape=(8, 8, 8))
    assert len(ds) == 0
