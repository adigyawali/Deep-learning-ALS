"""Verify dataset → DataLoader produces correct shapes and labels with synthetic NIfTI files.

Run from project root:
    pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from als_classifier.dataset import ALSDataset, compute_freq_magnitude  # noqa: E402


def _make_fake_subject(folder: Path, name: str, shape=(64, 64, 64)) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    rng = np.random.RandomState(abs(hash(name)) % (2**32))
    for mod in ("T1", "T2", "FLAIR"):
        arr = rng.randn(*shape).astype(np.float32) * 100 + 500
        nib.save(nib.Nifti1Image(arr, affine), str(folder / f"{name}_{mod}.nii.gz"))


def test_dataset_returns_3channel_volume_when_freq_off(tmp_path):
    name = "CALSNIC2_EDM_P999_V1_run-01"
    _make_fake_subject(tmp_path / name, name)

    ds = ALSDataset(
        [tmp_path / name], train=False, target_shape=(32, 32, 32),
        use_frequency=False,
    )
    vol, label = ds[0]

    assert torch.is_tensor(vol)
    assert vol.shape == (3, 32, 32, 32)
    assert label == 1


def test_dataset_returns_6channel_volume_when_freq_on(tmp_path):
    name = "CALSNIC2_EDM_P999_V1_run-01"
    _make_fake_subject(tmp_path / name, name)

    ds = ALSDataset(
        [tmp_path / name], train=False, target_shape=(32, 32, 32),
        use_frequency=True,
    )
    vol, _ = ds[0]

    assert vol.shape == (6, 32, 32, 32)
    # FFT-magnitude channels should not be identical to the spatial ones
    spatial, freq = vol[:3], vol[3:]
    assert not torch.allclose(spatial, freq)
    # z-scored channels should have ~0 mean / ~1 std (loose bounds)
    for c in range(3, 6):
        assert abs(vol[c].mean().item()) < 0.2
        assert 0.5 < vol[c].std().item() < 2.0


def test_dataset_label_for_control(tmp_path):
    name = "CALSNIC2_EDM_C001_V1_run-01"
    _make_fake_subject(tmp_path / name, name)
    ds = ALSDataset(
        [tmp_path / name], train=False, target_shape=(32, 32, 32),
        use_frequency=False,
    )
    _, label = ds[0]
    assert label == 0


def test_dataloader_batches(tmp_path):
    names = [
        "CALSNIC2_EDM_P101_V1_run-01",
        "CALSNIC2_EDM_P102_V1_run-01",
        "CALSNIC2_EDM_C201_V1_run-01",
        "CALSNIC2_EDM_C202_V1_run-01",
    ]
    folders = []
    for n in names:
        _make_fake_subject(tmp_path / n, n)
        folders.append(tmp_path / n)

    ds = ALSDataset(
        folders, train=False, target_shape=(32, 32, 32),
        use_frequency=True,
    )
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    batches = list(dl)
    assert len(batches) == 2
    x, y = batches[0]
    assert x.shape == (2, 6, 32, 32, 32)
    assert y.shape == (2,)


def test_train_augmentations_dont_break(tmp_path):
    """Medium augmentation pipeline should run cleanly and preserve shapes."""
    name = "CALSNIC2_EDM_P777_V1_run-01"
    _make_fake_subject(tmp_path / name, name)
    ds = ALSDataset(
        [tmp_path / name], train=True, target_shape=(32, 32, 32),
        use_frequency=True, aug_level="medium",
    )
    for _ in range(3):
        vol, _ = ds[0]
        assert vol.shape == (6, 32, 32, 32)
        assert torch.isfinite(vol).all()


def test_compute_freq_magnitude_shapes_and_invariants():
    """FFT-magnitude helper: shape preserved, real-valued, finite, low DC at center."""
    x = torch.randn(1, 16, 16, 16)
    mag = compute_freq_magnitude(x)
    assert mag.shape == x.shape
    assert mag.dtype == torch.float32
    assert torch.isfinite(mag).all()
    # After fftshift, DC bin lives at the center index — and DC = sum of voxels,
    # so it should be the (or near the) max of the log-magnitude spectrum.
    center = (8, 8, 8)
    assert mag[0, center[0], center[1], center[2]].item() >= mag.mean().item()


def _make_subject_identical_modalities(folder: Path, name: str, shape=(48, 48, 48)) -> None:
    """Write a subject whose T1/T2/FLAIR are byte-identical.

    Any geometry-preserving, per-channel-shared augmentation must keep the
    three spatial channels identical. If augmentation were sampled
    independently per modality (the old bug), the channels would diverge.
    """
    folder.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    # An asymmetric ramp so flips/affines actually change the array.
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    arr = (zz * 3 + yy * 2 + xx).astype(np.float32)
    for mod in ("T1", "T2", "FLAIR"):
        nib.save(nib.Nifti1Image(arr.copy(), affine), str(folder / f"{name}_{mod}.nii.gz"))


def test_geometric_augmentation_is_shared_across_modalities(tmp_path):
    """With geometric-only augmentation, identical modalities stay byte-identical.

    This is the decisive multi-modal-registration guarantee: flip/affine are
    sampled ONCE per sample and applied to every channel together. Under the
    old per-modality bug each channel got an independent flip/rotation and
    these assertions would fail (and the model collapsed to ~chance).
    """
    from monai.transforms import Compose, RandAffine, RandFlip, ToTensor

    name = "CALSNIC2_EDM_P555_V1_run-01"
    _make_subject_identical_modalities(tmp_path / name, name)

    ds = ALSDataset(
        [tmp_path / name], train=True, target_shape=(32, 32, 32),
        use_frequency=True, aug_level="medium",
    )
    # Isolate the geometric transforms (intensity noise legitimately differs
    # per channel and is exercised separately below).
    ds.augment = Compose([
        RandFlip(prob=0.5, spatial_axis=0),
        RandFlip(prob=0.5, spatial_axis=1),
        RandFlip(prob=0.5, spatial_axis=2),
        RandAffine(prob=0.8, rotate_range=(0.2, 0.2, 0.2),
                   scale_range=(0.15, 0.15, 0.15), padding_mode="zeros"),
        ToTensor(),
    ])

    for _ in range(6):
        vol, _ = ds[0]
        t1, t2, flair = vol[0], vol[1], vol[2]
        assert torch.equal(t1, t2), "T1/T2 diverged — geometry not shared"
        assert torch.equal(t1, flair), "T1/FLAIR diverged — geometry not shared"
        # FFT channels are derived per modality from the (shared) spatial
        # volume, so they must match across modalities too.
        f1, f2, f3 = vol[3], vol[4], vol[5]
        assert torch.allclose(f1, f2, atol=1e-5)
        assert torch.allclose(f1, f3, atol=1e-5)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    return float(
        ((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-12)
    )


def test_modalities_stay_registered_under_full_medium_aug(tmp_path):
    """Full medium aug (incl. per-channel intensity noise): geometry stays shared.

    Identical inputs differ only by sub-noise per-channel intensity jitter, so
    cross-modality correlation must stay ~1. The old bug (independent flips)
    drives this toward 0.
    """
    name = "CALSNIC2_EDM_P558_V1_run-01"
    _make_subject_identical_modalities(tmp_path / name, name)
    ds = ALSDataset(
        [tmp_path / name], train=True, target_shape=(40, 40, 40),
        use_frequency=False, aug_level="medium",
    )
    for _ in range(8):
        vol, _ = ds[0]
        assert _pearson(vol[0], vol[1]) > 0.99
        assert _pearson(vol[0], vol[2]) > 0.99


def test_augmentation_actually_perturbs_training_samples(tmp_path):
    """Sanity check: augmentation is live (output varies across draws)."""
    name = "CALSNIC2_EDM_P556_V1_run-01"
    _make_subject_identical_modalities(tmp_path / name, name)

    ds = ALSDataset(
        [tmp_path / name], train=True, target_shape=(32, 32, 32),
        use_frequency=False, aug_level="medium",
    )
    draws = [ds[0][0] for _ in range(8)]
    # At least one pair of draws should differ — otherwise aug is a no-op.
    assert any(
        not torch.equal(draws[0], d) for d in draws[1:]
    ), "Augmentation produced identical outputs every time"


def test_eval_split_is_deterministic(tmp_path):
    """train=False must be fully deterministic (no augmentation leakage)."""
    name = "CALSNIC2_EDM_C055_V1_run-01"
    _make_subject_identical_modalities(tmp_path / name, name)
    ds = ALSDataset(
        [tmp_path / name], train=False, target_shape=(32, 32, 32),
        use_frequency=True, aug_level="medium",
    )
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)


def test_dataset_input_divisible_by_16_recommended():
    """Documenting the constraint, not enforcing it — model needs spatial dims % 16 == 0."""
    for d in (64, 96, 128, 160):
        assert d % 16 == 0, f"{d} is not a valid model input size"
