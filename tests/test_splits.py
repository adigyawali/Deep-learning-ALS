"""
Unit tests for src/splits.py — the canonical splitter.

Regressions covered:
  - Subject IDs are extracted from BOTH the long form (CALSNIC2_EDM_C005_V1)
    and the short folder form (C005_V1). The old regex silently returned None
    on the short form, emptying the dataset.
  - CALSNIC2 is not mis-parsed as subject "C2".
  - Splits never split a subject across folds (no multi-visit leakage).
  - Class balance is preserved when stratify_by_label is on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from splits import (
    SampleMeta,
    extract_site,
    extract_subject_id,
    indices_from_split,
    label_from_subject_id,
    make_subject_splits,
    read_splits,
    write_splits,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("C005_V1", "C005"),
        ("P110_V2_run-02", "P110"),
        ("CALSNIC2_EDM_C005_V1", "C005"),
        ("CALSNIC2_EDM_P110_V2_run-02", "P110"),
        ("CALSNIC2_EDM_C100_V1_run-02", "C100"),
        ("c005_v1", "C005"),                # case insensitive
    ],
)
def test_extract_subject_id_real_names(name: str, expected: str) -> None:
    assert extract_subject_id(name) == expected


def test_extract_subject_id_does_not_match_calsnic2() -> None:
    """Regression: the leading '_C2' substring of CALSNIC2_ must not become a subject id."""
    # Without a real subject, the function falls back to the first token.
    assert extract_subject_id("CALSNIC2_EDM_X_V1") == "CALSNIC2"
    # When a real subject is present, that wins.
    assert extract_subject_id("CALSNIC2_EDM_P096_V1") == "P096"


def test_extract_site() -> None:
    assert extract_site("CALSNIC2_EDM_C005_V1") == "EDM"
    assert extract_site("C005_V1") is None


def test_label_from_subject_id() -> None:
    assert label_from_subject_id("C005") == 0.0
    assert label_from_subject_id("P110") == 1.0
    with pytest.raises(ValueError):
        label_from_subject_id("X999")


def _make_samples(n_controls: int, n_patients: int, visits_per_subject: int = 2) -> list[SampleMeta]:
    samples = []
    for i in range(n_controls):
        sid = f"C{i + 1:03d}"
        for v in range(1, visits_per_subject + 1):
            samples.append(SampleMeta(f"{sid}_V{v}", sid, 0.0, site="EDM"))
    for i in range(n_patients):
        sid = f"P{i + 1:03d}"
        for v in range(1, visits_per_subject + 1):
            samples.append(SampleMeta(f"{sid}_V{v}", sid, 1.0, site="EDM"))
    return samples


def test_no_subject_leakage_across_splits() -> None:
    samples = _make_samples(60, 50, visits_per_subject=3)
    splits = make_subject_splits(samples, train_ratio=0.8, val_ratio=0.1, seed=42)

    train = set(splits["train_subjects"])
    val = set(splits["val_subjects"])
    test = set(splits["test_subjects"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    # Every subject lands in exactly one split.
    all_subjects = {s.subject_id for s in samples}
    assert train | val | test == all_subjects


def test_label_stratification_preserves_balance() -> None:
    samples = _make_samples(60, 50, visits_per_subject=1)
    splits = make_subject_splits(samples, train_ratio=0.8, val_ratio=0.1, seed=42)
    # Each split has at least one of each class.
    for kind in ("train", "val", "test"):
        cc = splits["class_counts"][kind]
        assert cc["control"] >= 1, f"{kind} has 0 controls"
        assert cc["patient"] >= 1, f"{kind} has 0 patients"


def test_indices_from_split_picks_correct_indices() -> None:
    samples = _make_samples(20, 20, visits_per_subject=2)
    splits = make_subject_splits(samples, seed=42)
    train_idx = indices_from_split(samples, splits, "train")
    train_subjects = {samples[i].subject_id for i in train_idx}
    assert train_subjects == set(splits["train_subjects"])


def test_splits_json_roundtrip(tmp_path: Path) -> None:
    samples = _make_samples(20, 20, visits_per_subject=2)
    splits = make_subject_splits(samples, seed=42)
    path = tmp_path / "splits.json"
    write_splits(path, splits)
    re_read = read_splits(path)
    assert re_read == splits
    # File is real JSON (atomic write + JSON parsing).
    assert json.loads(path.read_text()) == splits


def test_splits_are_seed_reproducible() -> None:
    samples = _make_samples(20, 20, visits_per_subject=1)
    a = make_subject_splits(samples, seed=7)
    b = make_subject_splits(samples, seed=7)
    assert a == b
