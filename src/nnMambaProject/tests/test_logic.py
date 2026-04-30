"""CPU-only tests for label parsing and subject-aware splitting.

Run from project root:
    pytest tests/test_logic.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from als_classifier.dataset import ALSDataset, LABEL_RE  # noqa: E402
from als_classifier.split import (  # noqa: E402
    SUBJECT_RE,
    extract_subject_id,
    split_by_subject,
)


SAMPLE_NAMES = [
    "CALSNIC2_EDM_P110_V1_run-02",
    "CALSNIC2_EDM_P110_V2_run-01",  # same subject, different visit
    "CALSNIC2_EDM_P111_V1_run-01",
    "CALSNIC2_TOR_P205_V1_run-01",
    "CALSNIC2_TOR_P206_V1_run-01",
    "CALSNIC2_EDM_C045_V1_run-01",
    "CALSNIC2_EDM_C045_V2_run-01",  # same control, different visit
    "CALSNIC2_TOR_C046_V1_run-01",
    "CALSNIC2_TOR_C047_V1_run-01",
    "CALSNIC2_TOR_C048_V1_run-01",
]


# ---------- label parsing ----------


def test_label_parsing_patient():
    assert ALSDataset.parse_label("CALSNIC2_EDM_P110_V1_run-02") == 1


def test_label_parsing_control():
    assert ALSDataset.parse_label("CALSNIC2_EDM_C045_V1_run-01") == 0


def test_label_parsing_invalid_raises():
    with pytest.raises(ValueError):
        ALSDataset.parse_label("invalid_folder_name")


def test_label_re_matches_expected():
    m = LABEL_RE.search("CALSNIC2_EDM_P110_V1_run-02")
    assert m is not None
    assert m.group(1) == "P"


# ---------- subject ID extraction ----------


def test_subject_id_patient():
    assert extract_subject_id("CALSNIC2_EDM_P110_V1_run-02") == "P110"


def test_subject_id_control():
    assert extract_subject_id("CALSNIC2_TOR_C046_V1_run-01") == "C046"


def test_subject_re_invalid_returns_none():
    assert SUBJECT_RE.match("not_a_real_folder") is None


# ---------- split logic ----------


def _make_folders(tmp_path: Path, names: list[str]) -> list[Path]:
    out = []
    for name in names:
        d = tmp_path / name
        d.mkdir()
        out.append(d)
    return out


def test_split_no_subject_in_two_splits(tmp_path):
    folders = _make_folders(tmp_path, SAMPLE_NAMES)
    train, val, test = split_by_subject(folders, val_frac=0.25, test_frac=0.25, seed=0)

    def subjects(fs):
        return {extract_subject_id(f.name) for f in fs}

    s_tr, s_va, s_te = subjects(train), subjects(val), subjects(test)
    assert s_tr.isdisjoint(s_va), f"Train/Val overlap: {s_tr & s_va}"
    assert s_tr.isdisjoint(s_te), f"Train/Test overlap: {s_tr & s_te}"
    assert s_va.isdisjoint(s_te), f"Val/Test overlap: {s_va & s_te}"


def test_split_visits_stay_together(tmp_path):
    folders = _make_folders(tmp_path, SAMPLE_NAMES)
    train, val, test = split_by_subject(folders, val_frac=0.25, test_frac=0.25, seed=0)

    p110 = [f for f in folders if "P110" in f.name]
    in_train = [f in train for f in p110]
    in_val = [f in val for f in p110]
    in_test = [f in test for f in p110]
    # All P110 visits must land in exactly one split
    assert (all(in_train) or all(in_val) or all(in_test)), (
        "P110 visits ended up in different splits"
    )


def test_split_covers_all_folders(tmp_path):
    folders = _make_folders(tmp_path, SAMPLE_NAMES)
    train, val, test = split_by_subject(folders, val_frac=0.25, test_frac=0.25, seed=0)
    assert len(train) + len(val) + len(test) == len(folders)


def test_split_invalid_fractions_raise(tmp_path):
    folders = _make_folders(tmp_path, SAMPLE_NAMES)
    with pytest.raises(ValueError):
        split_by_subject(folders, val_frac=0.6, test_frac=0.6)
    with pytest.raises(ValueError):
        split_by_subject(folders, val_frac=0.0, test_frac=0.2)
