"""Canonical splitter: ID/label/site parsing, no subject leakage, reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from als.splits import (
    SampleMeta,
    extract_site,
    extract_subject_id,
    indices_from_split,
    label_from_subject_id,
    load_or_build_splits,
    make_subject_splits,
    read_splits,
    write_splits,
)


@pytest.mark.parametrize("name,expected", [
    ("C005_V1", "C005"),
    ("P110_V2_run-02", "P110"),
    ("CALSNIC2_EDM_C005_V1", "C005"),
    ("CALSNIC2_EDM_P110_V2_run-02", "P110"),
    ("c005_v1", "C005"),
])
def test_extract_subject_id(name, expected):
    assert extract_subject_id(name) == expected


def test_calsnic2_not_mistaken_for_subject():
    assert extract_subject_id("CALSNIC2_EDM_X_V1") == "CALSNIC2"
    assert extract_subject_id("CALSNIC2_EDM_P096_V1") == "P096"


def test_site_and_label():
    assert extract_site("CALSNIC2_EDM_C005_V1") == "EDM"
    assert extract_site("C005_V1") is None
    assert label_from_subject_id("C005") == 0.0
    assert label_from_subject_id("P110") == 1.0
    with pytest.raises(ValueError):
        label_from_subject_id("X999")


def _samples(n_c, n_p, visits=2):
    out = []
    for i in range(n_c):
        sid = f"C{i + 1:03d}"
        for v in range(1, visits + 1):
            out.append(SampleMeta(f"{sid}_V{v}", sid, 0.0, site="EDM"))
    for i in range(n_p):
        sid = f"P{i + 1:03d}"
        for v in range(1, visits + 1):
            out.append(SampleMeta(f"{sid}_V{v}", sid, 1.0, site="EDM"))
    return out


def test_no_subject_leakage_and_balance():
    s = _samples(60, 50, visits=3)
    sp = make_subject_splits(s, seed=42)
    tr, va, te = set(sp["train_subjects"]), set(sp["val_subjects"]), set(sp["test_subjects"])
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert tr | va | te == {x.subject_id for x in s}
    for kind in ("train", "val", "test"):
        cc = sp["class_counts"][kind]
        assert cc["control"] >= 1 and cc["patient"] >= 1


def test_roundtrip_and_reproducible(tmp_path: Path):
    s = _samples(20, 20)
    sp = make_subject_splits(s, seed=7)
    assert sp == make_subject_splits(s, seed=7)
    p = tmp_path / "splits.json"
    write_splits(p, sp)
    assert read_splits(p) == sp
    assert json.loads(p.read_text()) == sp


def test_load_or_build_writes_then_reads(tmp_path: Path):
    s = _samples(8, 8)
    p = tmp_path / "splits.json"
    a = load_or_build_splits(s, p, seed=1)
    assert p.exists()
    assert load_or_build_splits(s, p, seed=999) == a  # second call reuses the file, ignores seed
    idx = indices_from_split(s, a, "train")
    assert {s[i].subject_id for i in idx} == set(a["train_subjects"])
