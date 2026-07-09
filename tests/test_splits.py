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
    all_subjects = {x.subject_id for x in s}
    sp = make_subject_splits(s, n_folds=5, test_ratio=0.2, seed=42)
    assert sp["n_folds"] == 5 and len(sp["folds"]) == 5

    test = set(sp["test_subjects"])
    assert 0 < len(test) < len(all_subjects)                       # test is a real subset
    assert sp["class_counts"]["test"]["control"] >= 1 and sp["class_counts"]["test"]["patient"] >= 1

    pool = all_subjects - test
    val_union = set()
    for k, fold in enumerate(sp["folds"]):
        tr, va = set(fold["train_subjects"]), set(fold["val_subjects"])
        # No subject leaks between train / val / test within a fold.
        assert tr.isdisjoint(va) and tr.isdisjoint(test) and va.isdisjoint(test)
        # A fold's train+val is exactly the CV pool (every non-test subject used once).
        assert tr | va == pool
        # Each fold's validation set carries both classes.
        cc = sp["class_counts"]["folds"][k]["val"]
        assert cc["control"] >= 1 and cc["patient"] >= 1
        val_union |= va
    # Every CV-pool subject is a validation subject in exactly one fold.
    assert val_union == pool
    counts = {sid: sum(sid in set(f["val_subjects"]) for f in sp["folds"]) for sid in pool}
    assert all(c == 1 for c in counts.values())


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
    idx = indices_from_split(s, a, "train", fold=0)
    assert {s[i].subject_id for i in idx} == set(a["folds"][0]["train_subjects"])
    test_idx = indices_from_split(s, a, "test")
    assert {s[i].subject_id for i in test_idx} == set(a["test_subjects"])
