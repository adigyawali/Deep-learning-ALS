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
    make_splits_from_explicit,
    make_subject_splits,
    read_splits,
    resolve_splits,
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


# ── Explicit (config-defined) folds ─────────────────────────────────────────

def test_explicit_folds_used_verbatim():
    s = _samples(6, 6, visits=1)   # C001..C006, P001..P006
    sp = make_splits_from_explicit(
        s,
        test_subjects=["C001", "P001"],
        folds=[["C002", "C003", "P002", "P003"],
               ["C004", "C005", "P004", "P005"]],
    )
    assert sp["mode"] == "explicit" and sp["n_folds"] == 2
    assert set(sp["test_subjects"]) == {"C001", "P001"}
    assert set(sp["folds"][0]["val_subjects"]) == {"C002", "C003", "P002", "P003"}
    # Fold 0's train set is exactly the other fold's subjects (C006/P006 unlisted → dropped).
    assert set(sp["folds"][0]["train_subjects"]) == {"C004", "C005", "P004", "P005"}
    # A dropped subject appears in no partition at all.
    everywhere = set(sp["test_subjects"])
    for f in sp["folds"]:
        everywhere |= set(f["train_subjects"]) | set(f["val_subjects"])
    assert "C006" not in everywhere and "P006" not in everywhere


def test_explicit_case_insensitive_ids():
    s = _samples(3, 3, visits=1)
    sp = make_splits_from_explicit(s, test_subjects=["c001"],
                                   folds=[["p001", "C002"], ["P002", "c003"]])
    assert set(sp["test_subjects"]) == {"C001"}
    assert set(sp["folds"][0]["val_subjects"]) == {"P001", "C002"}


def test_explicit_overlap_raises():
    s = _samples(4, 4, visits=1)
    with pytest.raises(ValueError, match="only once|both"):
        make_splits_from_explicit(s, test_subjects=["C001"],
                                  folds=[["C001", "P001"], ["C002", "P002"]])  # C001 in test AND fold 0


def test_explicit_requires_folds():
    s = _samples(2, 2, visits=1)
    with pytest.raises(ValueError, match="no 'folds'"):
        make_splits_from_explicit(s, test_subjects=["C001"], folds=[])


def test_explicit_indices_map_back_to_samples():
    s = _samples(4, 4, visits=2)   # multi-visit: each subject has 2 samples
    sp = make_splits_from_explicit(
        s, test_subjects=["C001", "P001"],
        folds=[["C002", "P002"], ["C003", "P003"]],
    )
    val0 = indices_from_split(s, sp, "val", fold=0)
    assert {s[i].subject_id for i in val0} == {"C002", "P002"}
    assert len(val0) == 4    # 2 subjects × 2 visits each


def test_resolve_splits_dispatches_explicit(tmp_path: Path):
    s = _samples(5, 5, visits=1)
    p = tmp_path / "splits.json"
    cv = {"mode": "explicit", "test_subjects": ["C001", "P001"],
          "folds": [["C002", "C003", "P002", "P003"], ["C004", "C005", "P004", "P005"]]}
    sp = resolve_splits(s, p, cv_cfg=cv)
    assert sp["mode"] == "explicit" and p.exists()
    assert read_splits(p) == sp


def test_resolve_splits_auto_default(tmp_path: Path):
    s = _samples(8, 8, visits=1)
    p = tmp_path / "splits.json"
    sp = resolve_splits(s, p, cv_cfg={"mode": "auto", "n_folds": 4, "test_ratio": 0.2}, seed=3)
    assert sp["mode"] == "auto" and sp["n_folds"] == 4


def test_resolve_splits_bad_mode(tmp_path: Path):
    s = _samples(4, 4, visits=1)
    with pytest.raises(ValueError, match="must be 'auto' or 'explicit'"):
        resolve_splits(s, tmp_path / "s.json", cv_cfg={"mode": "banana"})
