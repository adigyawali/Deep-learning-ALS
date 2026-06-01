"""
Unit tests for the file-pairing logic in src/preprocessing/preprocessing.py.

These tests target the catastrophic-failure mode the audit found: pairing by
sorted-zip. We assert the new pairing is subject+visit keyed and survives
missing files, extra files, and run-02 reruns.
"""

from __future__ import annotations

from pathlib import Path

from preprocessing.preprocessing import (
    _FL_RE,
    _T1_RE,
    _T2_RE,
    _parse_scan,
    find_triplets,
    folder_name_from_path,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_parse_t1_t2_flair(tmp_path: Path) -> None:
    t1 = tmp_path / "CALSNIC2_EDM_C005_T1w10_V1.nii.gz"
    t2 = tmp_path / "CALSNIC2_EDM_C005_T2w10_V1.nii.gz"
    fl = tmp_path / "CALSNIC2_EDM_C005_FLAIR3D_V1.nii.gz"
    flair_epi = tmp_path / "CALSNIC2_EDM_P015_FLAIR_EPI_V1.nii.gz"
    for f in (t1, t2, fl, flair_epi):
        _touch(f)

    s1 = _parse_scan(t1, _T1_RE)
    assert s1 is not None and s1.subject_id == "C005" and s1.visit == "V1" and s1.run == 0
    s2 = _parse_scan(t2, _T2_RE)
    assert s2 is not None and s2.subject_id == "C005" and s2.visit == "V1"
    sf = _parse_scan(fl, _FL_RE)
    assert sf is not None and sf.subject_id == "C005" and sf.visit == "V1"
    sf2 = _parse_scan(flair_epi, _FL_RE)
    assert sf2 is not None and sf2.subject_id == "P015" and sf2.visit == "V1"


def test_run_suffix_parsed(tmp_path: Path) -> None:
    f = tmp_path / "CALSNIC2_EDM_C100_T1w10_V1_run-02.nii.gz"
    _touch(f)
    s = _parse_scan(f, _T1_RE)
    assert s is not None and s.run == 2


def test_find_triplets_subject_keyed(tmp_path: Path) -> None:
    """Three subjects, partial coverage; only fully-covered ones become triplets."""
    t1d = tmp_path / "T1W"
    t2d = tmp_path / "T2W"
    fld = tmp_path / "FLAIR"

    # C005 has all three modalities.
    _touch(t1d / "CALSNIC2_EDM_C005_T1w10_V1.nii.gz")
    _touch(t2d / "CALSNIC2_EDM_C005_T2w10_V1.nii.gz")
    _touch(fld / "CALSNIC2_EDM_C005_FLAIR3D_V1.nii.gz")

    # P096 has all three.
    _touch(t1d / "CALSNIC2_EDM_P096_T1w10_V1.nii.gz")
    _touch(t2d / "CALSNIC2_EDM_P096_T2w10_V1.nii.gz")
    _touch(fld / "CALSNIC2_EDM_P096_FLAIR3D_V1.nii.gz")

    # P099 is missing FLAIR — must NOT become a triplet.
    _touch(t1d / "CALSNIC2_EDM_P099_T1w10_V1.nii.gz")
    _touch(t2d / "CALSNIC2_EDM_P099_T2w10_V1.nii.gz")

    triplets = find_triplets(t1d, t2d, fld)
    ids = {t1.sample_id for t1, *_ in triplets}
    # Folder/sample name keeps the full DATASET_SITE_SUBJECT_VISIT structure.
    assert ids == {"CALSNIC2_EDM_C005_V1", "CALSNIC2_EDM_P096_V1"}


def test_folder_name_keeps_full_structure(tmp_path: Path) -> None:
    """Processed name = DATASET_SITE_SUBJECT_VISIT[_run-NN], modality token removed."""
    cases = {
        "CALSNIC2_CAL_C003_T1w10_V1.nii.gz": "CALSNIC2_CAL_C003_V1",
        "CALSNIC2_EDM_P115_T1w10_V1.nii.gz": "CALSNIC2_EDM_P115_V1",
        "CALSNIC2_EDM_P110_T1w10_V1_run-02.nii.gz": "CALSNIC2_EDM_P110_V1_run-02",
        "CALSNIC2_CAL_C007_T1w_V1_synthstrip.nii.gz": "CALSNIC2_CAL_C007_V1",
        "CALSNIC2_CAL_C003_FLAIR3D_V1.nii.gz": "CALSNIC2_CAL_C003_V1",
        "CALSNIC2_EDM_P015_FLAIR_EPI_V1.nii.gz": "CALSNIC2_EDM_P015_V1",
        # Lab layout: every raw file carries a trailing _synthstrip.
        "CALSNIC2_CAL_C003_T1w10_V1_synthstrip.nii.gz": "CALSNIC2_CAL_C003_V1",
        "CALSNIC2_EDM_P110_T1w10_V1_run-02_synthstrip.nii.gz": "CALSNIC2_EDM_P110_V1_run-02",
        "CALSNIC2_CAL_C003_FLAIR3D_V1_SynthStrip.nii.gz": "CALSNIC2_CAL_C003_V1",
    }
    for raw, expected in cases.items():
        assert folder_name_from_path(Path(raw)) == expected
    # And the full name still drives correct subject/label extraction.
    sf = _parse_scan(tmp_path / "CALSNIC2_CAL_C003_T1w10_V1.nii.gz", _T1_RE)
    assert sf is not None and sf.sample_id == "CALSNIC2_CAL_C003_V1" and sf.subject_id == "C003"


def test_find_triplets_prefers_highest_run(tmp_path: Path) -> None:
    """Reruns: when both base and _run-02 exist for the same (subject, visit), highest wins."""
    t1d = tmp_path / "T1W"
    t2d = tmp_path / "T2W"
    fld = tmp_path / "FLAIR"

    _touch(t1d / "CALSNIC2_EDM_C100_T1w10_V1.nii.gz")
    _touch(t1d / "CALSNIC2_EDM_C100_T1w10_V1_run-02.nii.gz")
    _touch(t2d / "CALSNIC2_EDM_C100_T2w10_V1.nii.gz")
    _touch(fld / "CALSNIC2_EDM_C100_FLAIR3D_V1.nii.gz")

    triplets = find_triplets(t1d, t2d, fld)
    assert len(triplets) == 1
    t1, t2, fl = triplets[0]
    assert t1.run == 2  # highest run won
    assert "run-02" in t1.path.name


def test_pairing_does_not_mix_modalities_across_subjects(tmp_path: Path) -> None:
    """
    The old sorted-zip approach would mis-align if any modality folder had a
    different file count. This test creates that exact scenario.
    """
    t1d = tmp_path / "T1W"
    t2d = tmp_path / "T2W"
    fld = tmp_path / "FLAIR"

    # T1 has C005, C007. T2 has only C007 (C005 missing). FLAIR has both.
    _touch(t1d / "CALSNIC2_EDM_C005_T1w10_V1.nii.gz")
    _touch(t1d / "CALSNIC2_EDM_C007_T1w10_V1.nii.gz")
    _touch(t2d / "CALSNIC2_EDM_C007_T2w10_V1.nii.gz")
    _touch(fld / "CALSNIC2_EDM_C005_FLAIR3D_V1.nii.gz")
    _touch(fld / "CALSNIC2_EDM_C007_FLAIR3D_V1.nii.gz")

    triplets = find_triplets(t1d, t2d, fld)
    # Only C007 has all three modalities.
    assert len(triplets) == 1
    t1, t2, fl = triplets[0]
    assert t1.subject_id == "C007"
    assert t2.subject_id == "C007"
    assert fl.subject_id == "C007"
