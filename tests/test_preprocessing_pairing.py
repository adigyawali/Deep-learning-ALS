"""File-pairing logic: subject+visit keyed, survives missing files / reruns."""

from __future__ import annotations

from pathlib import Path

from als.data.preprocessing import (
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


def test_parse_modalities(tmp_path: Path):
    cases = [
        ("CALSNIC2_EDM_C005_T1w10_V1.nii.gz", _T1_RE, "C005", "V1", 0),
        ("CALSNIC2_EDM_C005_T2w10_V1.nii.gz", _T2_RE, "C005", "V1", 0),
        ("CALSNIC2_EDM_C005_FLAIR3D_V1.nii.gz", _FL_RE, "C005", "V1", 0),
        ("CALSNIC2_EDM_P015_FLAIR_EPI_V1.nii.gz", _FL_RE, "P015", "V1", 0),
        ("CALSNIC2_EDM_C100_T1w10_V1_run-02.nii.gz", _T1_RE, "C100", "V1", 2),
    ]
    for fn, rx, subj, visit, run in cases:
        s = _parse_scan(tmp_path / fn, rx)
        assert s is not None and s.subject_id == subj and s.visit == visit and s.run == run


def test_find_triplets_subject_keyed(tmp_path: Path):
    t1d, t2d, fld = tmp_path / "T1W", tmp_path / "T2W", tmp_path / "FLAIR"
    for subj in ("C005", "P096"):
        _touch(t1d / f"CALSNIC2_EDM_{subj}_T1w10_V1.nii.gz")
        _touch(t2d / f"CALSNIC2_EDM_{subj}_T2w10_V1.nii.gz")
        _touch(fld / f"CALSNIC2_EDM_{subj}_FLAIR3D_V1.nii.gz")
    # P099 missing FLAIR — must not pair.
    _touch(t1d / "CALSNIC2_EDM_P099_T1w10_V1.nii.gz")
    _touch(t2d / "CALSNIC2_EDM_P099_T2w10_V1.nii.gz")
    triplets = find_triplets(t1d, t2d, fld)
    assert {t1.sample_id for t1, *_ in triplets} == {"CALSNIC2_EDM_C005_V1", "CALSNIC2_EDM_P096_V1"}


def test_highest_run_wins(tmp_path: Path):
    t1d, t2d, fld = tmp_path / "T1W", tmp_path / "T2W", tmp_path / "FLAIR"
    _touch(t1d / "CALSNIC2_EDM_C100_T1w10_V1.nii.gz")
    _touch(t1d / "CALSNIC2_EDM_C100_T1w10_V1_run-02.nii.gz")
    _touch(t2d / "CALSNIC2_EDM_C100_T2w10_V1.nii.gz")
    _touch(fld / "CALSNIC2_EDM_C100_FLAIR3D_V1.nii.gz")
    triplets = find_triplets(t1d, t2d, fld)
    assert len(triplets) == 1 and triplets[0][0].run == 2


def test_folder_name_strips_modality_and_synthstrip():
    cases = {
        "CALSNIC2_CAL_C003_T1w10_V1.nii.gz": "CALSNIC2_CAL_C003_V1",
        "CALSNIC2_EDM_P110_T1w10_V1_run-02.nii.gz": "CALSNIC2_EDM_P110_V1_run-02",
        "CALSNIC2_CAL_C007_T1w_V1_synthstrip.nii.gz": "CALSNIC2_CAL_C007_V1",
        "CALSNIC2_EDM_P015_FLAIR_EPI_V1.nii.gz": "CALSNIC2_EDM_P015_V1",
    }
    for raw, expected in cases.items():
        assert folder_name_from_path(Path(raw)) == expected
