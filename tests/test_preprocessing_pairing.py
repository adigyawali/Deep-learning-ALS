"""File-pairing logic: subject+timepoint keyed, CALSNIC + CAPTURE, reruns dropped."""

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
        ("CALSNIC2_EDM_C005_T1w10_V1.nii.gz", _T1_RE, "CALSNIC_C005", "V1", 0),
        ("CALSNIC2_EDM_C005_T2w10_V1.nii.gz", _T2_RE, "CALSNIC_C005", "V1", 0),
        ("CALSNIC2_EDM_C005_FLAIR3D_V1.nii.gz", _FL_RE, "CALSNIC_C005", "V1", 0),
        ("CALSNIC2_EDM_P015_FLAIR_EPI_V1.nii.gz", _FL_RE, "CALSNIC_P015", "V1", 0),
        ("CALSNIC2_EDM_C100_T1w10_V1_run-02.nii.gz", _T1_RE, "CALSNIC_C100", "V1", 2),
        # CAPTURE: timepoint is months-since-baseline, rerun suffix has no hyphen.
        ("CAPTURE_CHU_C178_T1w10_00M.nii.gz", _T1_RE, "CAPTURE_C178", "00M", 0),
        ("CAPTURE_CHU_P151_T2w10_12M.nii.gz", _T2_RE, "CAPTURE_P151", "12M", 0),
        ("CAPTURE_EDM_P180_FLAIR3D_04M.nii.gz", _FL_RE, "CAPTURE_P180", "04M", 0),
        ("CAPTURE_CHU_P151_T1w10_00M_run2.nii.gz", _T1_RE, "CAPTURE_P151", "00M", 2),
        ("CAPTURE_EDM_C065_T1w10_08M_synthstrip.nii.gz", _T1_RE, "CAPTURE_C065", "08M", 0),
    ]
    for fn, rx, subj, visit, run in cases:
        s = _parse_scan(tmp_path / fn, rx)
        assert s is not None, fn
        assert (s.subject_id, s.visit, s.run) == (subj, visit, run), fn


def test_capture_pdt2_is_the_t2_modality(tmp_path: Path):
    """CAPTURE exports T2 as the dual-echo `PDT2` series; a pure-PD file is not T2."""
    for fn in ("CAPTURE_CHU_C178_PDT2_00M_synthstrip.nii.gz",
               "CALSNIC2_EDM_C005_T2PD10_V1.nii.gz"):
        assert _parse_scan(tmp_path / fn, _T2_RE) is not None, fn
    for fn in ("CAPTURE_CHU_C178_PD_00M_synthstrip.nii.gz",
               "CALSNIC2_EDM_C005_PDw10_V1.nii.gz"):
        assert _parse_scan(tmp_path / fn, _T2_RE) is None, fn
    # A PDT2 file must not be mistaken for T1 or FLAIR.
    pdt2 = tmp_path / "CAPTURE_CHU_C178_PDT2_00M_synthstrip.nii.gz"
    assert _parse_scan(pdt2, _T1_RE) is None and _parse_scan(pdt2, _FL_RE) is None


def test_lab_synthstrip_layout_pairs(tmp_path: Path):
    """The real lab layout: *_synthstrip folders, PDT2 for T2, _synthstrip filenames."""
    t1d = tmp_path / "T1W_synthstrip"
    t2d = tmp_path / "T2PD_synthstrip"
    fld = tmp_path / "FLAIR_synthstrip"
    _touch(t1d / "CAPTURE_CHU_P151_T1w10_12M_synthstrip.nii.gz")
    _touch(t2d / "CAPTURE_CHU_P151_PDT2_12M_synthstrip.nii.gz")
    _touch(fld / "CAPTURE_CHU_P151_FLAIR3D_12M_synthstrip.nii.gz")
    triplets = find_triplets(t1d, t2d, fld)
    assert len(triplets) == 1
    assert {s.sample_id for s in triplets[0]} == {"CAPTURE_CHU_P151_12M"}


def test_same_subject_number_in_two_cohorts_stays_distinct(tmp_path: Path):
    calsnic = _parse_scan(tmp_path / "CALSNIC2_EDM_C003_T1w10_V1.nii.gz", _T1_RE)
    capture = _parse_scan(tmp_path / "CAPTURE_EDM_C003_T1w10_00M.nii.gz", _T1_RE)
    assert calsnic.subject_id != capture.subject_id


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


def test_rerun_tag_is_ignored_but_the_file_is_used(tmp_path: Path):
    """The `_run-02` / `_run2` tag never reaches the sample name; the scan is used."""
    t1d, t2d, fld = tmp_path / "T1W", tmp_path / "T2W", tmp_path / "FLAIR"
    # C100 has both an original and a rerun T1 -> one sample, the rerun wins.
    _touch(t1d / "CALSNIC2_EDM_C100_T1w10_V1.nii.gz")
    _touch(t1d / "CALSNIC2_EDM_C100_T1w10_V1_run-02.nii.gz")
    _touch(t2d / "CALSNIC2_EDM_C100_T2w10_V1.nii.gz")
    _touch(fld / "CALSNIC2_EDM_C100_FLAIR3D_V1.nii.gz")
    # P151's only FLAIR is a rerun -> still pairs, under the untagged name.
    _touch(t1d / "CAPTURE_CHU_P151_T1w10_00M.nii.gz")
    _touch(t2d / "CAPTURE_CHU_P151_T2w10_00M.nii.gz")
    _touch(fld / "CAPTURE_CHU_P151_FLAIR3D_00M_run2.nii.gz")

    triplets = find_triplets(t1d, t2d, fld)
    by_id = {t1.sample_id: (t1, t2, fl) for t1, t2, fl in triplets}
    assert set(by_id) == {"CALSNIC2_EDM_C100_V1", "CAPTURE_CHU_P151_00M"}
    # The rerun file is the one actually preprocessed for C100's T1 ...
    assert by_id["CALSNIC2_EDM_C100_V1"][0].path.name == "CALSNIC2_EDM_C100_T1w10_V1_run-02.nii.gz"
    # ... and P151's rerun FLAIR is used under a tag-free sample name.
    assert by_id["CAPTURE_CHU_P151_00M"][2].path.name == "CAPTURE_CHU_P151_FLAIR3D_00M_run2.nii.gz"


def test_capture_triplets_pair_by_months(tmp_path: Path):
    t1d, t2d, fld = tmp_path / "T1W", tmp_path / "T2W", tmp_path / "FLAIR"
    for months in ("00M", "04M", "12M"):
        _touch(t1d / f"CAPTURE_CHU_P152_T1w10_{months}.nii.gz")
        _touch(t2d / f"CAPTURE_CHU_P152_T2w10_{months}.nii.gz")
        _touch(fld / f"CAPTURE_CHU_P152_FLAIR3D_{months}.nii.gz")
    # A CALSNIC subject with the same number must not merge with the CAPTURE one.
    _touch(t1d / "CALSNIC2_EDM_P152_T1w10_V1.nii.gz")
    _touch(t2d / "CALSNIC2_EDM_P152_T2w10_V1.nii.gz")
    _touch(fld / "CALSNIC2_EDM_P152_FLAIR3D_V1.nii.gz")

    triplets = find_triplets(t1d, t2d, fld)
    assert {t1.sample_id for t1, *_ in triplets} == {
        "CAPTURE_CHU_P152_00M", "CAPTURE_CHU_P152_04M", "CAPTURE_CHU_P152_12M",
        "CALSNIC2_EDM_P152_V1",
    }


def test_rerun_beats_a_plain_scan_of_the_same_timepoint(tmp_path: Path):
    """A rerun is the repeat after a failed scan, so it supersedes the original."""
    t1d, t2d, fld = tmp_path / "T1W", tmp_path / "T2W", tmp_path / "FLAIR"
    _touch(t1d / "CAPTURE_CHU_P151_T1w10_00M.nii.gz")
    _touch(t1d / "CAPTURE_CHU_P151_T1w10_00M_run2.nii.gz")
    _touch(t2d / "CAPTURE_CHU_P151_T2w10_00M.nii.gz")
    _touch(fld / "CAPTURE_CHU_P151_FLAIR3D_00M.nii.gz")
    triplets = find_triplets(t1d, t2d, fld)
    assert len(triplets) == 1
    assert triplets[0][0].run == 2
    assert triplets[0][0].sample_id == "CAPTURE_CHU_P151_00M"


def test_duplicate_timepoint_keeps_the_canonical_name(tmp_path: Path):
    t1d, t2d, fld = tmp_path / "T1W", tmp_path / "T2W", tmp_path / "FLAIR"
    _touch(t1d / "CALSNIC2_EDM_P110_T1w10_V2.nii.gz")
    _touch(t2d / "CALSNIC2_EDM_P110_1_T2w10_V2.nii.gz")   # stray extra segment
    _touch(t2d / "CALSNIC2_EDM_P110_T2w10_V2.nii.gz")
    _touch(fld / "CALSNIC2_EDM_P110_FLAIR3D_V2.nii.gz")
    triplets = find_triplets(t1d, t2d, fld)
    assert len(triplets) == 1
    assert triplets[0][1].path.name == "CALSNIC2_EDM_P110_T2w10_V2.nii.gz"


def test_folder_name_strips_modality_and_synthstrip():
    cases = {
        "CALSNIC2_CAL_C003_T1w10_V1.nii.gz": "CALSNIC2_CAL_C003_V1",
        "CALSNIC2_CAL_C007_T1w_V1_synthstrip.nii.gz": "CALSNIC2_CAL_C007_V1",
        "CALSNIC2_EDM_P015_FLAIR_EPI_V1.nii.gz": "CALSNIC2_EDM_P015_V1",
        "CAPTURE_CHU_C178_T1w10_00M.nii.gz": "CAPTURE_CHU_C178_00M",
        "CAPTURE_EDM_P151_FLAIR3D_12M_synthstrip.nii.gz": "CAPTURE_EDM_P151_12M",
        # The rerun tag is dropped, so a rerun names the same sample as the original.
        "CALSNIC2_EDM_P110_T1w10_V1_run-02.nii.gz": "CALSNIC2_EDM_P110_V1",
        "CAPTURE_CHU_P151_T1w10_00M_run2.nii.gz": "CAPTURE_CHU_P151_00M",
        "CAPTURE_CHU_P151_T1w10_00M_run2_synthstrip.nii.gz": "CAPTURE_CHU_P151_00M",
    }
    for raw, expected in cases.items():
        assert folder_name_from_path(Path(raw)) == expected
