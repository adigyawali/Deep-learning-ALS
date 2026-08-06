"""
Multimodal MRI preprocessing for the ALS classification pipeline.

Pipeline per (subject, visit):
  1. Reorient T1, T2, FLAIR to RAS.
  2. N4 bias-field correction (optionally masked).
  3. Register T1 → MNI152 (affine by default; SyN if --nonlinear).
  4. Register T2 → T1 (rigid) and warp to MNI by composing transforms.
  5. Register FLAIR → T1 (rigid) and warp to MNI by composing transforms.
  6. Save the three MNI-space volumes and a QC montage.

Two cohorts are supported, and they share one raw layout (`raw/T1W`, `raw/T2W`,
`raw/FLAIR`, all subjects mixed together). They differ only in the trailing
token of the filename:

    CALSNIC : DATASET_SITE_SUBJECT_MODALITY_VISIT     CALSNIC2_EDM_C005_T1w10_V1
    CAPTURE : DATASET_SITE_SUBJECT_MODALITY_MONTHS    CAPTURE_CHU_C178_T1w10_00M

`V1/V2/V3` is a visit index; `00M/04M/08M/12M` is months since baseline. Both
are treated as the same thing — the timepoint that keys the pairing — so a
subject's timepoints stay distinct samples in either cohort.

Pairing is subject-keyed (NOT sorted-zip). Each modality folder is scanned
into a dict keyed by (subject_id, timepoint). A (subject, timepoint) is
processed iff all three modalities are present.

Rerun files (`_run-02` in CALSNIC, `_run2` in CAPTURE) are used like any other
scan — they are the repeat acquisition after a failed first scan, i.e. the good
one. Only the **suffix itself is ignored**: it is stripped from the processed
name, so `..._V1_run-02` and `..._V1` are the same sample. When both exist for
one (subject, timepoint, modality) the highest run wins.

The script auto-detects the modality subdirectories, skull-stripped first:
`T1W_synthstrip → T1W`, `T2W_synthstrip → T2PD_synthstrip → T2W → T2PD`,
`FLAIR_synthstrip → FLAIR`. The `T2PD*` names cover the dual-echo T2/PD series
some cohorts are exported as. Override any of them with `--t1-subdir` /
`--t2-subdir` / `--flair-subdir`. Filenames may also carry a trailing
`_synthstrip` regardless of which folder they sit in.

The processed folder/sample name keeps the full source structure
`DATASET_SITE_SUBJECT_TIMEPOINT` (e.g. `CALSNIC2_CAL_C003_V1`,
`CAPTURE_CHU_C178_00M`), derived from the raw filename minus its extension,
optional `_synthstrip` suffix, and modality token. Subject grouping/labels are
still extracted from this name by `src/als/splits.py`.

Output layout (one folder per subject-visit, all three modalities inside):
    Data/processed/<sample_id>/<sample_id>_T1.nii.gz
                              /<sample_id>_T2.nii.gz
                              /<sample_id>_FLAIR.nii.gz
    Data/processed/_QC_Snapshots/<sample_id>_QC.png
    Data/processed/manifest.csv      <-- index of all processed samples
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..splits import extract_subject_id, label_from_subject_id

try:
    import ants
except ImportError:
    ants = None  # imported lazily — the script can list pairs without ants


_T1_RE = re.compile(r"_T1w?\d*_", flags=re.IGNORECASE)
# T2 token, both cohorts: CALSNIC writes `T2w10`, CAPTURE writes `PDT2` (the
# dual-echo PD/T2 series). `T2PD` is accepted too — the folder is named that way
# in some exports. A pure-PD file (`PDw10`) deliberately does NOT match.
_T2_RE = re.compile(r"_(?:PD)?T2(?:PD)?w?\d*_", flags=re.IGNORECASE)
_FL_RE = re.compile(r"_FLAIR(?:3D|_?EPI)?_?", flags=re.IGNORECASE)
# Timepoint token: CALSNIC visit index (V1) or CAPTURE months-since-baseline (00M).
_TIMEPOINT = r"V\d+|\d+M"
_SUBJECT_VISIT_RE = re.compile(
    rf"(?:^|_)([CP]\d{{3,}})_(?:[A-Za-z0-9]+_)*?({_TIMEPOINT})(?=_|$)", flags=re.IGNORECASE
)
# Rerun suffix, both spellings: CALSNIC `_run-02`, CAPTURE `_run2`.
_RUN_RE = re.compile(r"_run-?(\d+)", flags=re.IGNORECASE)
# Modality token removed when building the processed folder/sample name, so the
# name keeps the full DATASET_SITE_SUBJECT_TIMEPOINT structure.
_MODALITY_TOKEN_RE = re.compile(
    r"_(FLAIR(?:3D|_?EPI)?|(?:PD)?T[12](?:PD)?w?\d*)(?=_|$)", flags=re.IGNORECASE
)


def folder_name_from_path(path: Path) -> str:
    """
    Processed folder / sample name from a raw filename: strip the `.nii.gz`
    extension, an optional `_synthstrip` suffix, the `_run-NN` / `_runN` rerun
    tag, and the modality token, leaving the DATASET_SITE_SUBJECT_TIMEPOINT
    structure.

    The rerun tag is dropped on purpose: a rerun is the same scan of the same
    subject at the same timepoint, so it must not become a separate sample.

    Examples
    --------
    >>> folder_name_from_path(Path("CALSNIC2_CAL_C003_T1w10_V1.nii.gz"))
    'CALSNIC2_CAL_C003_V1'
    >>> folder_name_from_path(Path("CALSNIC2_EDM_P110_T1w10_V1_run-02.nii.gz"))
    'CALSNIC2_EDM_P110_V1'
    >>> folder_name_from_path(Path("CAPTURE_CHU_P151_T1w10_00M_run2.nii.gz"))
    'CAPTURE_CHU_P151_00M'
    >>> folder_name_from_path(Path("CAPTURE_EDM_P151_T1w10_12M_synthstrip.nii.gz"))
    'CAPTURE_EDM_P151_12M'
    """
    name = path.name
    if name.lower().endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    name = _RUN_RE.sub("", name)
    if name.lower().endswith("_synthstrip"):
        name = name[: -len("_synthstrip")]
    return _MODALITY_TOKEN_RE.sub("", name)


# ─── Filename parsing ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScanFile:
    path: Path
    subject_id: str
    visit: str          # timepoint token: CALSNIC "V1" or CAPTURE "00M"
    run: int            # 0 = original acquisition, >0 = a rerun; higher supersedes

    @property
    def sample_id(self) -> str:
        # Full DATASET_SITE_SUBJECT_TIMEPOINT name (lab convention), derived from
        # the actual filename. Pairing still keys on (subject_id, visit).
        return folder_name_from_path(self.path)


def _parse_scan(path: Path, modality_re: re.Pattern) -> Optional[ScanFile]:
    """Return ScanFile if the filename matches `modality_re` and yields a subject+timepoint, else None."""
    name = path.name
    if not name.endswith(".nii.gz"):
        return None
    stripped = name[: -len(".nii.gz")]
    if not modality_re.search(stripped + "_"):
        return None
    # Remove the modality token so the subject+timepoint regex matches without it.
    cleaned = modality_re.sub("_", stripped + "_").strip("_")
    sv = _SUBJECT_VISIT_RE.search(cleaned)
    if not sv:
        return None
    # Dataset-qualified (CALSNIC_C003 vs CAPTURE_C003) so the two cohorts never
    # share a pairing key — same definition of "subject" the splitter uses.
    subject_id = extract_subject_id(cleaned)
    visit = sv.group(2).upper()
    run_match = _RUN_RE.search(name)
    run = int(run_match.group(1)) if run_match else 0
    return ScanFile(path=path, subject_id=subject_id, visit=visit, run=run)


def _scan_dir(directory: Path, modality_re: re.Pattern) -> dict[tuple[str, str], ScanFile]:
    """
    Scan a modality directory and return {(subject_id, timepoint) -> ScanFile}.

    Rerun files are used, only their `_run-NN` / `_runN` tag is ignored (it never
    reaches the sample name). Several files can therefore land on one key: the
    highest run wins, since a rerun is the repeat after a failed scan. Between
    two files of equal run, the canonical (shortest) name wins.
    """
    def rank(scan: ScanFile) -> tuple[int, int]:
        """"Which file represents this timepoint" — bigger wins: later rerun, then
        shorter (canonical) name. A full tie keeps the first in sorted order."""
        return (scan.run, -len(scan.path.name))

    best: dict[tuple[str, str], ScanFile] = {}
    skipped: list[Path] = []
    superseded: list[Path] = []
    for child in sorted(directory.iterdir()):
        if not child.is_file():
            continue
        scan = _parse_scan(child, modality_re)
        if scan is None:
            skipped.append(child)
            continue
        key = (scan.subject_id, scan.visit)
        prev = best.get(key)
        if prev is None:
            best[key] = scan
            continue
        # Same subject + timepoint: a rerun of it, or a stray extra name segment
        # (`..._P110_1_T2w10_V2` next to `..._P110_T2w10_V2`). Keep one, and say
        # which, so the choice is visible and deterministic rather than silent.
        if rank(scan) > rank(prev):
            best[key] = scan
            superseded.append(prev.path)
        else:
            superseded.append(scan.path)
    # Useful diagnostics without making the script error out.
    if superseded:
        print(f"  [scan] {directory.name}: {len(superseded)} file(s) superseded by a rerun or a "
              f"canonical name; e.g. {superseded[0].name}")
    if skipped:
        print(f"  [scan] {directory.name}: skipped {len(skipped)} unmatched file(s); e.g. {skipped[0].name}")
    return best


def find_triplets(
    t1_dir: Path,
    t2_dir: Path,
    flair_dir: Path,
) -> list[tuple[ScanFile, ScanFile, ScanFile]]:
    """Return matched (T1, T2, FLAIR) ScanFiles for every (subject, timepoint) with all three present."""
    t1 = _scan_dir(t1_dir, _T1_RE)
    t2 = _scan_dir(t2_dir, _T2_RE)
    fl = _scan_dir(flair_dir, _FL_RE)

    keys = sorted(set(t1) & set(t2) & set(fl))
    only_t1 = sorted(set(t1) - set(t2) - set(fl))
    only_missing_flair = sorted(set(t1) & set(t2) - set(fl))
    missing_t2 = sorted(set(t1) & set(fl) - set(t2))

    triplets = [(t1[k], t2[k], fl[k]) for k in keys]
    print(f"  Found triplets: {len(triplets)}")
    if only_t1:
        print(f"  Missing T2+FLAIR for : {[f'{a}_{b}' for a, b in only_t1[:5]]}...")
    if only_missing_flair:
        print(f"  Missing FLAIR for     : {[f'{a}_{b}' for a, b in only_missing_flair[:5]]}...")
    if missing_t2:
        print(f"  Missing T2 for        : {[f'{a}_{b}' for a, b in missing_t2[:5]]}...")
    return triplets


# ─── ANTs processing ──────────────────────────────────────────────────────


def _qc_snapshot(t1, t2, flair, out_path: Path) -> None:
    """Save a 3-modality middle-axial-slice QC PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mid(img) -> np.ndarray:
        if img is None:
            return np.zeros((10, 10), dtype=np.float32)
        arr = img.numpy()
        return np.rot90(arr[:, :, arr.shape[2] // 2])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, (t1, t2, flair), ("T1", "T2", "FLAIR")):
        ax.imshow(mid(img), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def process_case(
    sample_id: str,
    t1: ScanFile,
    t2: ScanFile,
    flair: ScanFile,
    processed_dir: Path,
    qc_dir: Path,
    nonlinear: bool,
) -> None:
    if ants is None:
        raise ImportError("ANTsPy is required: pip install antspyx")

    out_dir = processed_dir / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== [{sample_id}] ===")
    print(f"   T1    : {t1.path.name}")
    print(f"   T2    : {t2.path.name}")
    print(f"   FLAIR : {flair.path.name}")

    t1_img = ants.image_read(str(t1.path))
    t2_img = ants.image_read(str(t2.path))
    fl_img = ants.image_read(str(flair.path))

    print("   -> Reorienting to RAS...")
    t1_img = ants.reorient_image2(t1_img, orientation="RAS")
    t2_img = ants.reorient_image2(t2_img, orientation="RAS")
    fl_img = ants.reorient_image2(fl_img, orientation="RAS")

    print("   -> N4 bias-field correction...")
    t1_img = ants.n4_bias_field_correction(t1_img)
    t2_img = ants.n4_bias_field_correction(t2_img)
    fl_img = ants.n4_bias_field_correction(fl_img)

    mni = ants.image_read(ants.get_ants_data("mni"))

    transform_type = "SyN" if nonlinear else "Affine"
    print(f"   -> Registering T1 -> MNI152 ({transform_type})...")
    t1_reg = ants.registration(fixed=mni, moving=t1_img, type_of_transform=transform_type)
    t1_final = t1_reg["warpedmovout"]
    t1_fwd = t1_reg["fwdtransforms"]

    print("   -> Registering T2 -> T1 (Rigid) -> MNI...")
    t2_reg = ants.registration(fixed=t1_img, moving=t2_img, type_of_transform="Rigid")
    t2_final = ants.apply_transforms(
        fixed=mni,
        moving=t2_img,
        transformlist=t1_fwd + t2_reg["fwdtransforms"],
    )

    print("   -> Registering FLAIR -> T1 (Rigid) -> MNI...")
    fl_reg = ants.registration(fixed=t1_img, moving=fl_img, type_of_transform="Rigid")
    fl_final = ants.apply_transforms(
        fixed=mni,
        moving=fl_img,
        transformlist=t1_fwd + fl_reg["fwdtransforms"],
    )

    print("   -> Saving outputs...")
    ants.image_write(t1_final, str(out_dir / f"{sample_id}_T1.nii.gz"))
    ants.image_write(t2_final, str(out_dir / f"{sample_id}_T2.nii.gz"))
    ants.image_write(fl_final, str(out_dir / f"{sample_id}_FLAIR.nii.gz"))
    _qc_snapshot(t1_final, t2_final, fl_final, qc_dir / f"{sample_id}_QC.png")
    print(f"   -> Done: {sample_id}")


# ─── Manifest ─────────────────────────────────────────────────────────────


def write_manifest(processed_dir: Path, manifest_path: Path) -> int:
    """Re-scan processed_dir and rebuild manifest.csv. Returns the row count."""
    rows: list[dict[str, str]] = []
    for child in sorted(processed_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        name = child.name
        t1 = child / f"{name}_T1.nii.gz"
        t2 = child / f"{name}_T2.nii.gz"
        fl = child / f"{name}_FLAIR.nii.gz"
        if not (t1.exists() and t2.exists() and fl.exists()):
            continue
        sv = _SUBJECT_VISIT_RE.search(name)
        subject_id = extract_subject_id(name)
        visit = sv.group(2).upper() if sv else ""
        try:
            label = str(int(label_from_subject_id(subject_id)))
        except ValueError:
            continue
        rows.append({
            "sample_id": name,
            "subject_id": subject_id,
            "visit": visit,
            "label": label,
            "t1": str(t1),
            "t2": str(t2),
            "flair": str(fl),
        })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "subject_id", "visit", "label", "t1", "t2", "flair"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# ─── Entry point ──────────────────────────────────────────────────────────


# Modality subdirectory names to try, in order. Skull-stripped folders come first
# so they win when both are present. `T2PD*` covers the dual-echo T2/PD series the
# lab exports for some cohorts — the T2 images live in that folder.
_T1_SUBDIRS = ("T1W_synthstrip", "T1W")
_T2_SUBDIRS = ("T2W_synthstrip", "T2PD_synthstrip", "T2W", "T2PD")
_FLAIR_SUBDIRS = ("FLAIR_synthstrip", "FLAIR")


def _pick_dir(raw_root: Path, candidates: Iterable[str]) -> Path:
    """First candidate under ``raw_root`` that exists and holds .nii.gz files.

    Falls back to the first candidate so the caller's "folder not found" error
    names something recognisable when none of them are present.
    """
    names = list(candidates)
    for name in names:
        p = raw_root / name
        if p.is_dir() and any(p.glob("*.nii.gz")):
            return p
    return raw_root / names[0]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    # src/als/data/preprocessing.py -> project root is 3 levels up.
    project_root = Path(__file__).resolve().parents[3]
    default_raw = project_root / "Data" / "raw"
    default_processed = project_root / "Data" / "processed"

    p = argparse.ArgumentParser(description="ALS multimodal MRI preprocessing.")
    p.add_argument("--raw-dir", type=Path, default=default_raw, help="Raw NIfTI root (default: Data/raw).")
    p.add_argument("--processed-dir", type=Path, default=default_processed, help="Processed output root.")
    p.add_argument("--t1-subdir", type=str, default=None, help=f"Subdir under raw-dir for T1. Auto: {' → '.join(_T1_SUBDIRS)}.")
    p.add_argument("--t2-subdir", type=str, default=None, help=f"Subdir under raw-dir for T2. Auto: {' → '.join(_T2_SUBDIRS)}.")
    p.add_argument("--flair-subdir", type=str, default=None, help=f"Subdir under raw-dir for FLAIR. Auto: {' → '.join(_FLAIR_SUBDIRS)}.")
    p.add_argument("--nonlinear", action="store_true", help="Use SyN T1→MNI registration instead of Affine.")
    p.add_argument("--limit", type=int, default=0, help="If >0, only process the first N triplets (debug).")
    p.add_argument("--list-only", action="store_true", help="Print matched triplets and exit, no ANTs work.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    raw_dir: Path = args.raw_dir
    processed_dir: Path = args.processed_dir
    qc_dir = processed_dir / "_QC_Snapshots"

    t1_dir = raw_dir / args.t1_subdir if args.t1_subdir else _pick_dir(raw_dir, _T1_SUBDIRS)
    t2_dir = raw_dir / args.t2_subdir if args.t2_subdir else _pick_dir(raw_dir, _T2_SUBDIRS)
    flair_dir = raw_dir / args.flair_subdir if args.flair_subdir else _pick_dir(raw_dir, _FLAIR_SUBDIRS)

    print(f"Raw dir       : {raw_dir}")
    print(f"  T1 from     : {t1_dir.name}")
    print(f"  T2 from     : {t2_dir.name}")
    print(f"  FLAIR from  : {flair_dir.name}")
    print(f"Processed out : {processed_dir}")

    for d in (t1_dir, t2_dir, flair_dir):
        if not d.exists():
            print(f"ERROR: folder not found: {d}", file=sys.stderr)
            return 1

    processed_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    triplets = find_triplets(t1_dir, t2_dir, flair_dir)
    if args.limit > 0:
        triplets = triplets[: args.limit]

    if args.list_only:
        for t1, t2, fl in triplets:
            print(f"{t1.sample_id}\t{t1.path.name}\t{t2.path.name}\t{fl.path.name}")
        print(f"Total matched: {len(triplets)}")
        return 0

    if ants is None:
        print("ERROR: ANTsPy not installed (pip install antspyx) — required to run preprocessing.", file=sys.stderr)
        return 1

    processed = 0
    skipped = 0
    errors = 0
    for i, (t1, t2, fl) in enumerate(triplets, start=1):
        sample_id = t1.sample_id  # e.g. "CALSNIC2_EDM_C005_V1" / "CAPTURE_CHU_C178_00M"
        out_dir = processed_dir / sample_id
        outputs = [
            out_dir / f"{sample_id}_T1.nii.gz",
            out_dir / f"{sample_id}_T2.nii.gz",
            out_dir / f"{sample_id}_FLAIR.nii.gz",
        ]
        if all(p.exists() for p in outputs):
            print(f"  [{i:04d}/{len(triplets)}] SKIP (already done): {sample_id}")
            skipped += 1
            continue
        print(f"  [{i:04d}/{len(triplets)}] processing {sample_id}")
        try:
            process_case(sample_id, t1, t2, fl, processed_dir, qc_dir, nonlinear=args.nonlinear)
            processed += 1
        except Exception as exc:
            errors += 1
            print(f"  [{i:04d}/{len(triplets)}] ERROR for {sample_id}: {exc}", file=sys.stderr)
            traceback.print_exc()

    manifest_path = processed_dir / "manifest.csv"
    rows = write_manifest(processed_dir, manifest_path)

    print()
    print("=" * 64)
    print(f"Done.  newly_processed={processed}  already_done={skipped}  errors={errors}")
    print(f"manifest.csv rows: {rows}  ({manifest_path})")
    print("=" * 64)
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
