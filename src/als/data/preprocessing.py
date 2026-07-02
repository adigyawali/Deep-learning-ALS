"""
Multimodal MRI preprocessing for the ALS classification pipeline.

Pipeline per (subject, visit):
  1. Reorient T1, T2, FLAIR to RAS.
  2. N4 bias-field correction (optionally masked).
  3. Register T1 → MNI152 (affine by default; SyN if --nonlinear).
  4. Register T2 → T1 (rigid) and warp to MNI by composing transforms.
  5. Register FLAIR → T1 (rigid) and warp to MNI by composing transforms.
  6. Save the three MNI-space volumes and a QC montage.

Pairing is subject-keyed (NOT sorted-zip). Each modality folder is scanned
into a dict keyed by (subject_id, visit). A (subject, visit) is processed iff
all three modalities are present. When multiple files exist for the same
(subject, visit, modality) — e.g. `_run-02` reruns — the most recent rerun
wins (highest run number), with no-suffix treated as run 0.

The script auto-detects skull-stripped inputs: it tries `T1W_synthstrip` and
falls back to `T1W` (same for T2/FLAIR). Override via CLI flags or env vars.

The processed folder/sample name keeps the full source structure
`DATASET_SITE_SUBJECT_VISIT[_run-NN]` (e.g. `CALSNIC2_CAL_C003_V1`), derived
from the raw filename minus its extension, optional `_synthstrip` suffix, and
modality token. Subject grouping/labels are still extracted from this name by
`src/splits.py`.

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

try:
    import ants
except ImportError:
    ants = None  # imported lazily — the script can list pairs without ants


_T1_RE = re.compile(r"_T1w?\d*_", flags=re.IGNORECASE)
_T2_RE = re.compile(r"_T2w?\d*_", flags=re.IGNORECASE)
_FL_RE = re.compile(r"_FLAIR(?:3D|_?EPI)?_?", flags=re.IGNORECASE)
_SUBJECT_VISIT_RE = re.compile(r"(?:^|_)([CP]\d{3,})_(?:[A-Za-z0-9]+_)*?(V\d+)", flags=re.IGNORECASE)
_RUN_RE = re.compile(r"_run-(\d+)", flags=re.IGNORECASE)
# Modality token removed when building the processed folder/sample name, so the
# name keeps the full DATASET_SITE_SUBJECT_VISIT[_run-NN] structure.
_MODALITY_TOKEN_RE = re.compile(r"_(FLAIR(?:3D|_?EPI)?|T[12]w?\d*)(?=_|$)", flags=re.IGNORECASE)


def folder_name_from_path(path: Path) -> str:
    """
    Processed folder / sample name from a raw filename: strip the `.nii.gz`
    extension, an optional `_synthstrip` suffix, and the modality token, keeping
    the full DATASET_SITE_SUBJECT_VISIT[_run-NN] structure.

    Examples
    --------
    >>> folder_name_from_path(Path("CALSNIC2_CAL_C003_T1w10_V1.nii.gz"))
    'CALSNIC2_CAL_C003_V1'
    >>> folder_name_from_path(Path("CALSNIC2_EDM_P110_T1w10_V1_run-02.nii.gz"))
    'CALSNIC2_EDM_P110_V1_run-02'
    """
    name = path.name
    if name.lower().endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    if name.lower().endswith("_synthstrip"):
        name = name[: -len("_synthstrip")]
    return _MODALITY_TOKEN_RE.sub("", name)


# ─── Filename parsing ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScanFile:
    path: Path
    subject_id: str
    visit: str
    run: int

    @property
    def sample_id(self) -> str:
        # Full DATASET_SITE_SUBJECT_VISIT[_run-NN] name (lab convention), derived
        # from the actual filename. Pairing still keys on (subject_id, visit).
        return folder_name_from_path(self.path)


def _parse_scan(path: Path, modality_re: re.Pattern) -> Optional[ScanFile]:
    """Return ScanFile if the filename matches `modality_re` and yields a subject+visit, else None."""
    name = path.name
    if not name.endswith(".nii.gz"):
        return None
    stripped = name[: -len(".nii.gz")]
    if not modality_re.search(stripped + "_"):
        return None
    # Remove the modality token so the subject+visit regex matches without it.
    cleaned = modality_re.sub("_", stripped + "_").strip("_")
    sv = _SUBJECT_VISIT_RE.search(cleaned)
    if not sv:
        return None
    subject_id = sv.group(1).upper()
    visit = sv.group(2).upper()
    run_match = _RUN_RE.search(name)
    run = int(run_match.group(1)) if run_match else 0
    return ScanFile(path=path, subject_id=subject_id, visit=visit, run=run)


def _scan_dir(directory: Path, modality_re: re.Pattern) -> dict[tuple[str, str], ScanFile]:
    """
    Scan a modality directory and return {(subject_id, visit) -> ScanFile}.
    When several runs exist for the same (subject, visit), the highest run wins.
    """
    best: dict[tuple[str, str], ScanFile] = {}
    skipped: list[Path] = []
    for child in sorted(directory.iterdir()):
        if not child.is_file():
            continue
        scan = _parse_scan(child, modality_re)
        if scan is None:
            skipped.append(child)
            continue
        key = (scan.subject_id, scan.visit)
        prev = best.get(key)
        if prev is None or scan.run > prev.run:
            best[key] = scan
    if skipped:
        # Useful diagnostic without making the script error out.
        print(f"  [scan] {directory.name}: skipped {len(skipped)} unmatched file(s); e.g. {skipped[0].name}")
    return best


def find_triplets(
    t1_dir: Path,
    t2_dir: Path,
    flair_dir: Path,
) -> list[tuple[ScanFile, ScanFile, ScanFile]]:
    """Return matched (T1, T2, FLAIR) ScanFiles for every (subject, visit) with all three present."""
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
        subject_id = sv.group(1).upper() if sv else name.split("_")[0].upper()
        visit = sv.group(2).upper() if sv else ""
        if subject_id.startswith("C"):
            label = "0"
        elif subject_id.startswith("P"):
            label = "1"
        else:
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


def _pick_dir(raw_root: Path, preferred: str, fallback: str) -> Path:
    """Return raw_root/preferred if it exists (and has .nii.gz files), else fallback."""
    p = raw_root / preferred
    if p.exists() and any(p.glob("*.nii.gz")):
        return p
    return raw_root / fallback


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    # src/als/data/preprocessing.py -> project root is 3 levels up.
    project_root = Path(__file__).resolve().parents[3]
    default_raw = project_root / "Data" / "raw"
    default_processed = project_root / "Data" / "processed"

    p = argparse.ArgumentParser(description="ALS multimodal MRI preprocessing.")
    p.add_argument("--raw-dir", type=Path, default=default_raw, help="Raw NIfTI root (default: Data/raw).")
    p.add_argument("--processed-dir", type=Path, default=default_processed, help="Processed output root.")
    p.add_argument("--t1-subdir", type=str, default=None, help="Subdir under raw-dir for T1. Auto: T1W_synthstrip → T1W.")
    p.add_argument("--t2-subdir", type=str, default=None, help="Subdir under raw-dir for T2. Auto: T2W_synthstrip → T2W.")
    p.add_argument("--flair-subdir", type=str, default=None, help="Subdir under raw-dir for FLAIR. Auto: FLAIR_synthstrip → FLAIR.")
    p.add_argument("--nonlinear", action="store_true", help="Use SyN T1→MNI registration instead of Affine.")
    p.add_argument("--limit", type=int, default=0, help="If >0, only process the first N triplets (debug).")
    p.add_argument("--list-only", action="store_true", help="Print matched triplets and exit, no ANTs work.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    raw_dir: Path = args.raw_dir
    processed_dir: Path = args.processed_dir
    qc_dir = processed_dir / "_QC_Snapshots"

    t1_dir = raw_dir / args.t1_subdir if args.t1_subdir else _pick_dir(raw_dir, "T1W_synthstrip", "T1W")
    t2_dir = raw_dir / args.t2_subdir if args.t2_subdir else _pick_dir(raw_dir, "T2W_synthstrip", "T2W")
    flair_dir = raw_dir / args.flair_subdir if args.flair_subdir else _pick_dir(raw_dir, "FLAIR_synthstrip", "FLAIR")

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
        sample_id = t1.sample_id  # e.g. "C005_V1"
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
