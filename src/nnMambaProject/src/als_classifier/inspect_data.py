"""Sanity-check the processed data folder.

Logs class balance, shapes, voxel sizes, and intensity ranges.
Run from project root:
    python src/als_classifier/inspect_data.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from als_classifier.dataset import LABEL_RE, list_subject_folders  # noqa: E402

DATA_ROOT = ROOT / "Data" / "processed"
MODALITIES = ("T1", "T2", "FLAIR")


def main() -> None:
    folders = list_subject_folders(DATA_ROOT)
    print(f"Found {len(folders)} subject-visit folders under {DATA_ROOT}\n")

    labels = []
    missing_label = []
    for f in folders:
        m = LABEL_RE.search(f.name)
        if m:
            labels.append("ALS" if m.group(1) == "P" else "Control")
        else:
            missing_label.append(f.name)

    print("Class counts (visit-level):", Counter(labels))
    if missing_label:
        print(f"WARNING: {len(missing_label)} folders had unparseable names:")
        for n in missing_label[:10]:
            print(f"  {n}")

    print("\n--- inspecting first 3 subjects ---")
    for sample in folders[:3]:
        print(f"\n{sample.name}")
        for mod in MODALITIES:
            f = sample / f"{sample.name}_{mod}.nii.gz"
            if not f.exists():
                print(f"  {mod:6s} MISSING")
                continue
            img = nib.load(str(f))
            arr = img.get_fdata()
            voxel = tuple(round(v, 2) for v in img.header.get_zooms())
            print(
                f"  {mod:6s} shape={arr.shape}  voxel={voxel}  "
                f"range=[{arr.min():.0f}, {arr.max():.0f}]"
            )


if __name__ == "__main__":
    main()
