"""
Read-only diagnostic: how does the preprocessing parser see Data/raw?

Run this before a long preprocessing job, and any time a new cohort or a new
export convention shows up. It uses the *same* folder search and the *same*
modality regexes as the real pipeline, so if a file is listed as UNMATCHED here
it will be silently skipped by preprocessing too.

    python scripts/check_raw_layout.py                 # Data/raw
    python scripts/check_raw_layout.py /path/to/raw

For each modality it prints which folder was chosen, the distinct modality
tokens found (e.g. T1w10, PDT2, FLAIR3D), which cohorts and timepoint forms are
present, and every filename the parser could not read. It then runs the real
pairing and prints the triplet count.

Nothing is written or modified.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from als.data.preprocessing import (
    _FL_RE,
    _FLAIR_SUBDIRS,
    _T1_RE,
    _T1_SUBDIRS,
    _T2_RE,
    _T2_SUBDIRS,
    _parse_scan,
    _pick_dir,
    find_triplets,
)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    raw = Path(argv[0]) if argv else Path("Data/raw")
    if not raw.is_dir():
        print(f"ERROR: {raw} is not a directory.", file=sys.stderr)
        return 1

    modalities = [
        ("T1", _pick_dir(raw, _T1_SUBDIRS), _T1_RE),
        ("T2", _pick_dir(raw, _T2_SUBDIRS), _T2_RE),
        ("FLAIR", _pick_dir(raw, _FLAIR_SUBDIRS), _FL_RE),
    ]

    print(f"Raw root: {raw.resolve()}\n")
    total_unmatched = 0
    for name, directory, regex in modalities:
        if not directory.is_dir():
            print(f"=== {name}: {directory.name} — MISSING ===\n")
            continue
        files = sorted(directory.glob("*.nii.gz"))
        tokens: Counter[str] = Counter()
        cohorts: Counter[str] = Counter()
        timepoints: Counter[str] = Counter()
        reruns = 0
        unmatched: list[str] = []
        for path in files:
            scan = _parse_scan(path, regex)
            if scan is None:
                unmatched.append(path.name)
                continue
            stem = path.name[: -len(".nii.gz")]
            tokens[regex.search(stem + "_").group(0).strip("_")] += 1
            cohorts[scan.subject_id.split("_")[0]] += 1
            timepoints["V# (visit)" if scan.visit.upper().startswith("V") else "#M (months)"] += 1
            reruns += scan.run > 0

        print(f"=== {name}: {directory.name} — {len(files)} files ===")
        print(f"  modality tokens : {dict(tokens)}")
        print(f"  cohorts         : {dict(cohorts)}")
        print(f"  timepoint form  : {dict(timepoints)}")
        print(f"  rerun-tagged    : {reruns}")
        print(f"  UNMATCHED       : {len(unmatched)}")
        for u in unmatched[:15]:
            print(f"      {u}")
        if len(unmatched) > 15:
            print(f"      ... and {len(unmatched) - 15} more")
        print()
        total_unmatched += len(unmatched)

    print("=== pairing ===")
    find_triplets(*[d for _, d, _ in modalities])
    if total_unmatched:
        print(f"\nNOTE: {total_unmatched} file(s) the parser could not read. Ignore entries like "
              f".DS_Store or a pure-PD scan; anything else means a naming convention the "
              f"regexes do not cover yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
