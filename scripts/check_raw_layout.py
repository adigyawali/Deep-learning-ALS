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
from als.splits import extract_site, label_from_subject_id


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
    triplets = find_triplets(*[d for _, d, _ in modalities])

    # Splits are subject-level, so the subject count — not the sample count — is
    # what decides whether 5-fold CV is stable. Report it before any ANTs work.
    if triplets:
        by_subject: dict[str, int] = Counter(t1.subject_id for t1, _, _ in triplets)
        labels = Counter()
        sites = Counter()
        for t1, _, _ in triplets:
            sites[extract_site(t1.sample_id) or "UNK"] += 1
        for sid in by_subject:
            try:
                labels["patient" if label_from_subject_id(sid) == 1.0 else "control"] += 1
            except ValueError:
                labels["UNKNOWN"] += 1
        cohort_subjects = Counter(sid.split("_")[0] for sid in by_subject)
        visits = Counter(by_subject.values())

        print()
        print("=== subjects (what the splitter actually sees) ===")
        print(f"  samples          : {len(triplets)}")
        print(f"  unique subjects  : {len(by_subject)}")
        print(f"  labels (subject) : {dict(labels)}")
        # Sample-level balance can be much more skewed than subject-level when one
        # class is followed longitudinally and the other is scanned once. This is
        # what the loss and pos_weight actually see.
        sample_labels: Counter[str] = Counter()
        for t1, _, _ in triplets:
            try:
                sample_labels["patient" if label_from_subject_id(t1.subject_id) else "control"] += 1
            except ValueError:
                sample_labels["UNKNOWN"] += 1
        print(f"  labels (sample)  : {dict(sample_labels)}")
        print(f"  cohorts          : {dict(cohort_subjects)}")
        print(f"  sites (samples)  : {dict(sites)}")
        print("  timepoints/subject: "
              + ", ".join(f"{n} subj x {k} tp" for k, n in sorted(visits.items())))
        n_sub = len(by_subject)
        if n_sub:
            print(f"  -> a 20% test set is ~{round(n_sub * 0.2)} subjects; "
                  f"each of 5 folds validates on ~{round(n_sub * 0.8 / 5)}")

        # Will `stratify_by_site: true` actually engage? Mirror the rule in
        # splits.make_subject_splits: every (label, site) bucket needs at least
        # n_folds+1 subjects, and there must be more than 2 buckets. Otherwise the
        # splitter quietly falls back to label-only stratification, which with
        # several scanners can leave folds site-imbalanced.
        n_folds = 5
        subject_site: dict[str, str] = {}
        subject_label: dict[str, str] = {}
        for t1, _, _ in triplets:
            subject_site.setdefault(t1.subject_id, extract_site(t1.sample_id) or "UNK")
            try:
                subject_label.setdefault(
                    t1.subject_id, "patient" if label_from_subject_id(t1.subject_id) else "control")
            except ValueError:
                subject_label.setdefault(t1.subject_id, "UNKNOWN")
        buckets: Counter[tuple[str, str]] = Counter(
            (subject_label[s], subject_site[s]) for s in by_subject)
        print()
        print("=== site stratification (subjects per label x site) ===")
        for (lab, site), n in sorted(buckets.items()):
            flag = "" if n >= n_folds + 1 else f"   <-- under {n_folds + 1}"
            print(f"  {site:<6} {lab:<8} {n:>4}{flag}")
        engages = len(buckets) > 2 and all(n >= n_folds + 1 for n in buckets.values())
        print(f"  -> with n_folds={n_folds}, stratify_by_site would "
              f"{'ENGAGE' if engages else 'FALL BACK to label-only'}")
        if not engages:
            print("     (folds may end up site-imbalanced; see notes in config.yaml)")

    if total_unmatched:
        print(f"\nNOTE: {total_unmatched} file(s) the parser could not read. Ignore entries like "
              f".DS_Store or a pure-PD scan; anything else means a naming convention the "
              f"regexes do not cover yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
