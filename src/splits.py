"""
Canonical subject-level data splitting.

Every stage of the pipeline (CNN training, feature extraction, ViT training,
evaluation, Grad-CAM) must read the same splits.json artifact to guarantee no
cross-stage leakage. Produce it once with `write_splits`, reuse forever.

Split semantics:
  - Multi-visit subjects (V1/V2/V3, run-02 reruns) are grouped by subject_id
    so every visit of a subject lands in the same split.
  - Stratification is by label (control / patient), and by site when at least
    two sites are present and each site has enough subjects per class.
  - Splits are reproducible: `splits.json` records the seed, ratios, and per-
    split subject lists, so the file itself is the contract.

Folder/file naming assumptions:
  - Subject IDs look like `C005`, `P110` (one letter + digits).
  - Folder names look like `C005_V1`, `P110_V2_run-02`, or the longer raw
    form `CALSNIC2_EDM_C005_V1`. `extract_subject_id` handles both.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

_SUBJECT_ID_RE = re.compile(r"(?:^|_)([CP]\d{3,})(?:_|$)", flags=re.IGNORECASE)
_SITE_RE = re.compile(r"CALSNIC\d*_([A-Z]{2,4})_", flags=re.IGNORECASE)


@dataclass(frozen=True)
class SampleMeta:
    """Minimal per-sample metadata used by the splitter."""
    sample_id: str
    subject_id: str
    label: float          # 0.0 = control, 1.0 = patient
    site: Optional[str] = None


# ─── ID + site extraction ──────────────────────────────────────────────────

def extract_subject_id(name: str, fallback: Optional[str] = None) -> str:
    """
    Return the C### / P### subject token from a folder or file name.

    Examples
    --------
    >>> extract_subject_id("C005_V1")
    'C005'
    >>> extract_subject_id("CALSNIC2_EDM_P110_V2_run-02")
    'P110'
    >>> extract_subject_id("anything_else", fallback="X1")
    'X1'
    """
    if fallback:
        return fallback.upper()
    match = _SUBJECT_ID_RE.search(name)
    if match:
        return match.group(1).upper()
    # Last resort: first underscore-separated token, upper-cased.
    return name.split("_")[0].upper()


def extract_site(name: str) -> Optional[str]:
    """Return the CALSNIC site code (e.g. EDM) if present, else None."""
    match = _SITE_RE.search(name)
    return match.group(1).upper() if match else None


def label_from_subject_id(subject_id: str) -> float:
    """C### → 0.0 (control), P### → 1.0 (patient). Raises on anything else."""
    sid = subject_id.upper()
    if sid.startswith("C"):
        return 0.0
    if sid.startswith("P"):
        return 1.0
    raise ValueError(f"Cannot infer label from subject_id={subject_id!r}")


# ─── Splitter ──────────────────────────────────────────────────────────────

def _split_group(group: List[str], ratios: tuple[float, float, float], rng: random.Random) -> tuple[List[str], List[str], List[str]]:
    """Shuffle a flat list of subject IDs and chop it by `ratios`."""
    ids = list(group)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    # Ensure each split has at least 1 subject when n >= 3.
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n - n_train - n_val)
        while n_train + n_val + n_test > n:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
        return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:n_train + n_val + n_test]

    # Tiny groups: best-effort partition.
    n_train = max(1, n_train) if n > 0 else 0
    n_val = min(n - n_train, max(0, n_val))
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def make_subject_splits(
    samples: Sequence[SampleMeta],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    stratify_by_site: bool = True,
) -> dict:
    """
    Build subject-level stratified splits.

    Returns
    -------
    dict with keys:
      - 'seed', 'train_ratio', 'val_ratio'
      - 'train_subjects', 'val_subjects', 'test_subjects' (lists of subject IDs)
      - 'train_samples', 'val_samples', 'test_samples'   (lists of sample IDs)
      - 'class_counts'                                    (per split, per label)
    """
    if not (0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("Ratios must satisfy 0<train<1, 0<=val<1, train+val<1.")

    # Group subject IDs by (label, site). Site is optional.
    subject_to_label: dict[str, float] = {}
    subject_to_site: dict[str, Optional[str]] = {}
    subject_to_samples: dict[str, list[str]] = defaultdict(list)

    for s in samples:
        # First sample wins for label/site (they must agree across visits anyway).
        subject_to_label.setdefault(s.subject_id, s.label)
        subject_to_site.setdefault(s.subject_id, s.site)
        subject_to_samples[s.subject_id].append(s.sample_id)

    # Decide whether to stratify by site: only if every (label, site) bucket
    # has at least 3 subjects (so each split can be non-empty). Otherwise fall
    # back to label-only stratification.
    use_site = False
    if stratify_by_site and any(v is not None for v in subject_to_site.values()):
        per_bucket = defaultdict(list)
        for sid, lab in subject_to_label.items():
            per_bucket[(lab, subject_to_site[sid])].append(sid)
        use_site = all(len(v) >= 3 for v in per_bucket.values()) and len(per_bucket) > 2

    buckets: dict[tuple, list[str]] = defaultdict(list)
    for sid, lab in subject_to_label.items():
        key = (lab, subject_to_site[sid]) if use_site else (lab,)
        buckets[key].append(sid)

    rng = random.Random(seed)
    train_subjects: list[str] = []
    val_subjects: list[str] = []
    test_subjects: list[str] = []

    # Iterate in a deterministic order regardless of dict insertion.
    for key in sorted(buckets.keys(), key=lambda k: tuple(str(x) for x in k)):
        tr, va, te = _split_group(buckets[key], (train_ratio, val_ratio, 1.0 - train_ratio - val_ratio), rng)
        train_subjects.extend(tr)
        val_subjects.extend(va)
        test_subjects.extend(te)

    train_subjects.sort()
    val_subjects.sort()
    test_subjects.sort()

    def _samples_in(subset: Iterable[str]) -> list[str]:
        out: list[str] = []
        for sid in subset:
            out.extend(subject_to_samples.get(sid, []))
        return sorted(out)

    def _class_counts(subset: Iterable[str]) -> dict[str, int]:
        c = {"control": 0, "patient": 0}
        for sid in subset:
            c["patient" if subject_to_label[sid] == 1.0 else "control"] += 1
        return c

    return {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "stratify_by_site": use_site,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "train_samples": _samples_in(train_subjects),
        "val_samples": _samples_in(val_subjects),
        "test_samples": _samples_in(test_subjects),
        "class_counts": {
            "train": _class_counts(train_subjects),
            "val": _class_counts(val_subjects),
            "test": _class_counts(test_subjects),
        },
    }


# ─── IO ────────────────────────────────────────────────────────────────────

def write_splits(path: Path | str, splits: dict) -> None:
    """Atomic write of `splits.json`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(splits, indent=2, sort_keys=True))
    tmp.replace(path)


def read_splits(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def indices_from_split(samples: Sequence[SampleMeta], splits: dict, kind: str) -> list[int]:
    """
    Return integer indices into `samples` that belong to `kind ∈ {train,val,test}`.

    Splits are matched by subject_id (not sample_id), so adding visits to a
    subject later still lands them in the correct split without rewriting
    splits.json.
    """
    key = {"train": "train_subjects", "val": "val_subjects", "test": "test_subjects"}.get(kind)
    if key is None:
        raise ValueError(f"kind must be one of train/val/test, got {kind!r}")
    subjects = set(splits[key])
    return [i for i, s in enumerate(samples) if s.subject_id in subjects]


# ─── Convenience for callers that hand in raw dicts ────────────────────────

def make_meta_from_dicts(records: Iterable[dict]) -> list[SampleMeta]:
    """Build SampleMeta from records with id/subject_id/label/site keys."""
    out: list[SampleMeta] = []
    for r in records:
        sample_id = r["id"] if "id" in r else r["sample_id"]
        subject_id = r.get("subject_id") or extract_subject_id(sample_id)
        label = float(r["label"])
        site = r.get("site") or extract_site(sample_id)
        out.append(SampleMeta(sample_id=sample_id, subject_id=subject_id, label=label, site=site))
    return out
