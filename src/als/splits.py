"""
Canonical subject-level data splitting (held-out test + 5-fold CV).

Every stage of every model (CNN training, feature extraction, ViT training,
nnMamba training, evaluation, Grad-CAM) reads the same ``splits.json`` artifact
so there is no cross-stage leakage and the two models are compared on the
*identical* partition. Produce it once with ``write_splits``, reuse forever.

Split semantics:
  - One stratified, subject-level **held-out TEST set** (``test_ratio``, default
    20%) is carved out once and never used for training or threshold tuning.
  - The remaining subjects form ``n_folds`` (default 5) **stratified folds**.
    Fold ``k`` supplies the validation set; the other ``n_folds-1`` folds are
    that fold's training set. So each fold is a train/val split and there are
    ``n_folds`` of them, all sharing the one fixed test set.
  - Multi-visit subjects (V1/V2/V3, run-02 reruns) are grouped by subject_id so
    every visit of a subject lands in the same partition (no patient leakage).
  - Stratification is by label (control / patient), and by site when at least
    two sites are present and each site has enough subjects per class. Within
    each stratification bucket the CV-pool subjects are distributed round-robin
    across folds, which keeps class balance near-identical across folds.
  - Splits are reproducible: ``splits.json`` records the seed, ratios, fold
    count, and per-fold / test subject lists, so the file itself is the contract.

Folder/file naming assumptions:
  - Subject IDs look like ``C005``, ``P110`` (one letter + digits).
  - Folder names look like ``C005_V1``, ``P110_V2_run-02``, or the longer raw
    form ``CALSNIC2_EDM_C005_V1``. ``extract_subject_id`` handles both.
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

def _peel_test_and_fold(
    group: List[str], n_folds: int, test_ratio: float, rng: random.Random
) -> tuple[List[str], List[List[str]]]:
    """Split one stratification bucket into (test_subjects, [fold0..fold_{n-1}]).

    A ``test_ratio`` fraction is peeled off for the shared held-out test set;
    the remaining "CV pool" subjects are distributed round-robin across the
    ``n_folds`` folds so every fold gets a near-equal share of this bucket's
    class (and site). Round-robin keeps per-fold class balance tight even for
    small buckets.
    """
    ids = list(group)
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(round(n * test_ratio))
    # Never consume the whole bucket for test — the CV pool must stay non-empty.
    if n >= 2:
        n_test = min(n_test, n - 1)
    test = ids[:n_test]
    pool = ids[n_test:]
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    for i, sid in enumerate(pool):
        folds[i % n_folds].append(sid)
    return test, folds


def make_subject_splits(
    samples: Sequence[SampleMeta],
    *,
    n_folds: int = 5,
    test_ratio: float = 0.2,
    seed: int = 42,
    stratify_by_site: bool = True,
) -> dict:
    """
    Build subject-level stratified splits: one held-out test set + ``n_folds``
    train/val folds over the remaining subjects.

    Returns
    -------
    dict with keys:
      - 'seed', 'n_folds', 'test_ratio', 'stratify_by_site'
      - 'test_subjects', 'test_samples'
      - 'folds'  : list of ``n_folds`` dicts, each with
                   'fold', 'train_subjects', 'val_subjects',
                   'train_samples', 'val_samples'
      - 'class_counts' : {'test': {...}, 'folds': [{'train':.., 'val':..}, ...]}
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"test_ratio must satisfy 0<=test_ratio<1, got {test_ratio}.")

    # Group subject IDs by (label, site). Site is optional.
    subject_to_label: dict[str, float] = {}
    subject_to_site: dict[str, Optional[str]] = {}
    subject_to_samples: dict[str, list[str]] = defaultdict(list)

    for s in samples:
        # First sample wins for label/site (they must agree across visits anyway).
        subject_to_label.setdefault(s.subject_id, s.label)
        subject_to_site.setdefault(s.subject_id, s.site)
        subject_to_samples[s.subject_id].append(s.sample_id)

    # Decide whether to stratify by site: only if every (label, site) bucket has
    # enough subjects to fill the test set plus every fold. Otherwise fall back
    # to label-only stratification.
    use_site = False
    if stratify_by_site and any(v is not None for v in subject_to_site.values()):
        per_bucket = defaultdict(list)
        for sid, lab in subject_to_label.items():
            per_bucket[(lab, subject_to_site[sid])].append(sid)
        use_site = all(len(v) >= n_folds + 1 for v in per_bucket.values()) and len(per_bucket) > 2

    buckets: dict[tuple, list[str]] = defaultdict(list)
    for sid, lab in subject_to_label.items():
        key = (lab, subject_to_site[sid]) if use_site else (lab,)
        buckets[key].append(sid)

    rng = random.Random(seed)
    test_subjects: list[str] = []
    fold_subjects: list[list[str]] = [[] for _ in range(n_folds)]

    # Iterate in a deterministic order regardless of dict insertion.
    for key in sorted(buckets.keys(), key=lambda k: tuple(str(x) for x in k)):
        test_b, folds_b = _peel_test_and_fold(buckets[key], n_folds, test_ratio, rng)
        test_subjects.extend(test_b)
        for k in range(n_folds):
            fold_subjects[k].extend(folds_b[k])

    test_subjects.sort()
    for k in range(n_folds):
        fold_subjects[k].sort()

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

    folds: list[dict] = []
    for k in range(n_folds):
        val = fold_subjects[k]
        train = sorted(sid for j in range(n_folds) if j != k for sid in fold_subjects[j])
        folds.append({
            "fold": k,
            "train_subjects": train,
            "val_subjects": list(val),
            "train_samples": _samples_in(train),
            "val_samples": _samples_in(val),
        })

    return {
        "seed": seed,
        "n_folds": n_folds,
        "test_ratio": test_ratio,
        "stratify_by_site": use_site,
        "test_subjects": test_subjects,
        "test_samples": _samples_in(test_subjects),
        "folds": folds,
        "class_counts": {
            "test": _class_counts(test_subjects),
            "folds": [
                {"train": _class_counts(f["train_subjects"]), "val": _class_counts(f["val_subjects"])}
                for f in folds
            ],
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


def n_folds_in(splits: dict) -> int:
    """Number of CV folds recorded in a splits dict."""
    return int(splits.get("n_folds", len(splits.get("folds", []))))


def indices_from_split(
    samples: Sequence[SampleMeta], splits: dict, kind: str, fold: Optional[int] = None
) -> list[int]:
    """
    Return integer indices into `samples` that belong to `kind`.

    - ``kind == "test"``  → the one shared held-out test set (``fold`` ignored).
    - ``kind in {"train", "val"}`` → that partition of fold ``fold`` (required).

    Splits are matched by subject_id (not sample_id), so adding visits to a
    subject later still lands them in the correct partition without rewriting
    splits.json.
    """
    if kind == "test":
        subjects = set(splits["test_subjects"])
    elif kind in ("train", "val"):
        if fold is None:
            raise ValueError(f"kind={kind!r} requires a fold index.")
        folds = splits.get("folds", [])
        if not (0 <= fold < len(folds)):
            raise ValueError(f"fold {fold} out of range (n_folds={len(folds)}).")
        subjects = set(folds[fold][f"{kind}_subjects"])
    else:
        raise ValueError(f"kind must be one of train/val/test, got {kind!r}")
    return [i for i, s in enumerate(samples) if s.subject_id in subjects]


def load_or_build_splits(
    samples: Sequence[SampleMeta],
    splits_path: Path | str,
    *,
    n_folds: int = 5,
    test_ratio: float = 0.2,
    seed: int = 42,
    stratify_by_site: bool = True,
) -> dict:
    """Read `splits_path` if present, else compute the CV split and write it.

    The first stage that runs creates the shared file; every other stage, every
    fold, and both models reuse it. This is the single guarantee that
    ``cnn_vit`` and ``cnn_nnmamba`` are trained and evaluated on the same folds
    and the same held-out test subjects.
    """
    splits_path = Path(splits_path)
    if splits_path.exists():
        return read_splits(splits_path)
    splits = make_subject_splits(
        samples, n_folds=n_folds, test_ratio=test_ratio,
        seed=seed, stratify_by_site=stratify_by_site,
    )
    write_splits(splits_path, splits)
    return splits


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
