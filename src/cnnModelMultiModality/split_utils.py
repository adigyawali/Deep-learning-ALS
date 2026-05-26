"""
Deprecated. Splits are now produced by `src/splits.py` and persisted to
`splits.json` (see cnnModelMultiModality/train.py).

This shim re-exports the canonical API so any straggler imports keep working.
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))

from splits import indices_from_split, make_subject_splits  # noqa: F401,E402


def split_indices_by_subject(samples, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    """
    Back-compat shim. Builds an in-memory stratified split from raw sample dicts
    and returns (train_idx, val_idx, test_idx). Prefer reading splits.json.
    """
    import re
    from splits import SampleMeta, extract_site

    meta: list[SampleMeta] = []
    for s in samples:
        sample_id = s["id"] if "id" in s else s["sample_id"]
        subject_id = s.get("subject_id")
        if not subject_id:
            m = re.search(r"(?:^|_)([CP]\d{3,})(?:_|$)", sample_id, flags=re.IGNORECASE)
            subject_id = m.group(1).upper() if m else sample_id.split("_")[0].upper()
        meta.append(SampleMeta(
            sample_id=sample_id,
            subject_id=subject_id,
            label=float(s["label"]),
            site=s.get("site") or extract_site(sample_id),
        ))
    splits = make_subject_splits(meta, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    return (
        indices_from_split(meta, splits, "train"),
        indices_from_split(meta, splits, "val"),
        indices_from_split(meta, splits, "test"),
    )
