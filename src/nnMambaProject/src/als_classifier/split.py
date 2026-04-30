"""Subject-aware train/val/test splitting.

Critical: a single subject's visits must NEVER be split across train and test.
Otherwise the model effectively sees the test patient during training, which
is data leakage and inflates apparent performance.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from sklearn.model_selection import train_test_split


# Captures the subject ID, e.g. 'P110' or 'C045', from the start of the folder name.
SUBJECT_RE = re.compile(r"^[A-Z0-9]+_[A-Z0-9]+_([PC]\d+)_")


def extract_subject_id(folder_name: str) -> str:
    m = SUBJECT_RE.match(folder_name)
    if not m:
        raise ValueError(f"Cannot parse subject ID from {folder_name!r}")
    return m.group(1)


def split_by_subject(
    folders: Iterable[Path],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split folders so all visits of one subject land in the same split.

    Returns (train_folders, val_folders, test_folders), each a list of Path objects.
    """
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError(
            f"val_frac ({val_frac}) and test_frac ({test_frac}) must each be in (0,1) "
            f"and sum to less than 1."
        )

    by_subject: dict[str, list[Path]] = defaultdict(list)
    for f in folders:
        by_subject[extract_subject_id(f.name)].append(f)

    subjects = sorted(by_subject)
    labels = [1 if s.startswith("P") else 0 for s in subjects]

    train_s, temp_s, _, temp_y = train_test_split(
        subjects,
        labels,
        test_size=val_frac + test_frac,
        stratify=labels,
        random_state=seed,
    )
    rel_test = test_frac / (val_frac + test_frac)
    val_s, test_s = train_test_split(
        temp_s,
        test_size=rel_test,
        stratify=temp_y,
        random_state=seed,
    )

    def collect(subject_ids: list[str]) -> list[Path]:
        return [f for s in subject_ids for f in by_subject[s]]

    return collect(train_s), collect(val_s), collect(test_s)
