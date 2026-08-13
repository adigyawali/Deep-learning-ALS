"""Stage: inspect or re-freeze the cross-validation split.

The split lives in one file (``cross_validation.splits_file``, by default the
project-root ``splits.json``). Once written it never changes on its own — every
run, every stage, both models, and every ``--output-dir`` read that same file.
That is what makes two independent experiments comparable: they are scored on
the *identical* held-out test subjects, so a difference in their numbers is a
difference in the models rather than in the luck of the draw.

    python experiment.py --model cnn_nnmamba splits              # show the current split
    python experiment.py --model cnn_nnmamba splits --reshuffle  # draw a NEW split, then freeze it
    python experiment.py --model cnn_nnmamba splits --reshuffle --seed 7   # a named split

Why ``--reshuffle`` needs a new seed: the generator is deterministic, so
re-running it with the configured seed reproduces the *same* folds. Reshuffling
therefore draws a fresh seed from system entropy and records it in the file, so
the new split is itself reproducible — pass that seed back via ``--seed`` to
recreate it exactly.

This stage is model-independent (like ``preprocess``): running it under either
``--model`` produces the same file.
"""

from __future__ import annotations

from typing import Optional

from .. import sanity
from ..config import get
from ..data.volume_dataset import VolumeDataset
from ..paths import DEFAULT_DATA_DIR, RunPaths
from ..splits import (
    coverage,
    make_splits_from_explicit,
    make_subject_splits,
    new_seed,
    read_splits,
    write_splits,
)


def _load_samples(cfg: dict):
    """Scan Data/processed for sample metadata (no image data is read)."""
    data_dir = get(cfg, "data", "data_dir") or DEFAULT_DATA_DIR
    dataset = VolumeDataset(
        data_dir, return_mode="stack",
        target_shape=tuple(get(cfg, "data", "target_shape", default=[128, 128, 128])),
        transform=False,
    )
    return dataset.to_sample_meta()


def _report(splits: dict, samples, splits_path) -> None:
    print(f"[splits] file   : {splits_path}")
    print(f"[splits] mode   : {splits.get('mode')}   seed: {splits.get('seed')}   "
          f"stratify_by_site: {splits.get('stratify_by_site')}")
    sanity.split_report(splits)
    cov = coverage(samples, splits)
    if not cov["missing"] and not cov["unassigned"]:
        n_sub = len({s.subject_id for s in samples})
        print(f"[splits] matches the data exactly ({n_sub} subjects, {len(samples)} samples).")
    else:
        if cov["missing"]:
            print(f"[splits] {len(cov['missing'])} subject(s) in the split are missing from the "
                  f"data, e.g. {cov['missing'][:5]}")
        if cov["unassigned"]:
            print(f"[splits] {len(cov['unassigned'])} subject(s) in the data are not in the split "
                  f"and will be UNUSED, e.g. {cov['unassigned'][:5]}")
        print("[splits] Run `splits --reshuffle` to re-freeze against the current data.")


def run(cfg: dict, paths: RunPaths, device=None, *,
        reshuffle: bool = False, seed: Optional[int] = None) -> None:
    splits_path = paths.splits_path
    samples = _load_samples(cfg)
    if not samples:
        print("[splits] No processed samples found — run preprocess first.")
        return

    cv = cfg.get("cross_validation") or {}
    mode = str(cv.get("mode", "auto")).lower()

    if not reshuffle:
        if not splits_path.exists():
            print(f"[splits] No split yet at {splits_path}.")
            print("[splits] Create one with: "
                  "python experiment.py --model <model> splits --reshuffle")
            return
        _report(read_splits(splits_path), samples, splits_path)
        return

    # ── reshuffle ───────────────────────────────────────────────────────────
    if mode == "explicit":
        print("[splits] cross_validation.mode is 'explicit' — the folds are the patient-ID "
              "lists in config.yaml, so there is nothing to reshuffle. Rebuilding from "
              "config instead.")
        splits = make_splits_from_explicit(
            samples,
            test_subjects=cv.get("test_subjects", []),
            folds=cv.get("folds", []),
            seed=int(cv.get("seed", cfg.get("seed", 42))),
        )
    else:
        chosen = int(seed) if seed is not None else new_seed()
        splits = make_subject_splits(
            samples,
            n_folds=int(cv.get("n_folds", get(cfg, "split", "n_folds", default=5))),
            test_ratio=float(cv.get("test_ratio", get(cfg, "split", "test_ratio", default=0.2))),
            seed=chosen,
            stratify_by_site=bool(cv.get("stratify_by_site", True)),
        )

    previous = read_splits(splits_path) if splits_path.exists() else None
    write_splits(splits_path, splits)
    if previous is not None:
        moved = len(set(splits["test_subjects"]) ^ set(previous.get("test_subjects", [])))
        print(f"[splits] reshuffled — {moved} subject(s) changed place in the test set "
              f"(old seed {previous.get('seed')} → new seed {splits['seed']}).")
    print()
    _report(splits, samples, splits_path)
    print()
    print(f"[splits] FROZEN. Every run now uses this split until you reshuffle again.")
    print(f"[splits] To recreate it exactly: splits --reshuffle --seed {splits['seed']}")
