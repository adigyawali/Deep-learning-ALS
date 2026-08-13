"""The frozen-split workflow: one canonical file, stable until you reshuffle."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from als import config as cfgmod
from als.paths import build_run_paths
from als.splits import SampleMeta, coverage, make_subject_splits, new_seed, read_splits, warn_if_stale
from als.stages import make_splits


def _subject(root: Path, sample_id: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for m in ("T1", "T2", "FLAIR"):
        p = root / sample_id / f"{sample_id}_{m}.nii.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(rng.normal(size=(8, 8, 8)).astype(np.float32).clip(min=0),
                                 np.eye(4)), str(p))


@pytest.fixture
def workspace(tmp_path: Path):
    """A processed dir with 30 subjects x 2 timepoints, plus a cfg pointing at it."""
    data = tmp_path / "processed"
    for i in range(1, 31):
        lab = "C" if i <= 14 else "P"
        for tp in ("00M", "04M"):
            _subject(data, f"CAPTURE_EDM_{lab}{i:03d}_{tp}", seed=i)
    cfg = {
        "model": "cnn_nnmamba",
        "data": {"data_dir": str(data), "target_shape": [8, 8, 8]},
        "cross_validation": {"mode": "auto", "n_folds": 3, "test_ratio": 0.2},
    }
    paths = build_run_paths("cnn_nnmamba", tmp_path / "runs",
                            splits_file=tmp_path / "splits.json").ensure()
    return cfg, paths


def _test_set(paths) -> set[str]:
    return set(read_splits(paths.splits_path)["test_subjects"])


# ── the canonical location ─────────────────────────────────────────────────

def test_splits_file_is_shared_across_output_dirs(tmp_path: Path):
    """Two experiments with different --output-dir must resolve to ONE split file."""
    shared = tmp_path / "splits.json"
    a = build_run_paths("cnn_nnmamba", tmp_path / "runs_mixup", splits_file=shared)
    b = build_run_paths("cnn_vit", tmp_path / "runs_nomixup", splits_file=shared)
    assert a.splits_path == b.splits_path == shared
    assert a.root != b.root                      # outputs still separate


def test_without_splits_file_each_output_dir_gets_its_own(tmp_path: Path):
    a = build_run_paths("cnn_nnmamba", tmp_path / "runs_a")
    b = build_run_paths("cnn_nnmamba", tmp_path / "runs_b")
    assert a.splits_path != b.splits_path        # the old, per-experiment behaviour


def test_fold_paths_inherit_the_shared_split(tmp_path: Path):
    p = build_run_paths("cnn_nnmamba", tmp_path / "runs", splits_file=tmp_path / "s.json")
    assert p.fold(3).splits_path == p.splits_path


# ── freeze / stay / reshuffle ──────────────────────────────────────────────

def test_show_before_any_split_exists_does_not_create_one(workspace, capsys):
    cfg, paths = workspace
    make_splits.run(cfg, paths, reshuffle=False)
    assert not paths.splits_path.exists()
    assert "No split yet" in capsys.readouterr().out


def test_reshuffle_creates_and_freezes(workspace):
    cfg, paths = workspace
    make_splits.run(cfg, paths, reshuffle=True)
    assert paths.splits_path.exists()
    assert _test_set(paths)


def test_split_is_stable_across_repeated_runs(workspace):
    """The whole point: showing it, or running other stages, never re-draws it."""
    cfg, paths = workspace
    make_splits.run(cfg, paths, reshuffle=True)
    frozen = _test_set(paths)
    for _ in range(3):
        make_splits.run(cfg, paths, reshuffle=False)
    assert _test_set(paths) == frozen


def test_reshuffle_actually_changes_the_split(workspace):
    cfg, paths = workspace
    make_splits.run(cfg, paths, reshuffle=True, seed=1)
    first = _test_set(paths)
    make_splits.run(cfg, paths, reshuffle=True, seed=2)
    assert _test_set(paths) != first


def test_reshuffle_with_a_seed_is_reproducible(workspace):
    """The recorded seed recreates the split exactly — that is why it is recorded."""
    cfg, paths = workspace
    make_splits.run(cfg, paths, reshuffle=True)
    original = _test_set(paths)
    seed = read_splits(paths.splits_path)["seed"]

    make_splits.run(cfg, paths, reshuffle=True, seed=seed + 1)      # move away
    assert _test_set(paths) != original
    make_splits.run(cfg, paths, reshuffle=True, seed=seed)          # come back
    assert _test_set(paths) == original


def test_reshuffle_without_a_seed_draws_a_new_one(workspace):
    cfg, paths = workspace
    make_splits.run(cfg, paths, reshuffle=True)
    seeds = {read_splits(paths.splits_path)["seed"]}
    for _ in range(4):
        make_splits.run(cfg, paths, reshuffle=True)
        seeds.add(read_splits(paths.splits_path)["seed"])
    assert len(seeds) > 1, "reshuffle must not keep reusing the configured seed"


def test_new_seed_is_in_range():
    for _ in range(20):
        assert 1 <= new_seed() < 2**31 - 1


# ── staleness guard ────────────────────────────────────────────────────────

def _samples(prefix: str = "CALSNIC_") -> list[SampleMeta]:
    return [SampleMeta(f"{prefix}{p}{i:03d}_V1", f"{prefix}{p}{i:03d}",
                       0.0 if p == "C" else 1.0, "EDM")
            for p in ("C", "P") for i in range(1, 11)]


def test_coverage_clean_when_split_matches_data():
    s = _samples()
    sp = make_subject_splits(s, n_folds=2, test_ratio=0.2, seed=42)
    assert coverage(s, sp) == {"missing": [], "unassigned": []}
    assert warn_if_stale(s, sp, "x.json") is False


def test_stale_split_from_old_naming_is_flagged(capsys):
    """The exact failure mode that produced empty folds: bare IDs vs qualified ones."""
    s = _samples()
    sp = make_subject_splits(s, n_folds=2, test_ratio=0.2, seed=42)
    old = dict(sp)
    old["test_subjects"] = [x.split("_")[1] for x in sp["test_subjects"]]
    old["folds"] = [{**f, "val_subjects": [x.split("_")[1] for x in f["val_subjects"]]}
                    for f in sp["folds"]]
    assert warn_if_stale(s, old, "runs/splits.json") is True
    out = capsys.readouterr().out
    assert "NOT ONE listed subject matches" in out
    assert "--reshuffle" in out


def test_added_subjects_are_reported_as_unused(capsys):
    s = _samples()
    sp = make_subject_splits(s, n_folds=2, test_ratio=0.2, seed=42)
    grown = s + [SampleMeta("CALSNIC_P099_V1", "CALSNIC_P099", 1.0, "EDM")]
    assert warn_if_stale(grown, sp, "runs/splits.json") is True
    out = capsys.readouterr().out
    assert "NOT in the split" in out and "CALSNIC_P099" in out
    # Only the new subject is flagged — the other 20 still match.
    assert "1 subject(s) in the data" in out


def test_smoke_config_does_not_pin_the_shared_split():
    """--smoke must never touch the real frozen split."""
    cfg = {"model": "cnn_nnmamba", "cnn": {}, "vit": {}, "nnmamba": {},
           "cross_validation": {"splits_file": "splits.json", "mode": "auto"}}
    cfgmod.apply_smoke(cfg)
    assert "splits_file" not in cfg["cross_validation"]
