# Data, preprocessing and splits

## Raw layout

The data is never committed. Copy it onto the machine so it looks like this:

```
Data/raw/T1W/      *.nii.gz
Data/raw/T2W/      *.nii.gz
Data/raw/FLAIR/    *.nii.gz
```

Subjects from **both cohorts go in the same folders**. Skull-stripped exports go
in `T1W_synthstrip/` etc. and are auto-detected; a trailing `_synthstrip` in the
filename is also handled. The first folder that exists *and* holds `.nii.gz`
files wins:

| Modality | Folders tried, in order |
|---|---|
| T1 | `T1W_synthstrip` → `T1W` |
| T2 | `T2W_synthstrip` → `T2PD_synthstrip` → `T2W` → `T2PD` |
| FLAIR | `FLAIR_synthstrip` → `FLAIR` |

Override with `--t1-subdir` / `--t2-subdir` / `--flair-subdir`. The run prints
which folder it chose — check that line.

## Filenames

```
DATASET_SITE_SUBJECT_MODALITY_TIMEPOINT.nii.gz

CALSNIC2_EDM_C005_T1w10_V1.nii.gz        V1/V2/V3 = visit index
CAPTURE_CHU_C178_T1w10_00M.nii.gz        00M…12M  = months since baseline
```

Everything the pipeline needs is in that name:

- **Label** — `C###` is a control, `P###` is a patient. There is no label file.
- **Timepoint** — visit index or months; both are treated as "the timepoint", so
  each timepoint is a distinct sample.
- **Site** — used for stratification when possible.
- **Subject IDs are dataset-qualified.** Both cohorts number people `C###`/`P###`
  and the numbering is *not* shared, so the splitter keys on `CALSNIC_C003` /
  `CAPTURE_C003`. This shows up in `splits.json` and `manifest.csv`; processed
  folder names are unaffected.
- **The T2 token differs by cohort.** CALSNIC writes `T2w10`; CAPTURE exports T2
  as the dual-echo PD/T2 series, `PDT2` (`T2PD` is accepted too). A *pure* PD
  file (`PDw10`) is deliberately not treated as T2 and appears in the
  `skipped N unmatched file(s)` line.
- **Rerun files** (`_run-02` in CALSNIC, `_run2` in CAPTURE) are the repeat after
  a failed scan, so they are used. Only the tag is dropped, so `..._V1_run-02`
  and `..._V1` are the same sample; if both exist, the highest run wins.

Pairing is **subject-keyed**, not sorted-zip: each modality folder is scanned into
`{(subject, timepoint): file}`, and a sample is processed only when all three
modalities are present.

Check how the parser sees your data before committing to a long job — it uses the
same folder search and regexes as the real pipeline, and writes nothing:

```bash
python scripts/check_raw_layout.py
```

It reports the chosen folders, the modality tokens found, unmatched filenames,
the triplet count, subject/sample class balance, and whether site stratification
will engage.

## Preprocessing

```bash
python experiment.py --model cnn_nnmamba preprocess              # affine (default)
python experiment.py --model cnn_nnmamba preprocess --nonlinear  # SyN, much slower
python -m als.data.preprocessing --list-only                     # dry run, no ANTs work
```

Per sample: reorient to RAS → N4 bias-field correction → register T1 to MNI152 →
register T2 and FLAIR rigidly to T1 and compose with the T1→MNI transform, so
each modality is resampled once. CPU-bound (ANTs) and the slowest stage, but
**resumable**: a sample whose three outputs already exist is skipped, so a crash
costs nothing already done. Preprocessing is model-independent.

```
Data/processed/<sample_id>/<sample_id>_{T1,T2,FLAIR}.nii.gz
Data/processed/_QC_Snapshots/<sample_id>_QC.png     eyeball a few before training
Data/processed/manifest.csv                         index of every processed sample
```

> Folders created by older code kept the rerun tag
> (`Data/processed/..._V2_run-02/`). Those are orphans now — the loader would read
> them as a second copy of the same scan. Rename them with
> `scripts/migrate_processed_run_tags.sh` (dry run by default) or delete them and
> re-run preprocessing. `ls Data/processed | grep run` should come back empty.

## At load time

Each modality is z-scored over its foreground voxels, resized trilinearly to
`data.target_shape` (128³), and the three are stacked as channels →
`(3, 128, 128, 128)`. The frequency view is **not** built here; the models derive
it from their own feature maps.

## Augmentation — `config.yaml`

Augmentations live in the root `config.yaml` and apply to **training data only**;
validation and test are never augmented. You never edit Python to change them —
any class in `monai.transforms` can be named there.

Each entry declares a `group`, and the group is not cosmetic:

- **geometric** (flip, affine, rotate, zoom) moves voxels. Sampled **once** and
  applied to T1/T2/FLAIR together, so the co-registered modalities stay aligned.
- **intensity** (noise, contrast, bias field, blur) changes voxel values only and
  runs afterwards.

`augmentations.mixup` is separate because it blends *pairs* of subjects, so the
trainer applies it per batch rather than per sample. It is off by default; when
on, the training ROC-AUC prints as `nan` (the targets are soft) — read the
validation numbers.

## Splits — also `config.yaml`

One **frozen, subject-level** split is written to `splits.json` and reused by
every run, stage, model and `--output-dir`. That is what makes two experiments
comparable: they are scored on the identical held-out subjects, so a difference
in the numbers is a difference in the models.

- A stratified **20% test set** (`test_ratio`) is carved out once and never used
  for training or threshold tuning.
- The remaining 80% becomes **5 stratified folds**; fold *k* validates, the other
  four train. So one command trains five models per stage.
- Every visit of a subject stays in one partition — no patient leakage.
- Stratification is by label, and by site only when every (label, site) bucket
  has at least `n_folds + 1` subjects; otherwise it quietly falls back to
  label-only. The outcome is recorded as `stratify_by_site` in the file.

```bash
python experiment.py --model cnn_nnmamba splits              # show the current split
python experiment.py --model cnn_nnmamba splits --reshuffle  # draw a new one and freeze it
python experiment.py --model cnn_nnmamba splits --reshuffle --seed 7
```

The generator is deterministic, so `--reshuffle` draws a *fresh* seed and records
it in the file; pass that seed back with `--seed` to recreate a split exactly.

Set `cross_validation.mode: explicit` to define the folds yourself by patient ID
(supervisor-approved splits). Rules: a subject may appear at most once, an
unknown ID is ignored with a warning, and a subject listed nowhere is **dropped**
with a loud warning — so list every subject you want used.

## Cohort size is the thing to keep in mind

Run `scripts/check_raw_layout.py` for current numbers. On the first CAPTURE-only
export it was 302 samples from 135 subjects (51 control / 84 patient) across five
sites, with 1–4 timepoints each. Three consequences:

1. A ~22-subject validation fold gives a noisy ROC-AUC, which makes both early
   stopping and best-checkpoint selection noisy. Quote the pooled out-of-fold
   numbers, not per-fold thresholds.
2. A ~27-subject test set with ~10 controls gives a very wide specificity CI.
   Read the intervals in `test_evaluation.json` before claiming any difference
   between two runs — at this size most honest comparisons come out "tied".
3. **Sample-level balance is not subject-level balance.** A subject with four
   timepoints contributes four training samples. If follow-up correlates with
   being a patient — usual in an ALS cohort — the imbalance the loss sees is
   worse than the subject counts suggest. The diagnostic prints both
   `labels (subject)` and `labels (sample)`; compare them. Splits are still
   subject-level, so this is unequal contribution to the loss, not leakage.
