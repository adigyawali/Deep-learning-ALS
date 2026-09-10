# Deep-learning-ALS

Binary classification — **ALS patient vs. control** — from three co-registered 3D
MRI modalities (T1, T2, FLAIR), with three interchangeable models behind one
driver.

| Model | Stages | What it is |
|---|---|---|
| `cnn_nnmamba` | 2 | One 3D CNN stem over all three modalities. Its feature map is split into a spatial view and an FFT view, and both go into a single Mamba (selective state-space) classifier as one token sequence. **The model under active development.** |
| `cnn_vit` | 2 | One MedicalNet 3D ResNet per modality, then a Vision Transformer over the CNN feature maps. |
| `nnmamba` | 1 | The same Mamba classifier with the CNN removed — 3D patches go straight into the state-space stack. Isolates what the convolutional prior is worth. |

All three share one preprocessing pipeline, one frozen subject-level split
(5-fold cross-validation + a held-out test set), one training loop and one metric
set, so their results are directly comparable. See
[`docs/models.md`](docs/models.md) for how each one works.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                      # see docs/setup.md for PyTorch first
python -m pytest -q                          # ~10 s, no GPU needed

python experiment.py --model cnn_nnmamba --smoke     # end-to-end wiring check
python experiment.py --model cnn_nnmamba preprocess  # raw NIfTI -> Data/processed
python experiment.py --model cnn_nnmamba             # train 5 folds + evaluate
```

On the lab machine, read [`docs/setup.md`](docs/setup.md) **first** — the PyTorch
build matters, and the pretrained weights may need a manual download.

Useful flags: `--smoke`, `--device`, `--output-dir`, `--data-dir`, `--epochs`,
`--batch-size`, `--lr`, `--num-workers`, `--streams`, `--spatial-encoder`.
Named stages can be run alone, e.g.
`python experiment.py --model cnn_vit train_vit evaluate`.

## Layout

```
experiment.py            the only entry point — picks a model, runs its stages
config.yaml              augmentations + CV splits, shared by all models
configs/<model>.yaml     per-model architecture and training hyper-parameters
src/als/
  data/                  preprocessing, datasets, augmentation, mixup
  models/
    registry.py          what models exist and how each is built  <- start here
    components/          blocks shared by more than one model
    cnn_vit/             one folder per model
    cnn_nnmamba/
    nnmamba/
  stages/                one module per pipeline stage
  training/              trainer, metrics, optimiser, checkpointing
  splits.py              subject-level split: held-out test + K folds
  paths.py, config.py, sanity.py, gpu.py, seed.py
scripts/                 read-only diagnostics and one-off maintenance
tests/                   pytest suite (CPU only, no data required)
docs/                    setup, data, models
ProjectOverview/         architecture diagram (.pptx)
Data/                    raw + preprocessed MRI (gitignored, never committed)
runs/                    checkpoints, metrics, predictions (gitignored)
```

## Where results go

```
runs/
  splits.json                          the ONE shared split (all models, all runs)
  <model>/
    config.json                        the exact config that produced this run
    metrics/cv_summary.json            cross-validation estimate (pooled out-of-fold)
            test_evaluation.json       headline held-out test numbers  <- quote this
            predictions.json           per-subject ensemble test predictions
    fold0/ ... fold4/
      checkpoints/<prefix>_best.pt     best-validation weights for that fold
      metrics/<prefix>_train_history.json
```

Training saves **only** each fold's best-validation weights, and there is no
resume — a crashed run restarts from the beginning, but the best file written so
far stays valid for evaluation. Details in [`docs/models.md`](docs/models.md).

## Docs

- [`docs/setup.md`](docs/setup.md) — environment, GPU, pretrained weights, common errors
- [`docs/data.md`](docs/data.md) — raw layout, preprocessing, splits, augmentation
- [`docs/models.md`](docs/models.md) — the three architectures, the knobs that matter, ablations
- `ProjectOverview/CNN_nnMamba_Architecture.pptx` — the CNN→nnMamba architecture diagram

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
