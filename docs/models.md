# The models

Three models, one driver. They share the preprocessing, the frozen split, the
training loop and the metric set — only the architecture changes.

```bash
python experiment.py --model cnn_nnmamba      # train 5 folds, then evaluate
python experiment.py --model cnn_vit
python experiment.py --model nnmamba
```

Each training stage loops over all five CV folds internally, training one
independent model per fold into `runs/<model>/fold{k}/`. **Budget for ~5× a
single model's training time.**

## `cnn_nnmamba` — CNN → Mamba (the main model)

```
 (B, 3, 128³)             T1, T2, FLAIR — co-registered, 3 channels
      │  CNN stem         MedicalNet ResNet, shared across the three modalities
      ▼
 F  (B, 256, 8, 8, 8)     ONE fused feature map — no split yet
      ├────────────────────────────┐
 spatial view              FFT log-magnitude of F → 1×1×1 projection
      │                            │
 pool → 64 tokens          pool → 64 tokens          N = token_grid³
      └──── concat along the SEQUENCE axis (+ learned stream embedding) ────┘
      ▼
 (B, 128, 256) → Mamba × 2 → LayerNorm → mean over tokens → (B, 1) logit
```

Three things to know:

1. **The split happens *after* the CNN.** The frequency view is an FFT of the
   feature map, so it describes how *learned features* are spread across spatial
   frequencies. Earlier versions took the FFT of the raw voxels, or gave the
   spectrum its own conv encoder — both ask a 3D convolution to find local
   structure in a Fourier-magnitude volume, where there is none.
2. **One Mamba stack sees both streams**, concatenated along the sequence axis
   with a learned embedding marking which is which, so the state-space model can
   relate them instead of each being summarised separately and meeting only at
   the classifier. Mamba is causal, so the spatial tokens come first and the
   frequency tokens see them, not the reverse.
3. **The stem is pretrained and frozen** (`spatial_encoder: pretrained`,
   `freeze_backbone: true`). Training a 3D conv stem from scratch on a few
   hundred subjects is the dominant cause of the train→test gap; only the fusion
   conv, the frequency projection, the Mamba stack and the head are trained
   (~1.3 M of ~15.7 M parameters).

## `cnn_vit` — CNN → Vision Transformer

Two stages plus a caching step in between:
`train_cnn → extract_features → train_vit → evaluate`.

Stage 1 fine-tunes three MedicalNet ResNets (one per modality) fused by a small
transformer. Stage 2 dumps each encoder's layer4 map and trains a ViT in which
**every spatial position of every modality is a token**, so attention spans
anatomy *and* modality in one sequence (1 CLS + 3·4³ = 193 tokens at the
defaults). Because each fold fine-tunes its own CNN, features are extracted per
fold into `fold{k}/features/`.

`gradcam` is available for this model only:

```bash
python experiment.py --model cnn_vit gradcam                  # random test subject
python experiment.py --model cnn_vit gradcam --subject P096_V1
```

## `nnmamba` — one stage, no CNN

The volumes are cut into non-overlapping 3D patches, one linear projection turns
each patch into a token (this is also where the three modalities fuse), and the
Mamba stack does all of the representation learning. It answers one question:
how much of `cnn_nnmamba`'s score comes from the convolutional prior?

Two deliberate differences from the two-stage model: a **learned positional
embedding** per stream (flattening a 3D grid destroys adjacency, and there are no
convolutions here to encode locality), and **bidirectional scanning** on by
default (a token *is* its patch with no receptive field, so a causal forward-only
scan would leave the first patches with no context).

It has no MedicalNet stem, so `ALS_REQUIRE_PRETRAINED` and
`ALS_MEDICALNET_WEIGHTS` do nothing here. Every one of its ~4.6 M parameters is
trained from scratch (2.4 M in the patch projection alone), which makes it the
more overfit-prone of the two — `dropout` and `weight_decay` matter more.

## Adding or changing a model

Each model is a folder under `src/als/models/`, and
[`registry.py`](../src/als/models/registry.py) is the only file outside that
folder which knows it exists:

```
src/als/models/
  registry.py       one ModelSpec per model: its stages, config block, checkpoints
  components/       blocks shared by more than one model (CNN backbone, Mamba
                    block, FFT view, stream selection)
  cnn_vit/          encoder.py (stage 1) + vit.py (stage 2)
  cnn_nnmamba/      model.py + build()
  nnmamba/          model.py + build()
```

**To change one model**, edit its folder — nothing else references its internals.
**To add one:**

1. Create `src/als/models/<name>/` with the architecture in `model.py` and a
   `build(m, *, streams, input_shape, load_pretrained)` in `__init__.py` mapping
   the config block onto the constructor (copy `nnmamba/__init__.py`).
2. Add a `ModelSpec` to `registry.py`.
3. Add `configs/<name>.yaml`.

`--model <name>` then works, its stages run in the order the spec lists, and
outputs land in `runs/<name>/`. A model that does not fit the shared
`train_nnmamba` stage sets `volume_model=False` and supplies its own stage
modules, as `cnn_vit` does.

## The knobs that matter

Everything else is in `configs/<model>.yaml`; these are the ones you will
actually reach for.

| Knob | Where | What it does |
|---|---|---|
| `data.streams` | both Mamba models | `both` / `spatial` / `frequency` — which token streams reach the stack. Also `--streams`. |
| `nnmamba.spatial_encoder` | `cnn_nnmamba` | `pretrained` (transfer learning) or `scratch` (ablation). Also `--spatial-encoder`. |
| `nnmamba.token_grid` | `cnn_nnmamba` | Tokens **per stream** = `token_grid³`. The sequence-length / VRAM lever. |
| `nnmamba.patch_size` | `nnmamba` | Tokens per stream = `(target_shape / patch_size)³`. Same lever, one stage. |
| `cnn.backbone` | `cnn_vit` | `resnet10` / `resnet50`. First thing to shrink on OOM. |
| `data.target_shape` | all | Input resolution. Must be divisible by `patch_size` for `nnmamba`. |
| `grad_accum_steps` | all | Keeps the effective batch while cutting per-step VRAM. |
| `train.amp` | all | `bf16` (default on CUDA) / `fp16` / `none`. |
| `dropout`, `weight_decay`, `train.label_smoothing` | all | The regularisation dials for a small cohort. |

## Mamba backend and sequence length

The Mamba block uses the official `mamba-ssm` CUDA kernel if it imports, and
otherwise a built-in pure-PyTorch implementation of the same recurrence. You do
not choose; the log prints `Mamba backend: ...`.

The fallback's scan is an O(L) Python loop, so cost grows faster than linearly in
sequence length. Measured on CPU (128³, `d_model: 192`, 4 bidirectional layers,
one forward+backward at batch 2): **128 tokens → 1.5 s, 512 → 16 s, 1024 → 68 s.**

`cnn_nnmamba` is fine either way (128 tokens at the defaults). For the one-stage
`nnmamba` the default `patch_size: 16` gives 1024 tokens with `streams: both`,
which is only sensible on the CUDA kernel — **if the log says `pytorch-fallback`,
raise `patch_size` to 32 before starting a real run.** The stage prints a warning
when the sequence is long and the fallback is active.

## Training

Per fold: `BCEWithLogitsLoss` with `pos_weight` for class imbalance and optional
label smoothing, AdamW, linear warm-up then cosine decay, bf16 autocast on CUDA,
gradient clipping, and early stopping on validation ROC-AUC.

Only the best-validation weights are saved, per stage per fold, to
`runs/<model>/fold{k}/checkpoints/<prefix>_best.pt`. That file holds the weights,
the config needed to rebuild the architecture, the validation-tuned threshold and
the best score — and deliberately **no** optimizer, scheduler, scaler or RNG
state, so there is no per-epoch checkpoint and **no resume**. If a run dies,
restart it; the best file written so far is still a complete, usable model.

Logs print a live progress bar (stderr) and a one-line epoch summary with a
GPU/RAM line (stdout), so `python experiment.py ... | tee train.log` keeps a clean
file.

## Evaluation

`evaluate` runs at the end of a full run, or on its own:

```bash
python experiment.py --model cnn_nnmamba evaluate
```

It loads each fold's `*_best.pt` and writes to `runs/<model>/metrics/`:

- **`cv_summary.json`** — the cross-validation estimate.
  - `oof_metrics` — **read this one.** Every subject's prediction from the fold it
    was held out in, pooled and scored once. It also yields the single
    `oof_threshold` used below, which is far more stable than five per-fold ones.
  - `val_metrics_mean_std` — per-fold mean ± std, kept for the spread.
    `per_fold_thresholds` shows how much the per-fold thresholds varied.
- **`test_evaluation.json`** — the headline number. All five fold models scored on
  the fixed test set, as per-fold mean ± std and as a **mean-probability
  ensemble** (quote the ensemble). The ensemble uses the pooled-OOF threshold, so
  the operating point is never tuned on the data it scores.
- **`predictions.json`** — the ensemble's per-subject test outputs.

Because the data is imbalanced, both files report accuracy, balanced accuracy,
F1, precision, recall/sensitivity, specificity, ROC-AUC with bootstrap and DeLong
95% CIs, PR-AUC, Brier score, calibration error and a confusion matrix — not
accuracy alone. **Always compare the held-out test number and read its CI:** if
two models' intervals overlap heavily, treat them as tied.

## Ablations

Each is a clean one-factor A/B — same splits, same everything else. Use a
separate `--output-dir` per arm and compare `test_evaluation.json`.

```bash
# Is transfer learning earning its keep?
python experiment.py --model cnn_nnmamba --spatial-encoder pretrained --output-dir runs_pretrained
python experiment.py --model cnn_nnmamba --spatial-encoder scratch    --output-dir runs_scratch

# Which token streams help?
python experiment.py --model cnn_nnmamba --streams both      --output-dir runs_both
python experiment.py --model cnn_nnmamba --streams spatial   --output-dir runs_spatial
python experiment.py --model cnn_nnmamba --streams frequency --output-dir runs_freq

# Is the CNN stage earning its keep at all?
python experiment.py --model nnmamba --output-dir runs_onestage
```

For the stream ablation, `frequency` alone is expected to be the weakest — it is
the check that the FFT tokens carry *independent* signal rather than a shadow of
the spatial ones. If `frequency` alone is near chance while `both` beats
`spatial`, the frequency stream is working as a complement, which is the result
you want. If `frequency` alone is as good as `both`, the spatial tokens are
contributing little, and that is worth reporting.

`configs/cnn_nnmamba_noprior.yaml` is a fourth arm — the same ResNet10
architecture with the pretrained weights removed, so the difference is the
weights alone rather than the topology. It **must** be run with
`ALS_SKIP_PRETRAINED=1`, or it silently trains a fine-tuned pretrained backbone
instead.
