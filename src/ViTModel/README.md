# Spatial Multi-Modal ViT for ALS Classification

A real Vision Transformer that classifies ALS vs Control from **3D feature
maps** of T1, T2 and FLAIR MRI scans. Each anatomical position becomes a
transformer token, and attention is computed jointly across positions and
modalities.

This module consumes spatial features produced by the CNN backbone in
`src/cnnModelMultiModality/` -- it does NOT read NIfTI files directly.

---

## How it fits into the project

```
Data/processed/                            <-- per-subject 3D NIfTI volumes
       │
       ▼
src/cnnModelMultiModality/                 <-- frozen MedicalNet ResNet50
   featureExtractor.SingleModalityEncoder
   .forward_features(x)                    <-- new: returns layer4 spatial map
       │
       ▼
src/cnnModelMultiModality/
   generate_spatial_features.py            <-- writes *_spatial.pt files
       │
       ▼
src/cnn_features/cnn_multimodal/
   spatial_features/                       <-- one .pt per subject-visit
       │   payload = {t1_feat, t2_feat, flair_feat, label, ...}
       │   each *_feat tensor has shape (C, D', H', W')
       │
       ▼
src/ViTModel/                              <-- THIS module
   dataset.py     loads spatial features
   model.py       SpatialMultiModalViT
   train_vit.py   trains the ViT
   evaluate_vit.py  reports test metrics
       │
       ▼
src/cnn_features/vit_multimodal/
   checkpoints/vit_spatial_best.pth
   metrics/vit_train_history.json
   metrics/vit_evaluation_metrics.json
   metrics/vit_test_predictions.json
```

The CNN side is reused as a **frozen feature extractor**; the ViT learns the
multi-modal fusion and classification on top. This is much cheaper to train
than running the CNN end-to-end and lets you iterate on the ViT quickly.

---

## How the spatial ViT works (technical but plain)

### 1. Where the tokens come from

Each MRI volume is `(1, 128, 128, 128)`. It goes through the frozen ResNet50
backbone up to **layer4** (the last convolutional stage, before global
pooling). The output is a 3D feature map of shape:

```
(2048, 4, 4, 4)        # 2048 channels, 4x4x4 spatial grid
```

Reading those numbers:
- 2048 channels = "what" the network found at that location (a learned
  feature description).
- 4x4x4 = 64 spatial cells covering the brain. Each cell roughly corresponds
  to a 32-voxel-wide cube of the original MRI.

We do this independently for T1, T2, FLAIR, so per subject we have:
```
T1:    (2048, 4, 4, 4)
T2:    (2048, 4, 4, 4)
FLAIR: (2048, 4, 4, 4)
```

### 2. Turning the feature maps into tokens

Each spatial position `(d, h, w)` in each modality becomes one token. The
2048-D channel vector at that position is linearly projected down to
`embed_dim = 384` (a Linear layer applied per position).

```
patch_embed: Linear(2048 -> 384)
```

Each token is then enriched with two learned signals:

- **Positional embedding** — one learned 384-D vector per spatial position
  `(d, h, w)`. Same vector used across all three modalities. Tells the model
  "this token came from the front-left of the brain, not the back-right".
- **Modality embedding** — one learned 384-D vector per modality
  (T1 vs T2 vs FLAIR). Same vector used across all 64 spatial positions.
  Tells the model "this token came from FLAIR, not T2".

```
token = patch_embed(channel_vector)
      + positional_embed[d, h, w]
      + modality_embed[modality]
```

We then concatenate all the modality token sequences and prepend a learnable
**CLS token** (the standard ViT trick — its final value is used as the whole
sequence's summary):

```
sequence layout per subject (193 tokens):
  [CLS,  T1 x 64,  T2 x 64,  FLAIR x 64]
```

### 3. The transformer

The 193 tokens go through a stack of standard pre-LayerNorm transformer
blocks. Each block does two things:

1. **Multi-head self-attention** — every token can look at every other token
   and decide what to read from it. Because we put all three modalities into
   one sequence, attention can mix them freely (e.g., "this T1 region looks
   suspicious; what does the same FLAIR region say?").
2. **MLP** — each token is independently transformed by a 2-layer feed-forward
   network with GELU activation.

Defaults:
- `embed_dim = 384`
- `depth = 6` blocks
- `num_heads = 6` (head size 64)
- `mlp_ratio = 4` (FFN inner size = 1536)
- `dropout = 0.15` throughout

Total parameters: ~11.5M (small by ViT standards, appropriate for the dataset
size).

### 4. Classification head

After the encoder, the CLS token's final value is read out and passed
through a small MLP to a single logit:

```
logit = head(CLS_final)              # (B, 1)
prob_ALS = sigmoid(logit)
```

Loss: `BCEWithLogitsLoss` with `pos_weight = N_negative / N_positive` computed
from the training split (handles class imbalance).

### 5. Why "spatial" matters

The previous design pooled each modality's 3D volume into a single 512-D
vector before the transformer. That throws away anatomical location -- you
can't tell motor cortex from cerebellum. The spatial ViT keeps 64 distinct
positions per modality, so attention can localize to brain regions and the
final prediction is grounded in *where* signals appear, not just *that* they
appear.

---

## Training details

| Setting | Value |
|---|---|
| Optimizer | AdamW (`lr=3e-4`, `weight_decay=0.05`) |
| LR schedule | 5-epoch linear warmup, then cosine decay |
| Loss | `BCEWithLogitsLoss(pos_weight=N_neg/N_pos)` |
| Batch size | 8 |
| Epochs | 60 (early stop after 15 epochs of no val-AUC improvement) |
| Gradient clipping | max norm 1.0 |
| Best-checkpoint metric | validation **ROC-AUC** |
| Splits | 80% / 10% / 10% by subject, **stratified by label** |
| Seed | 42 |

The dataset is split at the **subject level** (all visits of one subject
land in the same split) and **stratified by label** (so train, val, test
each keep roughly the same controls/patients ratio).

---

## How to run it (lab machine, end to end)

You need a CUDA GPU with ResNet50 pretrained weights cached and the existing
CNN module already importable.

### 1. Generate the spatial features (one-time per dataset)

```bash
cd src/cnnModelMultiModality
python generate_spatial_features.py
```

This walks every subject in `Data/processed/`, runs the frozen ResNet50
backbone, and writes per-subject `.pt` files into:

```
src/cnn_features/cnn_multimodal/spatial_features/<id>_spatial.pt
```

Each file is around 1.5 MB. If a CNN checkpoint exists at
`src/cnn_features/cnn_multimodal/checkpoints/encoder_weights.pth` its backbone
weights are loaded; otherwise the MedicalNet pretrained weights from the hub
are used.

### 2. Train the ViT

```bash
cd src/ViTModel
python train_vit.py
```

Outputs:
- `src/cnn_features/vit_multimodal/checkpoints/vit_spatial_best.pth`
- `src/cnn_features/vit_multimodal/metrics/vit_train_history.json`

Override defaults from the command line:

```bash
python train_vit.py --batch-size 16 --epochs 100
python train_vit.py --features-dir /custom/path/to/spatial_features \
                    --artifacts-dir /custom/path/to/vit_artifacts
```

### 3. Evaluate on the test split

```bash
python evaluate_vit.py
```

Outputs:
- `metrics/vit_evaluation_metrics.json` — accuracy, precision, recall, F1,
  ROC-AUC, confusion matrix.
- `metrics/vit_test_predictions.json` — per-subject probability and prediction.

### 4. Colab

`colab_vit_runner.ipynb` mounts Drive and calls `train()` / `evaluate()` with
the right paths. Update the `FEATURES_DIR` cell to point at the
`spatial_features/` folder.

---

## File map

| File | Role |
|---|---|
| `model.py` | `SpatialMultiModalViT` — the transformer over 3D feature maps |
| `dataset.py` | `ALSSpatialFeatureDataset`, subject-aware stratified split, `pos_weight` helper |
| `paths.py` | Default and Colab-overridable paths for features and artifacts |
| `train_vit.py` | Training loop with cosine LR, early stopping, AUC-best checkpoint |
| `evaluate_vit.py` | Test-split metrics + per-subject predictions |
| `colab_vit_runner.ipynb` | Colab driver |

---

## Quick sanity check (no data needed)

```python
import torch
from model import SpatialMultiModalViT

m = SpatialMultiModalViT(in_channels=2048, spatial_shape=(4, 4, 4))
x = torch.randn(2, 3, 2048, 4, 4, 4)        # (B, modalities, C, D, H, W)
y = m(x)                                    # (2, 1) logits
print(y.shape, "params:", sum(p.numel() for p in m.parameters()))
```
