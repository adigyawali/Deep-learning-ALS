# Deep-learning-ALS — multi-modal 3D MRI ALS classifier

Binary classification (ALS vs non-ALS) from three co-registered 3D MRI
modalities (T1, T2, FLAIR), with three interchangeable models behind a single
driver:

- **`cnn_vit`** (two-stage) — tri-stream MedicalNet ResNet (one encoder per modality) → spatial multi-modal ViT over the CNN feature maps.
- **`cnn_nnmamba`** (two-stage) — 3D CNN stem → Mamba (selective state-space) classifier over spatial and/or frequency tokens.
- **`nnmamba`** (one-stage) — the same Mamba classifier with the CNN removed: 3D patch embedding straight into the state-space stack. Isolates the value of the convolutional prior.

Both Mamba models take `data.streams: both | spatial | frequency`, selecting
which token streams the state-space stack scans — a one-factor ablation of the
FFT branch.

All three share one preprocessing pipeline, one subject-level `splits.json`
(**5-fold stratified cross-validation + a held-out test set**), one
training/checkpointing core, and one metric set, so the comparison is fair. A
single run trains one model per fold and aggregates the results (`cv_summary.json`
+ `test_evaluation.json`).

```bash
pip install -e .
python experiment.py --model cnn_vit                 # full CNN→ViT pipeline
python experiment.py --model cnn_nnmamba             # full CNN→nnMamba pipeline
python experiment.py --model nnmamba                 # one-stage Mamba (no CNN)
python experiment.py --model cnn_nnmamba --streams frequency   # stream ablation
python experiment.py --model cnn_vit --smoke         # fast wiring check (no GPU needed)
```

**Full step-by-step lab-machine (WSL / RTX 5090) instructions are in
[`Instructions.md`](Instructions.md).** Read that first.
