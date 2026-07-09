# Deep-learning-ALS — multi-modal 3D MRI ALS classifier

Binary classification (ALS vs non-ALS) from three co-registered 3D MRI
modalities (T1, T2, FLAIR), with two interchangeable two-stage models behind a
single driver:

- **`cnn_vit`** — tri-stream MedicalNet ResNet (one encoder per modality) → spatial multi-modal ViT over the CNN feature maps.
- **`cnn_nnmamba`** — end-to-end 3D CNN stem → Mamba (selective state-space) classifier, with an optional frequency-domain stream.

Both models share one preprocessing pipeline, one subject-level `splits.json`
(**5-fold stratified cross-validation + a held-out test set**), one
training/checkpointing core, and one metric set, so the comparison is fair. A
single run trains one model per fold and aggregates the results (`cv_summary.json`
+ `test_evaluation.json`).

```bash
pip install -e .
python experiment.py --model cnn_vit                 # full CNN→ViT pipeline
python experiment.py --model cnn_nnmamba             # full CNN→nnMamba pipeline
python experiment.py --model cnn_vit --smoke         # fast wiring check (no GPU needed)
```

**Full step-by-step lab-machine (WSL / RTX 5090) instructions are in
[`Instructions.md`](Instructions.md).** Read that first.
