# Deep-learning-ALS — multi-modal MRI ALS classifier

Two-stage pipeline:

1. **Tri-stream 3D CNN** (MedicalNet ResNet50) — one independent encoder per modality (T1, T2, FLAIR), producing 3D feature maps in MNI152 space.
2. **Spatial multi-modal ViT** — fuses the per-modality feature maps with cross-modal attention and CLS-token classification.

The two stages share a single `splits.json` so the test set never touches training.

---

## Lab-machine quickstart (WSL2 / RTX 5090)

If your lab box already has the previous CNN environment set up (the one from `src/cnnModelMultiModality/README.md`), the only new thing is the orchestrator and a slightly fuller dependency list. The five commands below get you from a fresh clone to evaluated metrics.

```bash
# 0. Get the latest code
git pull

# 1. Activate the env you already have, or create a new one
#    (Python >= 3.10. Replace with conda/uv/poetry if you prefer.)
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 2. PyTorch FIRST, with the CUDA build that matches your driver
#    (CUDA 12.4 is the safe default for RTX 5090 today.)
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# 3. Everything else
pip install -r requirements.txt

# 4. Verify the wiring (no GPU work yet, ~3 seconds)
pytest tests/ -q

# 5. Run the entire pipeline
python src/experiment.py all --device cuda
```

That's it. `experiment.py all` runs: `preprocess → train_cnn → extract_features → train_vit → evaluate_vit`. Each stage is idempotent — re-running skips work that already has outputs on disk.

If you want finer control, run any subset of stages:

```bash
python src/experiment.py preprocess
python src/experiment.py train_cnn --cnn-epochs 60 --cnn-batch-size 4
python src/experiment.py extract_features --require-checkpoint
python src/experiment.py train_vit --vit-epochs 60 --vit-batch-size 8
python src/experiment.py evaluate_vit
python src/experiment.py gradcam --subject P096_V1
```

### What the lab machine specifically needs

- **CUDA-capable PyTorch** (the `cu124` wheel above works on the 5090).
- **`antspyx`** for preprocessing (registration, N4). One install: `pip install antspyx`. On WSL2 this just works; on macOS the wheel is also published.
- **MedicalNet pretrained weights** are pulled from `torch.hub` on first run (~5 minutes the first time, cached afterwards). If the lab machine is offline, copy the cached `~/.cache/torch/hub/Warvito_MedicalNet-models_main/` folder from a machine that has it.
- **Data layout** must be:
  ```
  Data/raw/T1W/      *.nii.gz
  Data/raw/T2W/      *.nii.gz
  Data/raw/FLAIR/    *.nii.gz
  ```
  If you have SynthStrip-stripped scans, put them in `T1W_synthstrip/`, `T2W_synthstrip/`, `FLAIR_synthstrip/` — the preprocessing script auto-detects and prefers stripped folders.

### Things you do NOT need to do again

- Filename mismatches between modalities are handled (subject-keyed pairing, FLAIR3D vs FLAIR_EPI, `_run-02` reruns).
- Subject IDs like `C005_V1` (short form) and `CALSNIC2_EDM_C005_V1` (long form) both parse correctly.
- Splits are subject-stratified and class-balanced — written once to `src/cnn_features/cnn_multimodal/splits.json`, read by every downstream stage.

---

## What changed since the audit

| File | What it does now |
|---|---|
| `src/splits.py` (new) | Canonical subject + label (+ site if available) stratified splitter. Writes `splits.json` once; every stage reads it. |
| `src/preprocessing/preprocessing.py` | Rewritten. Subject-keyed pairing (no more `zip(sorted)`); auto-detects `T1W_synthstrip` vs `T1W`; handles `FLAIR3D` and `FLAIR_EPI`; respects `_run-02` reruns (highest run wins); writes a `manifest.csv`. |
| `src/cnnModelMultiModality/dataset.py` | Fixed subject-ID regex (the old one silently emptied the dataset on short folder names). Mask-aware Z-score normalization when `*_mask.nii.gz` exists. |
| `src/cnnModelMultiModality/train.py` (new) | CLI trainer with AMP bf16 on CUDA, AdamW with layerwise LR decay (low for backbone, high for head), cosine LR with warmup, AUC-best checkpoint with optimizer/scheduler/RNG saved. Reads/writes `splits.json`. |
| `src/cnnModelMultiModality/generate_features.py` / `generate_spatial_features.py` | Atomic `.pt` writes; `--require-checkpoint`; sample metadata persisted alongside features. |
| `src/cnnModelMultiModality/evaluate.py` | Reads `splits.json` so the test set agrees with the ViT's test set. PR-AUC added. |
| `src/cnnModelMultiModality/gradcam.py` | Reads `splits.json` instead of recomputing a split. |
| `src/cnnModelMultiModality/split_utils.py` | Shim around `src/splits.py` for back-compat. |
| `src/ViTModel/dataset.py` | Uses `src/splits.py`. Site is preserved from metadata. |
| `src/ViTModel/train_vit.py` | AMP bf16 on CUDA; tracks val ROC-AUC + PR-AUC; saves Youden-J optimal threshold in the checkpoint; full RNG/optimizer/scheduler state for clean resume. |
| `src/ViTModel/evaluate_vit.py` | Bootstrap and DeLong 95% CIs for AUC; Brier + ECE calibration; sensitivity & specificity at val-tuned threshold; per-site metrics when ≥ 2 sites are present. |
| `src/experiment.py` (new) | One-shot end-to-end driver. `python src/experiment.py all`. |
| `tests/` (new) | 34 unit + smoke tests covering subject-ID regex, sorted-zip pairing regression, dataset shape, splits round-trip, ViT shape, AUC CI helpers. |
| `pyproject.toml`, `requirements.txt` | Cross-platform dependency declaration. |

---

## How to run individual stages locally on macOS (CPU/MPS)

Useful for development. Skip preprocessing on macOS unless ANTsPy is installed.

```bash
pytest tests/ -q                                # always works
python src/experiment.py train_vit --device mps # works once features exist
```

For preprocessing you need ANTsPy:

```bash
pip install antspyx
python src/preprocessing/preprocessing.py --list-only   # dry-run check
python src/preprocessing/preprocessing.py               # real run
```

---

## File map (post-audit)

```
src/
├── experiment.py                # one-shot pipeline driver
├── splits.py                    # canonical splitter
├── preprocessing/preprocessing.py
├── cnnModelMultiModality/
│   ├── dataset.py
│   ├── featureExtractor.py
│   ├── classifier.py
│   ├── train.py                 # NEW
│   ├── evaluate.py
│   ├── generate_features.py
│   ├── generate_spatial_features.py
│   ├── gradcam.py
│   └── split_utils.py           # back-compat shim around src/splits.py
└── ViTModel/
    ├── dataset.py
    ├── model.py
    ├── train_vit.py
    └── evaluate_vit.py
tests/                            # pytest, 34 tests, runs in ~3s
requirements.txt
pyproject.toml
```

---

## Expected runtimes (RTX 5090, ~120 subjects)

| Stage | First run | Re-run (idempotent) |
|---|---|---|
| `preprocess` (Affine) | ~25 min total | seconds (skips done cases) |
| `preprocess` (SyN, `--nonlinear`) | ~3–4 h | seconds |
| `train_cnn` (60 epochs, bf16, batch 4) | ~30–60 min | from checkpoint |
| `extract_features` | 2–5 min | seconds |
| `train_vit` (60 epochs, bf16, batch 8) | ~3–10 min | from checkpoint |
| `evaluate_vit` | seconds | seconds |

---

## Reading list (referenced in the audit)

- Chen et al., *Med3D: Transfer learning for 3D medical image analysis*, 2019 (MedicalNet backbone).
- Hatamizadeh et al., *Swin UNETR*, 2021/2022.
- Chen et al., *CrossViT*, ICCV 2021 (cross-modal token fusion).
- DeLong et al., 1988 (AUC variance).
- Sun & Xu, *Fast implementation of DeLong*, 2014.

---

## Whiteboard

https://webwhiteboard.com/board/uXjVG04c9cE=/?boardAccessToken=ojcnfv0uBRmX1bo0Dh0B3mB2xI1w1wPC
