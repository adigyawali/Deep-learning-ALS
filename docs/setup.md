# Setup

Written for the lab Windows machine running **WSL2 (Ubuntu)** with an **RTX
5090**. On any other machine, only the PyTorch install line changes.

## 1. Environment

```bash
wsl                                  # from a Windows terminal
git clone <repo-url> Deep-learning-ALS && cd Deep-learning-ALS
python3 -m venv .venv
source .venv/bin/activate            # required in every new terminal
pip install --upgrade pip
```

## 2. PyTorch — the one gotcha that matters

The RTX 5090 is Blackwell (**sm_120**). The older `cu124` / `cu121` wheels import
fine and even report `torch.cuda.is_available() == True`, but every real GPU
kernel then fails. You need **CUDA 12.8 wheels (`cu128`), PyTorch ≥ 2.7**.

```bash
# If a cu124 build is already installed, plain `pip install torch` says
# "Requirement already satisfied" and does nothing — replace it explicitly:
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

On macOS or a CPU box, plain `pip install torch torchvision torchaudio` is fine.

## 3. Install the project

```bash
pip install -e .                     # the `als` package + dependencies
pip install -e ".[dev]"              # + pytest (needed for the test suite)
pip install -e ".[monitor]"          # optional: nicer host-RAM numbers (psutil)
```

**Do not install `.[mamba]`.** That extra compiles the official CUDA Mamba
kernel and commonly fails to build; the failure looks like
`ModuleNotFoundError: No module named 'torch'`, which is a packaging quirk of
`mamba-ssm`, not a problem with your setup. You do not need it — the model falls
back to a built-in pure-PyTorch implementation automatically and prints which
backend is active. See [`models.md`](models.md) for the speed implications.

## 4. Verify

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); \
           x=torch.randn(8,device='cuda'); print((x*2).sum().item())"
```

Expect a `2.x+cu128` version, `NVIDIA GeForce RTX 5090`, and a number — with **no**
`sm_120 is not compatible` warning. If you see that warning or "no kernel image",
you are still on a `cu124` wheel; redo step 2.

```bash
python -m pytest -q                  # ~10 s, expect all tests to pass
python experiment.py --model cnn_nnmamba --smoke     # end-to-end wiring, ~1 min
```

Always use `python -m pytest`, never a bare `pytest` — the `python -m` form uses
this venv's Python and can't silently fall back to a conda or system install.

## 5. MedicalNet pretrained weights

`cnn_vit` and `cnn_nnmamba` start their CNN from **MedicalNet** 3D ResNet weights
(trained on 23 medical datasets). They download automatically on first use.
Before a real run, make a failed download loud instead of silent:

```bash
export ALS_REQUIRE_PRETRAINED=1      # abort rather than train random weights
```

On fold 0's first epoch the log must print a **non-zero**
`MedicalNet weights: N matched`. Only `resnet10` and `resnet50` have downloadable
weights — there is no resnet18/34.

**If the download is blocked or rate-limited** (the files live on Google Drive),
fetch them once by hand and point the code at the file:

| Model | Backbone | File | Google Drive ID |
|---|---|---|---|
| `cnn_nnmamba` | resnet10 | `resnet_10_23dataset.pth` | `1HLpyQ12SmzmCIFjMcNs4j3Ijyy79JYLk` |
| `cnn_vit` | resnet50 | `resnet_50_23dataset.pth` | `1qXyw9S5f-6N1gKECDfMroRnPZfARbqOP` |

```bash
pip install gdown                                                    # once, anywhere with internet
gdown 1HLpyQ12SmzmCIFjMcNs4j3Ijyy79JYLk -O resnet_10_23dataset.pth
export ALS_MEDICALNET_WEIGHTS=/full/path/to/resnet_10_23dataset.pth  # must match the model!
```

A real weights file is tens to hundreds of MB. A few-KB file means you saved
Google Drive's HTML warning page — use `gdown`, which handles that.
`ALS_MEDICALNET_WEIGHTS` is a single path, so switch it when you switch models.

Do **not** "fix" a failed load with `ALS_REQUIRE_PRETRAINED=0`. That flag only
decides what happens when the load fails: `1` aborts loudly, `0` silently trains
random weights — quietly removing the transfer learning that is the main defence
against overfitting.

## 6. On your first real run, check

The code was developed and unit-tested on a laptop (CPU/MPS). These can only be
confirmed on the lab box:

- `cu128` launches real kernels on the 5090 (step 4).
- `MedicalNet weights: N matched` is non-zero for both `cnn_vit` (resnet-50) and
  `cnn_nnmamba` (resnet-10).
- ANTs preprocessing produces sane QC images (`Data/processed/_QC_Snapshots/`).
- The Mamba backend line: `mamba-ssm` (fast) or `pytorch-fallback` (see
  [`models.md`](models.md) — raise `patch_size` for the one-stage model if so).
- Training fits in VRAM at the configured batch size — watch the live
  `gpu_alloc` line; if not, see the OOM entry below.
- `runs/splits.json` lists dataset-qualified IDs (`CALSNIC_C005`, `CAPTURE_C178`)
  and no subject appears in two partitions.
- All three models report the same `num_test_samples` in
  `test_evaluation.json` — that is what makes the comparison legitimate.

## Common errors

**`CUDA out of memory`, or the whole machine freezes.** The trainer prints a
remedy list. In order: lower `batch_size` and raise `grad_accum_steps` (keeps the
effective batch, cuts per-step VRAM); raise `grad_accum_steps` alone; use
`backbone: resnet10` or a smaller `data.target_shape`; keep `train.amp: bf16`. If
the *machine* froze rather than CUDA erroring, it is host RAM — lower
`--num-workers` (each worker loads three full volumes) and `prefetch_factor`.

**`sm_120 is not compatible` / `no kernel image is available`.** Wrong PyTorch
wheel — redo step 2.

**`Cannot find callable medicalnet_resnetXX in hubconf`.** You set `backbone:` to
a size with no MedicalNet weights. Use `resnet10` or `resnet50`.

**`NotImplementedError: aten::_adaptive_avg_pool3d ... for the MPS device`** (Apple
silicon only). `cnn_nnmamba` cannot run on MPS; use `--device cpu`, or set
`PYTORCH_ENABLE_MPS_FALLBACK=1`. Not an issue on the lab box.

**`ModuleNotFoundError: No module named 'als'`.** `pip install -e .` was never run
in the active venv, or the venv isn't activated.

**`pytest` fails to collect and the traceback shows a conda path.** You ran the
wrong `pytest`. Activate the venv and use `python -m pytest -q`.

**A fresh clone is missing source files** (e.g. all of `src/als/data/`). Almost
always a `.gitignore` rule that is too broad: an unanchored pattern like `Data/`
matches that folder name at *any* depth, and on a case-insensitive filesystem it
also matches `data/`. Anchor such rules to the repo root (`/Data/`), then
`git add` the recovered files.
