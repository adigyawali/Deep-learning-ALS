# Lab Machine Setup (RTX 5090, WSL Ubuntu)

The RTX 5090 is Blackwell architecture (sm_120). Stable PyTorch wheels and prebuilt `mamba-ssm` wheels do not include sm_120 kernels yet, so we install PyTorch nightly with cu128 and build `causal-conv1d` and `mamba-ssm` from source.

## 1. Verify the basics

```bash
nvidia-smi              # should show RTX 5090 + driver
nvcc --version          # should report CUDA 12.8 or 12.9
```

If `nvcc` is missing, install the toolkit:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

Add to `~/.bashrc`:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## 2. Conda env + PyTorch nightly

```bash
conda create -n nnmamba python=3.11 -y
conda activate nnmamba
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If the message "sm_120 is not compatible with the current PyTorch installation" appears, the nightly index URL above is wrong — check pytorch.org for the current Blackwell index.

## 3. Build causal-conv1d and mamba-ssm

```bash
sudo apt install -y build-essential gcc-11 g++-11 ninja-build

export CC=gcc-11 CXX=g++-11
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4

pip install packaging wheel ninja

git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install -e . --no-build-isolation
cd ..

git clone https://github.com/state-spaces/mamba.git
cd mamba
# Edit setup.py — find the cc_flag.append("arch=compute_90,...") section
# and add: cc_flag.append("arch=compute_120,code=sm_120")
MAMBA_FORCE_BUILD=TRUE pip install -e . --no-build-isolation
cd ..
```

Smoke test:

```bash
python -c "
import torch
from mamba_ssm import Mamba
m = Mamba(d_model=64).cuda()
x = torch.randn(2, 16, 64).cuda()
print('ok', m(x).shape)
"
```

## 4. Install the rest and clone the upstream repo

```bash
pip install -r requirements.txt
git clone https://github.com/lhaof/nnMamba.git external/nnMamba
```

## 5. Run

```bash
python src/als_classifier/inspect_data.py
python src/als_classifier/train.py 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M).log
```
