"""
Mamba (selective state-space) block with an automatic backend fallback.

If the official ``mamba-ssm`` CUDA kernel is importable we use it (fast, the
real thing). Otherwise we fall back to a compact pure-PyTorch implementation of
the same selective-SSM (S6) recurrence. The fallback is slower per step but has
zero build dependencies, so the nnMamba pipeline runs — and is smoke-testable —
on a laptop or any box where the kernel will not build (e.g. an as-yet
unsupported GPU). Both backends expose the same ``(B, L, D) -> (B, L, D)``
interface, and ``MAMBA_BACKEND`` records which one is active.

For this project the sequence length is small (a downsampled 3D feature map,
~hundreds of tokens), so the O(L) Python scan in the fallback is acceptable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _OfficialMamba  # type: ignore
    MAMBA_BACKEND = "mamba-ssm"
except Exception:  # ImportError, or a CUDA/compile error on import
    _OfficialMamba = None
    MAMBA_BACKEND = "pytorch-fallback"


class _MambaFallback(nn.Module):
    """Pure-PyTorch selective SSM, interface-compatible with mamba_ssm.Mamba.

    Implements the S6 recurrence  h_t = Ā_t h_{t-1} + B̄_t x_t,  y_t = C_t h_t + D x_t
    with input-dependent (selective) Δ, B, C. The state recurrence is a length-L
    loop — exact, just not kernel-fused.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is diagonal and negative (stable); stored as log for positivity of -A.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)                                   # (B, L, 2*d_inner)
        xs, z = xz.chunk(2, dim=-1)                            # each (B, L, d_inner)

        # Causal depthwise conv over the sequence axis.
        xs = xs.transpose(1, 2)                                # (B, d_inner, L)
        xs = self.conv1d(xs)[..., :L]
        xs = xs.transpose(1, 2)                                # (B, L, d_inner)
        xs = F.silu(xs)

        A = -torch.exp(self.A_log.float())                     # (d_inner, d_state)
        dbl = self.x_proj(xs)                                  # (B, L, dt_rank+2*d_state)
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                      # (B, L, d_inner)

        # Discretize and scan.
        deltaA = torch.exp(dt.unsqueeze(-1) * A)               # (B, L, d_inner, d_state)
        deltaB_u = dt.unsqueeze(-1) * Bm.unsqueeze(2) * xs.unsqueeze(-1)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=deltaA.dtype)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_u[:, t]              # (B, d_inner, d_state)
            ys.append(torch.einsum("bdn,bn->bd", h, Cm[:, t]))
        y = torch.stack(ys, dim=1)                             # (B, L, d_inner)
        y = y + xs * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


def make_mamba(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> nn.Module:
    """Return an official-kernel Mamba if available, else the PyTorch fallback."""
    if _OfficialMamba is not None:
        return _OfficialMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return _MambaFallback(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class MambaLayer(nn.Module):
    """Pre-norm residual Mamba block: ``x + Mamba(LN(x))``."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = make_mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


class BiMambaLayer(nn.Module):
    """Pre-norm residual Mamba scanned forwards **and** backwards, then averaged.

    Mamba's recurrence is causal, so in a plain ``MambaLayer`` token *t* is built
    only from tokens ``<= t``. That is tolerable when a CNN has already given each
    token a wide receptive field (``CNNnnMamba``), but not when the tokens are raw
    patch projections with no context at all (``NNMamba``): the first patches in
    the flattened 3D scan order would see nothing. Two independent scans over the
    sequence and its reverse give every token both sides of the volume for the cost
    of one extra scan per layer.

    Same ``(B, L, D) -> (B, L, D)`` interface as ``MambaLayer``, so the two are
    interchangeable.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.forward_mamba = make_mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = make_mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        fwd = self.forward_mamba(h)
        bwd = self.backward_mamba(h.flip(1)).flip(1)
        # Mean (not sum) so the residual branch keeps the scale of a single scan.
        return x + self.dropout(0.5 * (fwd + bwd))


def make_mamba_layer(
    d_model: int, *, d_state: int = 16, d_conv: int = 4, expand: int = 2,
    dropout: float = 0.1, bidirectional: bool = False,
) -> nn.Module:
    """``BiMambaLayer`` when ``bidirectional`` else ``MambaLayer``."""
    cls = BiMambaLayer if bidirectional else MambaLayer
    return cls(d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
