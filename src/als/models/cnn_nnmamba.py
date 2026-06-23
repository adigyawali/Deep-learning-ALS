"""
End-to-end CNN→nnMamba classifier for ALS.

Design notes / why this differs from the earlier 6-channel version that sat
at ~60% (Goal 4):

  * **Two separate streams, late fusion.** Previously the 3 FFT channels were
    concatenated onto the 3 spatial channels and pushed through ONE conv stem.
    A 3D conv assumes spatial locality; a Fourier-magnitude volume has none, so
    one shared first-layer filter bank had to reconcile two unrelated domains.
    Here the spatial volume and the frequency volume each get their own small
    3D encoder, and we fuse the two pooled embeddings just before the classifier.
    With ``use_frequency=False`` the frequency stream is simply absent, so the
    spatial-only and spatial+frequency models share an identical spatial path —
    a clean ablation.
  * **GroupNorm, not BatchNorm.** Training runs at tiny batch sizes (2–4 for 3D
    volumes); BatchNorm statistics are unreliable there. GroupNorm is
    batch-size independent.
  * **Bounded sequence length.** Each encoder ends in an adaptive pool to a
    fixed ``token_grid³`` so the Mamba sequence length (and memory) is constant
    regardless of input resolution.
  * **Single logit + BCEWithLogitsLoss + pos_weight** (handled by the trainer),
    matching the ViT exactly so the two models are directly comparable and class
    imbalance is handled the same way.

Forward input  : ``(B, C, D, H, W)`` with ``C = 3`` (spatial) or ``6`` (spatial+freq).
Forward output : ``(B, 1)`` raw logit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .components.mamba_block import MambaLayer


def _gn(channels: int) -> nn.GroupNorm:
    # 8 groups, but never more groups than channels.
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class Encoder3D(nn.Module):
    """Small 3D conv encoder → fixed-size token sequence → stacked Mamba layers.

    Output is a pooled ``(B, d_model)`` embedding.
    """

    def __init__(
        self,
        in_ch: int,
        base: int = 32,
        blocks: int = 3,
        token_grid: int = 4,
        mamba_layers: int = 2,
        d_state: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(in_ch, base, kernel_size=3, padding=1, bias=False),
            _gn(base), nn.GELU(),
        ]
        ch = base
        for _ in range(blocks):
            out = min(ch * 2, 256)
            layers += [
                nn.Conv3d(ch, out, kernel_size=3, stride=2, padding=1, bias=False),
                _gn(out), nn.GELU(),
            ]
            ch = out
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(token_grid)           # bound sequence length
        self.d_model = ch
        self.mamba = nn.Sequential(*[
            MambaLayer(ch, d_state=d_state, dropout=dropout) for _ in range(mamba_layers)
        ])
        self.norm = nn.LayerNorm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)                                       # (B, C, g, g, g)
        b, c = x.shape[0], x.shape[1]
        x = x.reshape(b, c, -1).transpose(1, 2)                # (B, N, C) — N = g³ tokens
        x = self.mamba(x)
        x = self.norm(x)
        return x.mean(dim=1)                                   # (B, C) mean over tokens


class CNNnnMamba(nn.Module):
    SPATIAL_CHANNELS = 3
    FREQ_CHANNELS = 3

    def __init__(
        self,
        use_frequency: bool = True,
        base: int = 32,
        blocks: int = 3,
        token_grid: int = 4,
        mamba_layers: int = 2,
        d_state: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_frequency = use_frequency
        self.spatial = Encoder3D(
            self.SPATIAL_CHANNELS, base=base, blocks=blocks, token_grid=token_grid,
            mamba_layers=mamba_layers, d_state=d_state, dropout=dropout,
        )
        fused_dim = self.spatial.d_model
        if use_frequency:
            # Smaller frequency encoder: the spectrum carries less discriminative
            # signal than the spatial image, so we give it fewer parameters to
            # limit overfitting on a small dataset.
            self.freq = Encoder3D(
                self.FREQ_CHANNELS, base=max(8, base // 2), blocks=blocks, token_grid=token_grid,
                mamba_layers=max(1, mamba_layers - 1), d_state=d_state, dropout=dropout,
            )
            fused_dim += self.freq.d_model
        else:
            self.freq = None

        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freq is None:
            emb = self.spatial(x[:, : self.SPATIAL_CHANNELS])
        else:
            spatial = x[:, : self.SPATIAL_CHANNELS]
            freq = x[:, self.SPATIAL_CHANNELS : self.SPATIAL_CHANNELS + self.FREQ_CHANNELS]
            emb = torch.cat([self.spatial(spatial), self.freq(freq)], dim=1)
        return self.head(emb)                                  # (B, 1)
