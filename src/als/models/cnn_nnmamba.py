"""
End-to-end CNN→nnMamba classifier for ALS.

Shape of the model:

    (B, 3, D, H, W)                     three co-registered modalities, T1/T2/FLAIR
        │
    CNN stem  (pretrained MedicalNet ResNet, or the from-scratch conv trunk)
        │
    F  (B, d_model, d, h, w)            ONE feature map — no split yet
        ├───────────────┐
      spatial          FFT log-magnitude → 1×1×1 projection
        │               │
      pool→N tokens   pool→N tokens
        └──── concat along the SEQUENCE axis (+ learned stream embedding) ────┘
        │
    (B, 2N, d_model)  →  Mamba × L  →  LayerNorm  →  mean over tokens
        │
    (B, 1) raw logit

Design notes:

  * **The split happens after the CNN, not before it.** An earlier version fed a
    6-channel volume (3 spatial + 3 FFT-of-the-raw-voxels) into the network, and
    the version after that gave the raw spectrum its own separate conv encoder.
    Both asked a 3D convolution to make sense of a Fourier-magnitude *volume*,
    which has no spatial locality for a conv kernel to exploit. Here the
    convolutional work happens once, in the domain it is good at, and the
    frequency view is taken of the resulting **feature map** — so it describes
    how learned features are distributed across spatial frequencies, which is a
    meaningful thing to hand a sequence model.
  * **One Mamba over both streams.** The spatial and frequency tokens are
    concatenated along the sequence axis with a learned per-stream embedding, so
    a single Mamba stack can relate the two rather than each being summarised
    independently and only met at the classifier. Note Mamba is causal: the
    frequency tokens come second and therefore see the spatial ones, not the
    other way round. The stream embedding is what keeps them distinguishable.
  * **``use_frequency=False`` drops the frequency tokens only.** The CNN stem,
    the Mamba stack, and the head are otherwise identical, so spatial-only vs
    spatial+frequency is a clean one-factor ablation.
  * **Optional transfer learning for the stem** (``spatial_encoder="pretrained"``).
    Training a 3D conv stem from scratch on ~380 subjects is the dominant cause
    of the train→test overfitting gap; the pretrained stem is a MedicalNet 3D
    ResNet shared across the three modalities. ``"scratch"`` keeps the original
    from-scratch trunk for the ablation.
  * **GroupNorm, not BatchNorm** — training runs at batch size 2–4, where
    BatchNorm statistics are unreliable.
  * **Bounded sequence length.** Each branch is adaptive-pooled to ``token_grid³``
    so the Mamba sequence is ``2·token_grid³`` regardless of input resolution.
  * **Single logit + BCEWithLogitsLoss + pos_weight** (handled by the trainer),
    matching the ViT so the two models are directly comparable.

Forward input  : ``(B, 3, D, H, W)`` — spatial modalities only. The frequency
                 view is derived inside the model, so the dataset no longer
                 builds FFT channels.
Forward output : ``(B, 1)`` raw logit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .components.cnn_backbone import backbone_forward_features, build_medicalnet_backbone
from .components.frequency import freq_magnitude
from .components.mamba_block import MambaLayer


def _gn(channels: int) -> nn.GroupNorm:
    # 8 groups, but never more groups than channels.
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class ScratchStem(nn.Module):
    """From-scratch 3D conv trunk over the 3 modalities → ``(B, d_model, d, h, w)``."""

    def __init__(self, in_ch: int = 3, base: int = 32, blocks: int = 3):
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
        self.d_model = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PretrainedStem(nn.Module):
    """MedicalNet-pretrained 3D ResNet trunk → ``(B, d_model, d, h, w)``.

    The same (weight-shared) pretrained ResNet is run on each of the three
    co-registered modalities; the three layer4 maps are concatenated and fused by
    a 1×1×1 conv down to ``d_model``. Interchangeable with ``ScratchStem``.
    """

    def __init__(
        self,
        backbone: str = "resnet10",
        d_model: int = 256,
        freeze_backbone: bool = True,
        load_pretrained: bool = True,
    ):
        super().__init__()
        self.backbone, out_ch = build_medicalnet_backbone(
            backbone, freeze=freeze_backbone, load_pretrained=load_pretrained)
        self.fuse = nn.Sequential(
            nn.Conv3d(out_ch * 3, d_model, kernel_size=1, bias=False),
            _gn(d_model), nn.GELU(),
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, D, H, W) — one channel per co-registered modality.
        feats = [backbone_forward_features(self.backbone, x[:, i : i + 1]) for i in range(3)]
        return self.fuse(torch.cat(feats, dim=1))


class CNNnnMamba(nn.Module):
    SPATIAL_CHANNELS = 3
    SPATIAL_STREAM = 0
    FREQ_STREAM = 1

    def __init__(
        self,
        use_frequency: bool = True,
        base: int = 32,
        blocks: int = 3,
        token_grid: int = 4,
        mamba_layers: int = 2,
        d_state: int = 16,
        dropout: float = 0.1,
        spatial_encoder: str = "scratch",
        backbone: str = "resnet10",
        freeze_backbone: bool = True,
        pretrained_d_model: int = 256,
        load_pretrained: bool = True,
    ):
        super().__init__()
        self.use_frequency = use_frequency
        if spatial_encoder == "pretrained":
            self.stem = PretrainedStem(
                backbone=backbone, d_model=pretrained_d_model,
                freeze_backbone=freeze_backbone, load_pretrained=load_pretrained,
            )
        elif spatial_encoder == "scratch":
            self.stem = ScratchStem(self.SPATIAL_CHANNELS, base=base, blocks=blocks)
        else:
            raise ValueError(f"spatial_encoder must be 'scratch' or 'pretrained', got {spatial_encoder!r}")

        d_model = self.stem.d_model
        self.d_model = d_model
        self.pool = nn.AdaptiveAvgPool3d(token_grid)           # bounds the sequence length

        # The frequency branch is a different domain, so it gets its own learned
        # projection before joining the spatial tokens in one sequence.
        self.freq_proj = nn.Sequential(
            nn.Conv3d(d_model, d_model, kernel_size=1, bias=False),
            _gn(d_model), nn.GELU(),
        ) if use_frequency else None

        # Marks which stream a token came from — without it the concatenated
        # sequence would be two indistinguishable halves.
        n_streams = 2 if use_frequency else 1
        self.stream_embed = nn.Parameter(torch.zeros(n_streams, d_model))

        self.mamba = nn.Sequential(*[
            MambaLayer(d_model, d_state=d_state, dropout=dropout) for _ in range(mamba_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _tokens(self, feature_map: torch.Tensor, stream: int) -> torch.Tensor:
        """``(B, C, d, h, w)`` → ``(B, token_grid³, C)`` plus the stream embedding."""
        f = self.pool(feature_map)                             # (B, C, g, g, g)
        b, c = f.shape[0], f.shape[1]
        f = f.reshape(b, c, -1).transpose(1, 2)                # (B, N, C)
        return f + self.stream_embed[stream]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.SPATIAL_CHANNELS:
            raise ValueError(
                f"CNNnnMamba expects {self.SPATIAL_CHANNELS} spatial channels (T1/T2/FLAIR), "
                f"got {x.shape[1]}. The frequency view is now computed inside the model from "
                f"the CNN feature map, so the dataset must no longer append FFT channels."
            )
        features = self.stem(x)                                # (B, d_model, d, h, w)
        tokens = self._tokens(features, self.SPATIAL_STREAM)   # (B, N, d_model)
        if self.freq_proj is not None:
            spectrum = self.freq_proj(freq_magnitude(features))
            tokens = torch.cat([tokens, self._tokens(spectrum, self.FREQ_STREAM)], dim=1)

        z = self.norm(self.mamba(tokens))                      # (B, 2N, d_model)
        return self.head(z.mean(dim=1))                        # (B, 1)
