"""
End-to-end CNN → nnMamba classifier.

    (B, 3, D, H, W)                 T1/T2/FLAIR, co-registered
        │  CNN stem                 MedicalNet ResNet (shared across the three
        ▼                           modalities) or a from-scratch conv trunk
    F  (B, d_model, d, h, w)        ONE fused feature map — no split yet
        ├──────────────┐
     spatial view    FFT log-magnitude of F → 1×1×1 projection
        │              │
     pool → N tokens  pool → N tokens        N = token_grid³
        └──── concat along the SEQUENCE axis (+ learned stream embedding) ────┘
        ▼
    (B, 2N, d_model) → Mamba × L → LayerNorm → mean over tokens → (B, 1) logit

Four things worth knowing:

  * **The split happens after the CNN, not before it.** The frequency view is an
    FFT of the *feature map*, so it describes how learned features spread across
    spatial frequencies. Earlier versions took the FFT of the raw voxels, which
    gives a 3D convolution no spatial locality to exploit.
  * **One Mamba stack sees both streams**, concatenated along the sequence axis
    with a learned per-stream embedding, so it can relate them rather than
    summarising each separately. Mamba is causal, so spatial tokens come first
    and the frequency tokens see them, not the reverse.
  * **``streams``** picks which streams reach the stack — ``"both"`` (default),
    ``"spatial"``, ``"frequency"``. Stem, Mamba and head are identical in all
    three, so it is a clean one-factor ablation. The older ``use_frequency``
    boolean still maps to ``"both"`` / ``"spatial"``.
  * **``spatial_encoder="pretrained"``** starts the stem from MedicalNet weights
    (frozen by default) — the main defence against overfitting a ~380-subject
    cohort. ``"scratch"`` is the ablation. GroupNorm throughout, since batch size
    is 2–4. Adaptive pooling bounds the sequence at ``2·token_grid³`` whatever
    the input resolution.

Forward: ``(B, 3, D, H, W)`` spatial modalities only → ``(B, 1)`` raw logit.
The dataset never builds FFT channels; that happens here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..components.cnn_backbone import backbone_forward_features, build_medicalnet_backbone
from ..components.frequency import freq_magnitude
from ..components.mamba_block import MambaLayer
from ..components.streams import active_streams, resolve_stream_mode


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
    # Canonical stream order. When both are active the spatial tokens come first,
    # so row 0 of ``stream_embed`` is always spatial and row 1 always frequency —
    # which keeps checkpoints written before ``streams`` existed loadable.
    STREAM_ORDER = ("spatial", "frequency")

    def __init__(
        self,
        streams: str | None = None,
        use_frequency: bool | None = None,
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
        self.streams = resolve_stream_mode(streams, use_frequency)
        self.active_streams = active_streams(self.streams, self.STREAM_ORDER)
        self._stream_row = {name: i for i, name in enumerate(self.active_streams)}
        self.use_spatial = "spatial" in self._stream_row
        self.use_frequency = "frequency" in self._stream_row
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
        self.tokens_per_stream = int(token_grid) ** 3

        # The frequency branch is a different domain, so it gets its own learned
        # projection before joining the spatial tokens in one sequence.
        self.freq_proj = nn.Sequential(
            nn.Conv3d(d_model, d_model, kernel_size=1, bias=False),
            _gn(d_model), nn.GELU(),
        ) if self.use_frequency else None

        # Marks which stream a token came from — without it the concatenated
        # sequence would be two indistinguishable halves. Only the active streams
        # get a row, so a single-stream ablation carries no dead parameters.
        self.stream_embed = nn.Parameter(torch.zeros(len(self.active_streams), d_model))

        self.mamba = nn.Sequential(*[
            MambaLayer(d_model, d_state=d_state, dropout=dropout) for _ in range(mamba_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    @property
    def sequence_length(self) -> int:
        """Tokens the Mamba stack scans per sample — the main cost driver."""
        return len(self.active_streams) * self.tokens_per_stream

    def _tokens(self, feature_map: torch.Tensor, stream: str) -> torch.Tensor:
        """``(B, C, d, h, w)`` → ``(B, token_grid³, C)`` plus the stream embedding."""
        f = self.pool(feature_map)                             # (B, C, g, g, g)
        b, c = f.shape[0], f.shape[1]
        f = f.reshape(b, c, -1).transpose(1, 2)                # (B, N, C)
        return f + self.stream_embed[self._stream_row[stream]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.SPATIAL_CHANNELS:
            raise ValueError(
                f"CNNnnMamba expects {self.SPATIAL_CHANNELS} spatial channels (T1/T2/FLAIR), "
                f"got {x.shape[1]}. The frequency view is now computed inside the model from "
                f"the CNN feature map, so the dataset must no longer append FFT channels."
            )
        features = self.stem(x)                                # (B, d_model, d, h, w)
        # Streams are built in STREAM_ORDER, so with both active the spatial
        # tokens precede the frequency ones in the sequence Mamba scans.
        parts: list[torch.Tensor] = []
        if self.use_spatial:
            parts.append(self._tokens(features, "spatial"))
        if self.freq_proj is not None:
            parts.append(self._tokens(self.freq_proj(freq_magnitude(features)), "frequency"))
        tokens = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

        z = self.norm(self.mamba(tokens))                      # (B, n_streams*N, d_model)
        return self.head(z.mean(dim=1))                        # (B, 1)
