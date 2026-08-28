"""
One-stage nnMamba classifier for ALS — no CNN feature extractor at all.

``CNNnnMamba`` is a two-stage model: a 3D CNN turns the volumes into a feature
map, and only then does the Mamba stack see anything. This model deletes stage
one. The preprocessed volumes are cut into non-overlapping 3D patches, each patch
is projected to a token by a single linear map, and the Mamba stack does *all* of
the representation learning. It exists to answer one question directly — how much
of the two-stage model's score comes from the convolutional prior, and how much
from the state-space sequence model?

Shape of the model:

    (B, 3, D, H, W)                     three co-registered modalities, T1/T2/FLAIR
        │
    PatchEmbed3D                        ONE strided linear projection, kernel = stride
        │                               = patch_size. Fuses the 3 modalities here.
    E  (B, d_model, D/p, H/p, W/p)      the patch grid — no split yet
        ├───────────────┐
      spatial          FFT log-magnitude of E → 1×1×1 projection
        │               │
      N tokens        N tokens          N = (D/p)·(H/p)·(W/p); either can be off
        └──── concat along the SEQUENCE axis (+ per-stream position embed) ────┘
        │
    (B, 2N, d_model)  →  BiMamba × L  →  LayerNorm  →  mean over tokens
        │
    (B, 1) raw logit

Design notes — the four decisions that define this architecture:

  * **Modality fusion happens at the patch embedding (input level).** T1/T2/FLAIR
    are co-registered voxel-for-voxel, so one patch location describes the same
    tissue in all three; a projection that sees all three at once can learn the
    cross-modal contrast of that tissue directly, which is the most information
    a linear map can extract there. The alternative — one token stream per
    modality — would triple the sequence length for a fusion the model then has
    to relearn, and the sequence length is the binding constraint here (see
    ``patch_size``). This mirrors ``CNNnnMamba``'s scratch stem, which likewise
    fuses the three modalities in its first convolution.
  * **The stream split happens after the patch embedding**, exactly as in
    ``CNNnnMamba`` it happens after the CNN stem. The frequency view is an FFT of
    the *learned patch features*, not of the raw voxels — so it describes how
    learned features are distributed across spatial frequency, and so that the
    ``streams`` ablation measures the same thing in both models and their numbers
    stay comparable.
  * **The two streams fuse *before* Mamba, not at the classifier.** They are
    concatenated along the sequence axis so a single Mamba stack can relate them,
    rather than being summarised independently and only meeting at the head. A
    learned per-stream positional embedding is what keeps the two halves
    distinguishable — and it doubles as the stream marker, so unlike
    ``CNNnnMamba`` there is no separate ``stream_embed`` parameter.
  * **Bidirectional scanning is on by default** (``bidirectional=True``). Mamba
    is causal. In the two-stage model that is survivable because the CNN has
    already given every token a wide receptive field, but here a token *is* its
    patch and nothing else, so a forward-only scan would leave the first patches
    of the flattened volume with no context at all. See ``BiMambaLayer``.

Positional information is a **learned** embedding over the patch grid rather than
the implicit ordering Mamba's scan provides, because flattening a 3D grid to 1D
destroys most spatial adjacency (voxel neighbours in z land N² tokens apart).
``CNNnnMamba`` needs no equivalent — its convolutions encode locality.

The main cost knob is ``patch_size``: it sets the sequence length as
``(D/p)·(H/p)·(W/p)`` per stream. At the default 128³ input and ``patch_size=16``
that is 512 tokens per stream, 1024 with both. Raise it to 32 (64 tokens/stream)
if the pure-PyTorch Mamba fallback is too slow — its scan is an O(L) Python loop.

Note this model concentrates a lot of its parameters in the patch projection
(``3·p³·d_model``), so it is the more overfit-prone of the two on a few-hundred
subject dataset; ``dropout`` and ``weight_decay`` matter more here than they do
for the frozen-backbone two-stage model.

Forward input  : ``(B, 3, D, H, W)`` — spatial modalities only, same as the
                 two-stage model. The frequency view is derived inside.
Forward output : ``(B, 1)`` raw logit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .components.frequency import freq_magnitude
from .components.mamba_block import make_mamba_layer
from .components.streams import active_streams, resolve_stream_mode


def _gn(channels: int) -> nn.GroupNorm:
    # 8 groups, but never more groups than channels.
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class PatchEmbed3D(nn.Module):
    """Non-overlapping 3D patches → one token each.

    Implemented as a ``Conv3d`` whose kernel equals its stride, which is exactly a
    per-patch linear projection (each output position sees one patch, and no two
    positions share input voxels) — not a convolutional feature extractor. That
    distinction is the whole point of this model.
    """

    def __init__(self, in_channels: int = 3, d_model: int = 192, patch_size: int = 16):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv3d(in_channels, d_model, kernel_size=self.patch_size,
                              stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        spatial = tuple(x.shape[-3:])
        if any(s % p for s in spatial):
            # A stride-p conv would silently drop the remainder voxels off one
            # edge, which is an asymmetric crop of the brain — refuse instead.
            raise ValueError(
                f"NNMamba input {spatial} is not divisible by patch_size={p}. "
                f"Set data.target_shape to a multiple of the patch size (e.g. "
                f"[128, 128, 128] with patch_size 16), or change nnmamba.patch_size."
            )
        return self.proj(x)                                    # (B, d_model, D/p, H/p, W/p)


class NNMamba(nn.Module):
    """Patch-embed → (spatial | frequency) tokens → shared Mamba → logit."""

    SPATIAL_CHANNELS = 3
    # Canonical stream order: with both active the spatial tokens are scanned first.
    STREAM_ORDER = ("spatial", "frequency")

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (128, 128, 128),
        streams: str | None = None,
        use_frequency: bool | None = None,
        patch_size: int = 16,
        d_model: int = 192,
        mamba_layers: int = 4,
        d_state: int = 16,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.streams = resolve_stream_mode(streams, use_frequency)
        self.active_streams = active_streams(self.streams, self.STREAM_ORDER)
        self._stream_row = {name: i for i, name in enumerate(self.active_streams)}
        self.use_spatial = "spatial" in self._stream_row
        self.use_frequency = "frequency" in self._stream_row

        self.input_shape = tuple(int(s) for s in input_shape)
        self.patch_size = int(patch_size)
        self.d_model = int(d_model)
        self.bidirectional = bool(bidirectional)

        self.patch_embed = PatchEmbed3D(self.SPATIAL_CHANNELS, self.d_model, self.patch_size)
        if any(s % self.patch_size for s in self.input_shape):
            raise ValueError(
                f"input_shape {self.input_shape} is not divisible by patch_size="
                f"{self.patch_size}; the positional embedding has no well-defined size."
            )
        self.grid = tuple(s // self.patch_size for s in self.input_shape)
        self.tokens_per_stream = self.grid[0] * self.grid[1] * self.grid[2]

        # A different domain, so the frequency branch gets its own learned
        # projection before its tokens join the spatial ones in one sequence.
        self.freq_proj = nn.Sequential(
            nn.Conv3d(self.d_model, self.d_model, kernel_size=1, bias=False),
            _gn(self.d_model), nn.GELU(),
        ) if self.use_frequency else None

        # Shared token norm: the patch projection's output scale is arbitrary, and
        # without this the positional embedding below would be swamped by it.
        self.token_norm = nn.LayerNorm(self.d_model)
        # One positional embedding PER STREAM. Position means different things in
        # the two (a place in the brain vs. a spatial-frequency band), and having
        # separate tables also marks which stream a token came from — so this
        # replaces the separate stream embedding CNNnnMamba carries.
        self.pos_embed = nn.Parameter(
            torch.zeros(len(self.active_streams), self.tokens_per_stream, self.d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.mamba = nn.Sequential(*[
            make_mamba_layer(self.d_model, d_state=d_state, dropout=dropout,
                             bidirectional=self.bidirectional)
            for _ in range(mamba_layers)
        ])
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )

    @property
    def sequence_length(self) -> int:
        """Tokens the Mamba stack scans per sample — the main cost driver."""
        return len(self.active_streams) * self.tokens_per_stream

    def _tokens(self, feature_map: torch.Tensor, stream: str) -> torch.Tensor:
        """``(B, C, d, h, w)`` → ``(B, N, C)`` normalised, plus this stream's position."""
        b, c = feature_map.shape[0], feature_map.shape[1]
        t = feature_map.reshape(b, c, -1).transpose(1, 2)      # (B, N, C)
        return self.token_norm(t) + self.pos_embed[self._stream_row[stream]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.SPATIAL_CHANNELS:
            raise ValueError(
                f"NNMamba expects {self.SPATIAL_CHANNELS} spatial channels (T1/T2/FLAIR), "
                f"got {x.shape[1]}. The frequency view is computed inside the model from "
                f"the patch embedding, so the dataset must not append FFT channels."
            )
        patches = self.patch_embed(x)                          # (B, d_model, gd, gh, gw)
        if patches.shape[2:] != torch.Size(self.grid):
            raise ValueError(
                f"NNMamba was built for input_shape={self.input_shape} (patch grid "
                f"{self.grid}) but got {tuple(x.shape[-3:])} (grid {tuple(patches.shape[2:])}). "
                f"The positional embedding is tied to the grid — keep data.target_shape "
                f"the same as the one the model was constructed with."
            )

        parts: list[torch.Tensor] = []
        if self.use_spatial:
            parts.append(self._tokens(patches, "spatial"))
        if self.freq_proj is not None:
            parts.append(self._tokens(self.freq_proj(freq_magnitude(patches)), "frequency"))
        tokens = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

        z = self.norm(self.mamba(tokens))                      # (B, n_streams*N, d_model)
        return self.head(z.mean(dim=1))                        # (B, 1)
