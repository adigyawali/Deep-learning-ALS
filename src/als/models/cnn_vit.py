"""
Spatial multi-modal ViT for ALS classification (stage 2 of CNN→ViT).

Inputs are 3D feature maps from the CNN backbone (one per modality), NOT pooled
vectors. Each spatial position becomes a transformer token, so attention spans
anatomical locations *and* modalities in one sequence.

Token layout for one subject (default 128³ input → ResNet50 layer4, 4³ map):
    [ CLS, T1@(0,0,0)…T1@(3,3,3), T2@…, FL@… ]  →  1 + 3·64 = 193 tokens.
Each token = patch_embed(channel_vector) + positional_embed + modality_embed.

Forward input  : ``(B, 3, C, D, H, W)``  (modalities fixed order T1, T2, FLAIR)
Forward output : ``(B, 1)`` raw logit (use BCEWithLogitsLoss).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialMultiModalViT(nn.Module):
    NUM_MODALITIES = 3

    def __init__(
        self,
        in_channels: int,
        spatial_shape: tuple[int, int, int],
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.15,
        attn_dropout: float = 0.1,
        modality_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_shape = tuple(spatial_shape)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        # Per-modality, per-sample drop probability, training only; at least one
        # modality is always kept.
        self.modality_dropout_prob = float(modality_dropout_prob)

        d, h, w = self.spatial_shape
        self.num_spatial_tokens = d * h * w
        self.num_tokens = 1 + self.NUM_MODALITIES * self.num_spatial_tokens  # +CLS

        self.patch_embed = nn.Linear(in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_spatial_tokens, embed_dim) * 0.02)
        self.modality_embed = nn.Parameter(torch.randn(1, self.NUM_MODALITIES, 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,           # pre-LN: stable for ViTs trained from scratch
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        B, M, C, D, H, W = x.shape
        assert M == self.NUM_MODALITIES, f"Expected {self.NUM_MODALITIES} modalities, got {M}"
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert (D, H, W) == self.spatial_shape, f"Expected spatial shape {self.spatial_shape}, got {(D, H, W)}"
        tokens = x.flatten(start_dim=3).transpose(2, 3)        # (B, M, N, C)
        tokens = self.patch_embed(tokens)                      # (B, M, N, embed_dim)
        tokens = tokens + self.pos_embed.unsqueeze(1) + self.modality_embed
        return tokens.flatten(start_dim=1, end_dim=2)          # (B, M*N, embed_dim)

    def _build_modality_dropout_mask(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if not self.training or self.modality_dropout_prob <= 0.0:
            return None
        M, N = self.NUM_MODALITIES, self.num_spatial_tokens
        drop = torch.rand(batch_size, M, device=device) < self.modality_dropout_prob
        all_dropped = drop.all(dim=1)
        if all_dropped.any():
            idx = torch.nonzero(all_dropped, as_tuple=True)[0]
            rand_keep = torch.randint(0, M, (idx.numel(),), device=device)
            drop[idx, rand_keep] = False
        token_mask = drop.unsqueeze(-1).expand(-1, -1, N).reshape(batch_size, M * N)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        return torch.cat([cls_mask, token_mask], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        tokens = self._tokenize(x)
        cls = self.cls_token.expand(B, -1, -1) + self.cls_pos_embed
        seq = torch.cat([cls, tokens], dim=1)
        seq = self.input_dropout(seq)
        key_padding_mask = self._build_modality_dropout_mask(B, seq.device)
        seq = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        seq = self.norm(seq)
        return self.head(seq[:, 0])                            # (B, 1)
