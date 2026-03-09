import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Simple cross-attention fusion:
    - Query comes from ViT tokens
    - Key/Value come from CNN modality tokens
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vit_tokens: torch.Tensor, cnn_tokens: torch.Tensor) -> torch.Tensor:
        # vit_tokens: [B, T_vit, D], cnn_tokens: [B, T_cnn, D]
        fused, _ = self.attn(query=vit_tokens, key=cnn_tokens, value=cnn_tokens)
        return self.norm(vit_tokens + self.dropout(fused))


class SimpleMultiModalViT(nn.Module):
    """
    Basic ViT for 3 full-3D modalities using CNN-extracted tokens.
    Input is not 2D slices. Input is [T1, T2, FLAIR] feature vectors from full 3D scans.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Project each modality feature vector into transformer dimension.
        self.feature_proj = nn.Linear(input_dim, embed_dim)

        # Learn one embedding per modality (T1/T2/FLAIR) so the model knows token meaning.
        self.modality_embed = nn.Parameter(torch.randn(1, 3, embed_dim) * 0.02)

        # Standard CLS token for final classification pooling.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.vit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.caf = CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 3, input_dim] for [T1, T2, FLAIR]
        cnn_tokens = self.feature_proj(x)  # [B, 3, D]
        cnn_tokens = cnn_tokens + self.modality_embed

        cls = self.cls_token.expand(x.size(0), -1, -1)
        vit_in = torch.cat([cls, cnn_tokens], dim=1)  # [B, 4, D]

        vit_tokens = self.vit_encoder(vit_in)
        fused_tokens = self.caf(vit_tokens, cnn_tokens)

        cls_out = fused_tokens[:, 0]  # [B, D]
        logits = self.classifier(cls_out)  # [B, 1]
        return logits
