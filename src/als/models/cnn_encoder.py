"""
Tri-stream 3D CNN classifier (stage 1 of CNN→ViT).

Three independent ``SingleModalityEncoder`` backbones (T1/T2/FLAIR) are fused by
a small Transformer over the three modality embeddings, then classified to a
single logit (BCEWithLogitsLoss). After this stage is trained, the
``extract_features`` stage reads each encoder's ``forward_features`` (layer4
maps) and dumps them for the spatial ViT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .components.cnn_backbone import FEATURE_DIM, SingleModalityEncoder


class ALSTriStreamClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        feature_dim: int = FEATURE_DIM,
        freeze_backbone: bool = True,
        dropout_prob: float = 0.1,
        n_layers: int = 4,
        n_heads: int = 8,
    ):
        super().__init__()
        self.t1Encoder = SingleModalityEncoder(backbone, feature_dim, freeze_backbone)
        self.t2Encoder = SingleModalityEncoder(backbone, feature_dim, freeze_backbone)
        self.flairEncoder = SingleModalityEncoder(backbone, feature_dim, freeze_backbone)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=n_heads, dim_feedforward=feature_dim * 4,
            dropout=dropout_prob, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(feature_dim * 3, 1),
        )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor, flair: torch.Tensor) -> torch.Tensor:
        feats = torch.stack(
            [self.t1Encoder(t1), self.t2Encoder(t2), self.flairEncoder(flair)], dim=1
        )                                            # (B, 3, D)
        mixed = self.transformer(feats)              # (B, 3, D)
        return self.classifier(mixed.flatten(1))     # (B, 1)
