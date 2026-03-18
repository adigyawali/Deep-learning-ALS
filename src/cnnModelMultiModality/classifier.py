"""
classifier.py

Contract
--------
  Output : single logit per sample, shape (B, 1)
  Loss   : BCEWithLogitsLoss
  Eval   : sigmoid(logit) >= 0.5 -> ALS positive
"""

import torch.nn as nn
from featureExtractor import CascadedMixingTransformer, FEATURE_DIM


class ALSTriStreamClassifier(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.model = CascadedMixingTransformer(
            feature_dim=FEATURE_DIM,
            num_classes=1,
            freeze_backbone=freeze_backbone,
            dropout_prob=0.1,
        )

    def forward(self, t1, t2, flair):
        return self.model(t1, t2, flair)   # (B, 1)