import torch
import torch.nn as nn
from featureExtractor import CascadedMixingTransformer

class ALSTriStreamClassifier(nn.Module):
    """Thin wrapper to expose one clean classifier interface for train/eval scripts."""

    def __init__(self):
        super(ALSTriStreamClassifier, self).__init__()

        # Binary output logit: 0 = control, 1 = ALS.
        self.model = CascadedMixingTransformer(num_classes=1)

    def forward(self, t1, t2, flair):
        # Return raw logits; callers apply sigmoid only for metrics/inference.
        return self.model(t1, t2, flair)
