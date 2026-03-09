import torch
import torch.nn as nn
from featureExtractor import CascadedMixingTransformer

class ALSTriStreamClassifier(nn.Module):
    """Thin wrapper to expose one clean classifier interface for train/eval scripts."""

    def __init__(self):
        super(ALSTriStreamClassifier, self).__init__()

        # Binary output logit: 0 = control, 1 = ALS.
        # feature_dim=128 matches the reduced SingleModalityEncoder.
        self.model = CascadedMixingTransformer(feature_dim=128, num_classes=1)

    def forward(self, t1, t2, flair):
        return self.model(t1, t2, flair)