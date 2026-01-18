import torch
import torch.nn as nn
from .featureExtractor import CascadedMixingTransformer

# this class wraps the modern transformer-based fusion model
# it provides the interface expected by the training script
class ALSTriStreamClassifier(nn.Module):
    def __init__(self):
        super(ALSTriStreamClassifier, self).__init__()
        
        # initialize the new transformer-based backbone
        # we set num_classes=1 for binary classification (ALS vs Control)
        # this model handles the encoding, mixing, and classification internally
        self.model = CascadedMixingTransformer(num_classes=1)

    # forward pass delegating to the transformer model
    def forward(self, t1, t2, flair):
        # the model returns the raw logit (score)
        return self.model(t1, t2, flair)
