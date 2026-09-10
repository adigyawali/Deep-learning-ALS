"""``cnn_vit`` — tri-stream 3D CNN, then a Vision Transformer over its feature maps.

Two stages, so two modules:

  * :mod:`.encoder` — ``ALSTriStreamClassifier``, stage 1. One MedicalNet ResNet
    per modality, fused by a small transformer over the three embeddings.
  * :mod:`.vit` — ``SpatialMultiModalViT``, stage 2. Trains on the layer4 maps
    the ``extract_features`` stage dumps, with every spatial position of every
    modality as a token.

The two stages are built by ``stages/train_cnn.py`` and ``stages/train_vit.py``
respectively, because each needs shapes that only its dataset knows.
"""

from .encoder import ALSTriStreamClassifier
from .vit import SpatialMultiModalViT

__all__ = ["ALSTriStreamClassifier", "SpatialMultiModalViT"]
