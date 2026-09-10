"""
ALS multi-modal MRI classification.

Three interchangeable models share one preprocessing, split, training,
checkpointing, metric and evaluation core:

  * ``cnn_vit``     — tri-stream MedicalNet ResNet features → spatial ViT
  * ``cnn_nnmamba`` — 3D CNN stem → Mamba over spatial + frequency tokens
  * ``nnmamba``     — the same Mamba stack with the CNN removed

Run everything through the top-level ``experiment.py`` driver.
"""

__version__ = "0.2.0"
