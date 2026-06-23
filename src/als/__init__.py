"""ALS multi-modal MRI classification package.

Two interchangeable two-stage models share one preprocessing, split, training,
checkpointing, metric, and evaluation core:

  * ``cnn_vit``     — tri-stream MedicalNet ResNet50 features → spatial ViT
  * ``cnn_nnmamba`` — end-to-end 3D CNN stem → Mamba (selective SSM) classifier

Run everything through the top-level ``experiment.py`` driver.
"""

__version__ = "0.2.0"
