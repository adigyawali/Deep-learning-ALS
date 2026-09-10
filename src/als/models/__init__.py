"""Model definitions — one package per model, plus the shared building blocks.

    models/
      registry.py     what models exist, their stages, and how each is built
      components/     pieces more than one model uses (CNN backbone, Mamba
                      block, FFT view, stream selection)
      cnn_vit/        tri-stream CNN (stage 1) + spatial ViT (stage 2)
      cnn_nnmamba/    CNN stem -> Mamba over spatial + frequency tokens
      nnmamba/        patch embedding -> Mamba, no CNN

To add a model, read ``registry.py`` — it is the only file outside the new
model's own folder that needs an edit.
"""

from .registry import (
    MODEL_SPECS,
    MODELS,
    VOLUME_MODELS,
    ModelSpec,
    build_volume_model,
    hparam_section,
    spec,
)

__all__ = [
    "MODEL_SPECS",
    "MODELS",
    "VOLUME_MODELS",
    "ModelSpec",
    "build_volume_model",
    "hparam_section",
    "spec",
]
