"""``cnn_nnmamba`` — 3D CNN stem, then one Mamba stack over spatial + frequency tokens.

:mod:`.model` holds the architecture; ``build`` maps a config's ``nnmamba:``
block onto it and is what ``models.registry`` calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch is imported lazily so importing the registry stays cheap
    import torch.nn as nn

from .model import CNNnnMamba

__all__ = ["CNNnnMamba", "build"]


def build(m: dict, *, streams: str = "both", input_shape=(128, 128, 128),
          load_pretrained: bool = True) -> "nn.Module":
    """Build the model from its config block. ``input_shape`` is unused — the
    adaptive pool fixes the token grid, so this model is resolution-agnostic."""
    return CNNnnMamba(
        streams=streams,
        base=m.get("base", 32), blocks=m.get("blocks", 3),
        token_grid=m.get("token_grid", 4), mamba_layers=m.get("mamba_layers", 2),
        d_state=m.get("d_state", 16), dropout=m.get("dropout", 0.1),
        spatial_encoder=m.get("spatial_encoder", "scratch"),
        backbone=m.get("backbone", "resnet10"),
        freeze_backbone=m.get("freeze_backbone", True),
        pretrained_d_model=m.get("pretrained_d_model", 256),
        load_pretrained=load_pretrained,
    )
