"""``nnmamba`` — one stage: 3D patch embedding straight into the Mamba stack.

:mod:`.model` holds the architecture; ``build`` maps a config's ``nnmamba:``
block onto it and is what ``models.registry`` calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch is imported lazily so importing the registry stays cheap
    import torch.nn as nn

from .model import NNMamba

__all__ = ["NNMamba", "build"]


def build(m: dict, *, streams: str = "both", input_shape=(128, 128, 128),
          load_pretrained: bool = True) -> "nn.Module":
    """Build the model from its config block. ``load_pretrained`` is unused —
    this model has no CNN and therefore no MedicalNet weights."""
    return NNMamba(
        input_shape=tuple(input_shape),
        streams=streams,
        patch_size=m.get("patch_size", 16), d_model=m.get("d_model", 192),
        mamba_layers=m.get("mamba_layers", 4), d_state=m.get("d_state", 16),
        dropout=m.get("dropout", 0.3), bidirectional=m.get("bidirectional", True),
    )
