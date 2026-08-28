"""Model definitions: shared components, ViT, and the two nnMamba variants.

``build_volume_model`` is the one place that maps a config's ``nnmamba:`` block
onto a constructor. Training and evaluation both go through it, so a checkpoint
is always rebuilt with the exact architecture that produced it — previously the
two call sites each spelled out the same twelve keyword arguments and could
drift apart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch is imported lazily below so `als.config` stays light.
    import torch.nn as nn

VOLUME_MODELS = ("cnn_nnmamba", "nnmamba")


def build_volume_model(
    model: str,
    m: dict,
    *,
    streams: str = "both",
    input_shape: tuple[int, int, int] = (128, 128, 128),
    load_pretrained: bool = True,
) -> "nn.Module":
    """Build the raw-volume classifier named by ``model`` from its config block.

    Parameters
    ----------
    model : str
        ``"cnn_nnmamba"`` (two-stage: CNN stem → Mamba) or ``"nnmamba"``
        (one-stage: patch embedding → Mamba, no CNN).
    m : dict
        The config's ``nnmamba:`` section. Missing keys fall back to the
        constructor defaults documented in each model module.
    streams : str
        ``both`` / ``spatial`` / ``frequency`` — see ``models.components.streams``.
    input_shape : tuple
        ``data.target_shape``. Only ``nnmamba`` needs it (its positional embedding
        is sized from the patch grid); ``cnn_nnmamba`` pools to a fixed token grid
        and is resolution-agnostic.
    load_pretrained : bool
        ``cnn_nnmamba`` only: fetch MedicalNet weights. False at evaluation time,
        where the checkpoint already carries them (and the box may be offline).
    """
    if model == "cnn_nnmamba":
        from .cnn_nnmamba import CNNnnMamba
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
    if model == "nnmamba":
        from .nnmamba import NNMamba
        return NNMamba(
            input_shape=tuple(input_shape),
            streams=streams,
            patch_size=m.get("patch_size", 16), d_model=m.get("d_model", 192),
            mamba_layers=m.get("mamba_layers", 4), d_state=m.get("d_state", 16),
            dropout=m.get("dropout", 0.3), bidirectional=m.get("bidirectional", True),
        )
    raise ValueError(f"build_volume_model: unknown model {model!r}. Choices: {VOLUME_MODELS}")
