"""The one place that knows which models exist and how each is wired up.

Every model lives in its own package under ``als/models/`` and is described by
one :class:`ModelSpec` here. The driver, the path layout and the config loader
all read this registry instead of hard-coding model names, so nothing anywhere
else has to enumerate the models.

**To add a model:**

1. Create ``als/models/<name>/`` with the architecture in ``model.py`` and a
   ``build(m, *, streams, input_shape, load_pretrained)`` in ``__init__.py`` that
   maps its config block onto the constructor (copy ``nnmamba/__init__.py``).
2. Add a ``ModelSpec`` below.
3. Add ``configs/<name>.yaml``.

Nothing else needs editing: ``--model <name>`` starts working, the stages run in
the order the spec lists, and outputs land in ``runs/<name>/``. A model whose
training does not fit the shared ``train_nnmamba`` stage (as ``cnn_vit`` does not)
sets ``volume_model=False`` and provides its own stage modules.

This module imports no torch at module level — the builders import lazily — so
reading the registry stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    """Everything the rest of the codebase needs to know about one model."""

    name: str
    summary: str
    #: Pipeline stages, in the order ``all`` runs them.
    stages: tuple[str, ...]
    #: Stages that are valid but not part of ``all`` (e.g. gradcam).
    extra_stages: tuple[str, ...] = ()
    #: Config section holding this model's training hyper-parameters.
    hparams: str = "nnmamba"
    #: Checkpoint prefixes written per fold: ``<prefix>_best.pt``.
    checkpoints: tuple[str, ...] = ("nnmamba",)
    #: True when the model is built from one volume tensor by ``build_volume_model``
    #: and trained by the shared ``train_nnmamba`` stage.
    volume_model: bool = True

    @property
    def default_stages(self) -> list[str]:
        """What ``all`` runs."""
        return list(self.stages)

    @property
    def valid_stages(self) -> set[str]:
        """Every stage name accepted on the command line for this model."""
        return set(self.stages) | set(self.extra_stages)


MODEL_SPECS: dict[str, ModelSpec] = {
    "cnn_vit": ModelSpec(
        name="cnn_vit",
        summary="tri-stream 3D CNN, then a ViT over its feature maps (two stages)",
        stages=("preprocess", "splits", "train_cnn", "extract_features", "train_vit",
                "evaluate", "gradcam"),
        hparams="cnn",
        checkpoints=("cnn", "vit"),
        volume_model=False,
    ),
    "cnn_nnmamba": ModelSpec(
        name="cnn_nnmamba",
        summary="one CNN stem, then a Mamba over spatial + frequency tokens",
        stages=("preprocess", "splits", "train_nnmamba", "evaluate"),
        extra_stages=("gradcam",),
    ),
    "nnmamba": ModelSpec(
        name="nnmamba",
        summary="3D patch embedding straight into Mamba — no CNN",
        stages=("preprocess", "splits", "train_nnmamba", "evaluate"),
        extra_stages=("gradcam",),
    ),
}

#: Model names, in the order they are offered on the command line.
MODELS: tuple[str, ...] = tuple(MODEL_SPECS)
#: Models the shared ``train_nnmamba`` stage can build and train.
VOLUME_MODELS: tuple[str, ...] = tuple(
    name for name, s in MODEL_SPECS.items() if s.volume_model)


def spec(model: str) -> ModelSpec:
    """The :class:`ModelSpec` for ``model``, or a clear error."""
    try:
        return MODEL_SPECS[model]
    except KeyError:
        raise ValueError(f"Unknown model {model!r}. Choices: {MODELS}") from None


def hparam_section(cfg: dict) -> dict:
    """The config block holding this model's training hyper-parameters."""
    return cfg[spec(cfg["model"]).hparams]


def build_volume_model(
    model: str,
    m: dict,
    *,
    streams: str = "both",
    input_shape: tuple[int, int, int] = (128, 128, 128),
    load_pretrained: bool = True,
) -> "nn.Module":
    """Build the raw-volume classifier named by ``model`` from its config block.

    Training and evaluation both come through here, so a checkpoint is always
    rebuilt with the architecture that produced it. Each model package owns the
    config-to-constructor mapping in its own ``build``.

    Parameters
    ----------
    m : dict
        The config's ``nnmamba:`` section. Missing keys fall back to the
        constructor defaults documented in each model module.
    streams : str
        ``both`` / ``spatial`` / ``frequency`` — see ``models.components.streams``.
    input_shape : tuple
        ``data.target_shape``. Only ``nnmamba`` needs it (its positional embedding
        is sized from the patch grid).
    load_pretrained : bool
        ``cnn_nnmamba`` only: fetch MedicalNet weights. False at evaluation time,
        where the checkpoint already carries them (and the box may be offline).
    """
    s = spec(model)
    if not s.volume_model:
        raise ValueError(
            f"{model!r} is not a single-volume model, so it has no build_volume_model "
            f"path. Its stages build it directly. Volume models: {VOLUME_MODELS}")
    if model == "cnn_nnmamba":
        from .cnn_nnmamba import build
    else:
        from .nnmamba import build
    return build(m, streams=streams, input_shape=input_shape,
                 load_pretrained=load_pretrained)
