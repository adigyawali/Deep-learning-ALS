"""
Config-driven MixUp, applied by the trainer to training batches only.

    x̃ = λ·xᵢ + (1−λ)·xⱼ      ỹ = λ·yᵢ + (1−λ)·yⱼ      λ ~ Beta(α, α)

It needs two samples at once, so unlike everything in ``augment.py`` it cannot
live in a per-sample MONAI ``Compose``. Configured under ``augmentations.mixup``
in the root ``config.yaml``, and off by default.

Three choices worth knowing:

  * **One λ and one pairing per batch, shared across every input tensor.** T1/T2/
    FLAIR are co-registered views of one subject, so mixing them independently
    would fuse three different brains into one "sample".
  * **Soft targets, not a two-term loss.** BCE is linear in the target, so
    evaluating the loss once against the mixed target is exactly equivalent to
    ``λ·L(ŷ,yᵢ) + (1−λ)·L(ŷ,yⱼ)`` — including with ``pos_weight`` and label
    smoothing. That keeps every loss and the trainer's contract unchanged.
  * **α = 0.2** is U-shaped: most batches barely mix, a few mix near 50/50. The
    conservative small-data setting. Raise toward 0.4 for more regularisation.

While mixup is on the training ROC-AUC is reported as ``nan`` (the targets are
soft); validation is never mixed.
"""

from __future__ import annotations

from typing import Sequence

import torch


class MixUp:
    """Mix a training batch and its labels in place of the identity.

    Call as ``mixed_inputs, mixed_labels = mixup(inputs, labels)`` where
    ``inputs`` is one or more ``(B, ...)`` tensors that all belong to the same
    samples (e.g. the T1/T2/FLAIR triple) and ``labels`` is ``(B, 1)``.

    A batch is left untouched when ``B < 2`` (nothing to pair with) or when the
    per-batch draw exceeds ``prob``.
    """

    def __init__(self, alpha: float = 0.2, prob: float = 1.0):
        if alpha <= 0.0:
            raise ValueError(f"mixup.alpha must be > 0, got {alpha}.")
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"mixup.prob must be in [0,1], got {prob}.")
        self.alpha = float(alpha)
        self.prob = float(prob)
        # Draws use the global torch RNG, which `als.seed.set_seed` seeds, so a
        # run is reproducible from the config's seed like every other augmentation.
        self._beta = torch.distributions.Beta(self.alpha, self.alpha)

    def __call__(
        self, inputs: Sequence[torch.Tensor] | torch.Tensor, labels: torch.Tensor
    ) -> tuple[list[torch.Tensor] | torch.Tensor, torch.Tensor]:
        single = isinstance(inputs, torch.Tensor)
        tensors = [inputs] if single else list(inputs)
        batch_size = labels.size(0)

        if batch_size < 2 or float(torch.rand(())) >= self.prob:
            return (tensors[0] if single else tensors), labels

        lam = float(self._beta.sample())
        perm = torch.randperm(batch_size, device=labels.device)
        mixed = [lam * t + (1.0 - lam) * t[perm] for t in tensors]
        mixed_labels = lam * labels + (1.0 - lam) * labels[perm]
        return (mixed[0] if single else mixed), mixed_labels


def build_mixup(aug_config: dict | None) -> MixUp | None:
    """Return a ``MixUp`` from the ``augmentations`` config section, or None.

    Mirrors ``augment.build_transforms``: the root ``config.yaml`` is the single
    source of truth, the master ``augmentations.enabled`` switch turns this off
    with everything else, and the feature is off unless explicitly enabled (so
    adding it does not silently change an existing training recipe).
    """
    if not aug_config or not aug_config.get("enabled", True):
        return None
    spec = aug_config.get("mixup")
    if not spec or not spec.get("enabled", False):
        return None
    unknown = set(spec) - {"enabled", "alpha", "prob"}
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in augmentations.mixup (config.yaml); "
            f"expected 'enabled', 'alpha', 'prob'."
        )
    return MixUp(alpha=float(spec.get("alpha", 0.2)), prob=float(spec.get("prob", 1.0)))
