"""
Config-driven MixUp for the 3D multi-modal classifier.

MixUp (Zhang et al., 2018) trains on convex combinations of pairs of examples:

    x~ = λ·x_i + (1-λ)·x_j        y~ = λ·y_i + (1-λ)·y_j        λ ~ Beta(α, α)

It is a *batch-level* augmentation, so unlike everything in ``augment.py`` it
cannot live inside a per-sample MONAI ``Compose`` — it needs two samples and
their labels at once. It is therefore built here and handed to the trainer,
which applies it to each training batch just before the model forward. Config
still owns it: the ``augmentations.mixup`` block of the root ``config.yaml``,
under the same master ``augmentations.enabled`` switch as everything else.

Why this form is the right one for this project
-----------------------------------------------
* **One λ and one pairing per batch, shared across every input tensor.** T1, T2
  and FLAIR are co-registered views of one subject, so they must be mixed with
  the *same* partner and the *same* weight — mixing them independently would
  fuse three different brains into one "sample". This mirrors the ``geometric``
  group's rule in ``augment.py``.
* **Soft targets, not a two-term loss.** The usual implementation returns
  ``(y_a, y_b, λ)`` and computes ``λ·L(ŷ,y_a) + (1-λ)·L(ŷ,y_b)``. Binary
  cross-entropy is *linear in the target*, so for our losses that is exactly
  equal to evaluating the loss once against the mixed target — including with
  ``pos_weight`` and with ``label_smoothing`` (also linear in the target). The
  single-target form is used because it needs no change to any loss, keeps
  ``pos_weight`` class balancing intact, and keeps the trainer's
  ``(logits, labels)`` contract.
* **Labels stay in [0,1] for one logit.** The task is binary (control vs ALS)
  with a single sigmoid output, so a mixed label is just a soft probability —
  no one-hot expansion is needed, and a control/patient pair yields a target
  strictly between the classes rather than a nonsensical third class.
* **α defaults to 0.2.** Beta(0.2, 0.2) is U-shaped: most batches are barely
  mixed and only a few are near 50/50. That is the conservative setting used
  for small datasets — strong enough to smooth the decision boundary and curb
  the memorisation this project is fighting (see Instructions.md §12.5),
  without destroying the anatomy the encoder has to learn. Raise α toward 0.4
  for more regularisation; α → 0 disables mixing in all but name.
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
