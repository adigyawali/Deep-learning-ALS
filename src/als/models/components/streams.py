"""
Which token streams a Mamba stack sees: spatial, frequency, or both.

Both Mamba models build the same two token streams — a spatial view of a feature
map and an FFT log-magnitude view of that same map — and concatenate them into
one sequence. This module owns the vocabulary, so the models, the configs and the
CLI cannot disagree about what ``"spatial"`` means.

The three modes are a one-factor ablation: stem, Mamba stack and head are
identical in all three; only the token sequence changes.

  * ``"both"``      — spatial tokens then frequency tokens (default).
  * ``"spatial"``   — spatial tokens only.
  * ``"frequency"`` — frequency tokens only.

``use_frequency`` is the older boolean; it still maps to ``both`` / ``spatial``
so existing configs and checkpoints keep their meaning.
"""

from __future__ import annotations

STREAM_MODES = ("both", "spatial", "frequency")


def resolve_stream_mode(streams: str | None = None, use_frequency: bool | None = None) -> str:
    """Normalise the ``(streams, use_frequency)`` pair into one mode string.

    ``streams`` wins when both are given; ``use_frequency`` is consulted only
    when ``streams`` is ``None`` (the legacy call style). Both ``None`` → ``"both"``.
    """
    if streams is None:
        if use_frequency is None:
            return "both"
        return "both" if bool(use_frequency) else "spatial"
    mode = str(streams).strip().lower()
    if mode not in STREAM_MODES:
        raise ValueError(
            f"streams must be one of {STREAM_MODES}, got {streams!r}. "
            f"Set it in the config's `data.streams:` field."
        )
    return mode


def active_streams(mode: str, order: tuple[str, ...] = ("spatial", "frequency")) -> tuple[str, ...]:
    """The stream names active under ``mode``, in canonical sequence order.

    Keeping the order fixed matters twice over: Mamba is a causal scan, so
    "spatial first" is an architectural fact rather than a detail, and it pins
    row 0 of the learned stream embedding to the spatial stream in every mode
    where it is present.
    """
    mode = resolve_stream_mode(mode)
    if mode == "both":
        return tuple(order)
    return (mode,)
