"""
3D FFT log-magnitude — the Mamba models' frequency view.

This lives on the model side on purpose: the frequency view is taken of a
learned feature map, not of the raw voxels, so it is part of the architecture.
The dataset only ever hands over the three spatial modalities.
"""

from __future__ import annotations

import torch


def freq_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Per-channel 3D FFT log-magnitude, fftshifted and z-scored.

    Input  : ``(B, C, D, H, W)`` real feature map.
    Output : ``(B, C, D, H, W)`` real frequency-magnitude map.

    ``log1p`` compresses the DC-dominated dynamic range; ``fftshift`` moves the
    zero frequency to the centre, so after the adaptive pool the token grid is
    ordered by frequency band (low in the middle, high at the edges) rather than
    by the FFT's wrap-around layout. The per-channel z-score puts the result on
    the same scale as the spatial branch it is concatenated with.
    """
    spectrum = torch.fft.fftn(x.float(), dim=(-3, -2, -1))
    mag = torch.log1p(torch.abs(spectrum))
    mag = torch.fft.fftshift(mag, dim=(-3, -2, -1))
    dims = (-3, -2, -1)
    mean = mag.mean(dim=dims, keepdim=True)
    std = mag.std(dim=dims, keepdim=True)
    return ((mag - mean) / (std + 1e-6)).to(x.dtype)
