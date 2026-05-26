"""
Shape and forward-pass tests for SpatialMultiModalViT.

Uses a tiny embed_dim / depth so the test runs in milliseconds even on CPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC / "ViTModel"))
_spec = importlib.util.spec_from_file_location("vit_model_module", _SRC / "ViTModel" / "model.py")
_vit_model = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["vit_model_module"] = _vit_model
_spec.loader.exec_module(_vit_model)
SpatialMultiModalViT = _vit_model.SpatialMultiModalViT


def test_vit_forward_shape() -> None:
    m = SpatialMultiModalViT(in_channels=8, spatial_shape=(2, 2, 2), embed_dim=16, depth=2, num_heads=2)
    x = torch.randn(3, 3, 8, 2, 2, 2)  # (B, modalities, C, D, H, W)
    out = m(x)
    assert out.shape == (3, 1)


def test_vit_token_count_includes_cls() -> None:
    m = SpatialMultiModalViT(in_channels=4, spatial_shape=(2, 2, 2), embed_dim=8, depth=1, num_heads=2)
    # 2*2*2 = 8 tokens per modality * 3 modalities + CLS = 25.
    assert m.num_tokens == 1 + 3 * 8


def test_vit_handles_modality_dropout() -> None:
    m = SpatialMultiModalViT(
        in_channels=4, spatial_shape=(2, 2, 2),
        embed_dim=8, depth=1, num_heads=2,
        modality_dropout_prob=0.5,
    )
    m.train()
    x = torch.randn(4, 3, 4, 2, 2, 2)
    out = m(x)
    assert out.shape == (4, 1)
    # Eval mode disables the mask.
    m.eval()
    out2 = m(x)
    assert out2.shape == (4, 1)


def test_vit_rejects_wrong_modality_count() -> None:
    m = SpatialMultiModalViT(in_channels=4, spatial_shape=(2, 2, 2), embed_dim=8, depth=1, num_heads=2)
    bad = torch.randn(1, 2, 4, 2, 2, 2)  # only 2 modalities
    try:
        m(bad)
    except AssertionError:
        return
    raise AssertionError("Expected an AssertionError for wrong modality count.")
