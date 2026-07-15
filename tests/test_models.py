"""Forward/backward shape tests for the ViT, nnMamba, and the Mamba block."""

from __future__ import annotations

import torch

from als.models.cnn_nnmamba import CNNnnMamba
from als.models.cnn_vit import SpatialMultiModalViT
from als.models.components.mamba_block import MambaLayer, make_mamba


def test_mamba_block_forward_backward():
    m = MambaLayer(16, d_state=8)
    x = torch.randn(2, 20, 16, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert y.shape == (2, 20, 16) and x.grad is not None and torch.isfinite(y).all()


def test_make_mamba_is_length_preserving():
    m = make_mamba(12, d_state=8)
    out = m(torch.randn(3, 7, 12))
    assert out.shape == (3, 7, 12)


def test_vit_forward_and_token_count():
    m = SpatialMultiModalViT(in_channels=8, spatial_shape=(2, 2, 2), embed_dim=16, depth=2, num_heads=2)
    assert m.num_tokens == 1 + 3 * 8
    out = m(torch.randn(3, 3, 8, 2, 2, 2))
    out.sum().backward()
    assert out.shape == (3, 1)


def test_vit_modality_dropout_train_and_eval():
    m = SpatialMultiModalViT(in_channels=4, spatial_shape=(2, 2, 2), embed_dim=8, depth=1,
                             num_heads=2, modality_dropout_prob=0.5)
    x = torch.randn(4, 3, 4, 2, 2, 2)
    m.train(); assert m(x).shape == (4, 1)
    m.eval(); assert m(x).shape == (4, 1)


def test_vit_rejects_wrong_modality_count():
    m = SpatialMultiModalViT(in_channels=4, spatial_shape=(2, 2, 2), embed_dim=8, depth=1, num_heads=2)
    try:
        m(torch.randn(1, 2, 4, 2, 2, 2))
    except AssertionError:
        return
    raise AssertionError("expected AssertionError for 2 modalities")


def test_nnmamba_spatial_only():
    m = CNNnnMamba(use_frequency=False, base=8, blocks=2, token_grid=2, mamba_layers=1)
    out = m(torch.randn(2, 3, 24, 24, 24))
    out.sum().backward()
    assert out.shape == (2, 1) and torch.isfinite(out).all()


def test_nnmamba_with_frequency():
    m = CNNnnMamba(use_frequency=True, base=8, blocks=2, token_grid=2, mamba_layers=1)
    out = m(torch.randn(2, 6, 24, 24, 24))
    out.sum().backward()
    assert out.shape == (2, 1)
    # spatial-only model must not accept frequency channels through the spatial slice silently:
    assert m.freq is not None


def test_nnmamba_rejects_bad_spatial_encoder():
    try:
        CNNnnMamba(spatial_encoder="nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown spatial_encoder")


def test_nnmamba_pretrained_encoder_frozen_backbone():
    # conftest sets ALS_SKIP_PRETRAINED=1 so this builds a random (offline) resnet18.
    m = CNNnnMamba(use_frequency=True, spatial_encoder="pretrained", backbone="resnet18",
                   freeze_backbone=True, pretrained_d_model=32, token_grid=2, mamba_layers=1)
    out = m(torch.randn(2, 6, 24, 24, 24))
    out.sum().backward()
    assert out.shape == (2, 1) and torch.isfinite(out).all()
    # Frozen MedicalNet stem: no grad on the backbone, grad on the trainable head.
    assert all(p.grad is None for p in m.spatial.backbone.parameters())
    assert all(not p.requires_grad for p in m.spatial.backbone.parameters())
    assert all(p.grad is not None for p in m.head.parameters())


def test_nnmamba_pretrained_encoder_unfrozen_trains_backbone():
    m = CNNnnMamba(use_frequency=False, spatial_encoder="pretrained", backbone="resnet10",
                   freeze_backbone=False, pretrained_d_model=32, token_grid=2, mamba_layers=1)
    m(torch.randn(1, 3, 24, 24, 24)).sum().backward()
    assert any(p.grad is not None for p in m.spatial.backbone.parameters())
