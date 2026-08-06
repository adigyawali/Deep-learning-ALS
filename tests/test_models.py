"""Forward/backward shape tests for the ViT, nnMamba, and the Mamba block."""

from __future__ import annotations

import pytest
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
    """The frequency view is derived inside the model, so the input is still 3 channels."""
    m = CNNnnMamba(use_frequency=True, base=8, blocks=2, token_grid=2, mamba_layers=1)
    out = m(torch.randn(2, 3, 24, 24, 24))
    out.sum().backward()
    assert out.shape == (2, 1) and torch.isfinite(out).all()
    assert m.freq_proj is not None
    assert m.stream_embed.shape == (2, m.d_model)      # one embedding per stream


def test_nnmamba_frequency_doubles_the_token_sequence():
    """Spatial and frequency tokens share ONE Mamba, concatenated along the sequence."""
    seen: list[int] = []

    def spy(module, inputs):
        seen.append(inputs[0].shape[1])

    for use_freq, expected in ((False, 2 ** 3), (True, 2 * 2 ** 3)):
        m = CNNnnMamba(use_frequency=use_freq, base=8, blocks=2, token_grid=2, mamba_layers=1)
        seen.clear()
        handle = m.mamba.register_forward_pre_hook(spy)
        m(torch.randn(2, 3, 24, 24, 24))
        handle.remove()
        assert seen == [expected], f"use_frequency={use_freq}: {seen} != [{expected}]"


def test_nnmamba_rejects_stale_six_channel_input():
    """The old 6-channel dataset output must fail loudly, not be silently sliced."""
    m = CNNnnMamba(use_frequency=True, base=8, blocks=2, token_grid=2, mamba_layers=1)
    try:
        m(torch.randn(2, 6, 24, 24, 24))
    except ValueError as exc:
        assert "3 spatial channels" in str(exc)
        return
    raise AssertionError("expected ValueError for a 6-channel input")


def test_nnmamba_frequency_branch_changes_the_output():
    """The FFT branch must actually reach the logit, not sit inert."""
    torch.manual_seed(0)
    m = CNNnnMamba(use_frequency=True, base=8, blocks=2, token_grid=2, mamba_layers=1).eval()
    x = torch.randn(2, 3, 24, 24, 24)
    with torch.no_grad():
        before = m(x)
        # Perturb only the frequency projection; a spatial-only path would not move.
        for p in m.freq_proj.parameters():
            p.add_(torch.randn_like(p))
        after = m(x)
    assert not torch.allclose(before, after)


def test_nnmamba_rejects_bad_spatial_encoder():
    try:
        CNNnnMamba(spatial_encoder="nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown spatial_encoder")


def test_nnmamba_pretrained_encoder_frozen_backbone():
    # conftest sets ALS_SKIP_PRETRAINED=1 so this builds a random (offline) resnet10.
    m = CNNnnMamba(use_frequency=True, spatial_encoder="pretrained", backbone="resnet10",
                   freeze_backbone=True, pretrained_d_model=32, token_grid=2, mamba_layers=1)
    out = m(torch.randn(2, 3, 24, 24, 24))
    out.sum().backward()
    assert out.shape == (2, 1) and torch.isfinite(out).all()
    # Frozen MedicalNet stem: no grad on the backbone, grad on the trainable head.
    assert all(p.grad is None for p in m.stem.backbone.parameters())
    assert all(not p.requires_grad for p in m.stem.backbone.parameters())
    assert all(p.grad is not None for p in m.head.parameters())


def test_nnmamba_pretrained_encoder_unfrozen_trains_backbone():
    m = CNNnnMamba(use_frequency=False, spatial_encoder="pretrained", backbone="resnet10",
                   freeze_backbone=False, pretrained_d_model=32, token_grid=2, mamba_layers=1)
    m(torch.randn(1, 3, 24, 24, 24)).sum().backward()
    assert any(p.grad is not None for p in m.stem.backbone.parameters())


def test_only_downloadable_backbones_are_allowed():
    # resnet18/34 have no MedicalNet hub weights; asking for one must fail fast
    # (clear ValueError) rather than later blowing up on a missing hub entrypoint.
    from als.models.components.cnn_backbone import _BACKBONES, build_medicalnet_backbone
    assert set(_BACKBONES) == {"resnet10", "resnet50"}
    with pytest.raises(ValueError, match="backbone must be one of"):
        build_medicalnet_backbone("resnet18", load_pretrained=False)
