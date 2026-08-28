"""Forward/backward shape tests for the ViT, both nnMamba models, and the Mamba block."""

from __future__ import annotations

import pytest
import torch

from als.models import build_volume_model
from als.models.cnn_nnmamba import CNNnnMamba
from als.models.cnn_vit import SpatialMultiModalViT
from als.models.components.mamba_block import BiMambaLayer, MambaLayer, make_mamba
from als.models.components.streams import active_streams, resolve_stream_mode
from als.models.nnmamba import NNMamba


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


def test_nnmamba_frequency_only_stream():
    """streams='frequency' drops the spatial tokens and keeps the sequence single-stream."""
    seen: list[int] = []
    m = CNNnnMamba(streams="frequency", base=8, blocks=2, token_grid=2, mamba_layers=1)
    handle = m.mamba.register_forward_pre_hook(lambda mod, inp: seen.append(inp[0].shape[1]))
    out = m(torch.randn(2, 3, 24, 24, 24))
    handle.remove()
    out.sum().backward()
    assert out.shape == (2, 1) and torch.isfinite(out).all()
    assert seen == [2 ** 3]
    assert m.use_frequency and not m.use_spatial
    assert m.freq_proj is not None
    assert m.stream_embed.shape == (1, m.d_model)      # no dead embedding row


def test_cnn_nnmamba_streams_modes_and_token_counts():
    for mode, expected in (("both", 2 * 2 ** 3), ("spatial", 2 ** 3), ("frequency", 2 ** 3)):
        seen: list[int] = []
        m = CNNnnMamba(streams=mode, base=8, blocks=2, token_grid=2, mamba_layers=1)
        handle = m.mamba.register_forward_pre_hook(lambda mod, inp: seen.append(inp[0].shape[1]))
        m(torch.randn(2, 3, 24, 24, 24))
        handle.remove()
        assert seen == [expected], f"streams={mode}: {seen} != [{expected}]"
        assert m.streams == mode


def test_streams_wins_over_legacy_use_frequency():
    # `streams` is explicit, so the deprecated boolean must not override it.
    m = CNNnnMamba(streams="spatial", use_frequency=True, base=8, blocks=2,
                   token_grid=2, mamba_layers=1)
    assert m.streams == "spatial" and m.freq_proj is None


def test_resolve_stream_mode_and_order():
    assert resolve_stream_mode(None, None) == "both"
    assert resolve_stream_mode(None, True) == "both"
    assert resolve_stream_mode(None, False) == "spatial"
    assert resolve_stream_mode("FREQUENCY ") == "frequency"
    assert active_streams("both") == ("spatial", "frequency")   # spatial is scanned first
    assert active_streams("frequency") == ("frequency",)
    with pytest.raises(ValueError, match="streams must be one of"):
        resolve_stream_mode("fourier")


def test_only_downloadable_backbones_are_allowed():
    # resnet18/34 have no MedicalNet hub weights; asking for one must fail fast
    # (clear ValueError) rather than later blowing up on a missing hub entrypoint.
    from als.models.components.cnn_backbone import _BACKBONES, build_medicalnet_backbone
    assert set(_BACKBONES) == {"resnet10", "resnet50"}
    with pytest.raises(ValueError, match="backbone must be one of"):
        build_medicalnet_backbone("resnet18", load_pretrained=False)


# ── one-stage NNMamba (no CNN) ────────────────────────────────────────────

def _nnmamba(**kw):
    """Tiny NNMamba: 32³ input, patch 16 → a 2³ = 8-token grid per stream."""
    defaults = dict(input_shape=(32, 32, 32), patch_size=16, d_model=16,
                    mamba_layers=1, d_state=8)
    return NNMamba(**{**defaults, **kw})


def test_bi_mamba_layer_forward_backward():
    m = BiMambaLayer(16, d_state=8)
    x = torch.randn(2, 20, 16, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert y.shape == (2, 20, 16) and x.grad is not None and torch.isfinite(y).all()


def test_bi_mamba_layer_sees_later_tokens():
    """The backward scan must make token 0 depend on the LAST token — a plain
    causal MambaLayer cannot, and that is the whole reason this class exists."""
    torch.manual_seed(0)
    bi = BiMambaLayer(8, d_state=4).eval()
    uni = MambaLayer(8, d_state=4).eval()
    x = torch.randn(1, 4, 8)
    x2 = x.clone()
    # Replace the last token, rather than adding a constant to it: both layers
    # pre-norm with LayerNorm, which subtracts the per-token mean and would erase
    # a constant offset before the scan ever saw it.
    x2[:, -1] = torch.randn(8) * 10
    with torch.no_grad():
        moved = (bi(x)[:, 0] - bi(x2)[:, 0]).abs().max().item()
        causal = (uni(x)[:, 0] - uni(x2)[:, 0]).abs().max().item()
    # The state decays over the scan, so the effect is small but must be real.
    assert moved > 1e-4, f"backward scan did not reach token 0 (diff {moved:g})"
    assert causal == 0.0, "a causal MambaLayer must not see later tokens"


def test_nnmamba_onestage_forward_backward():
    m = _nnmamba()
    out = m(torch.randn(2, 3, 32, 32, 32))
    out.sum().backward()
    assert out.shape == (2, 1) and torch.isfinite(out).all()
    assert m.tokens_per_stream == 2 ** 3 and m.grid == (2, 2, 2)


def test_nnmamba_onestage_has_no_cnn_stem():
    """The only thing between voxels and tokens is the patchify projection, and
    its kernel equals its stride (so no patch overlaps another)."""
    m = _nnmamba()
    assert not hasattr(m, "stem")
    convs = [mod for mod in m.patch_embed.modules() if isinstance(mod, torch.nn.Conv3d)]
    assert len(convs) == 1
    assert convs[0].kernel_size == convs[0].stride == (16, 16, 16)


def test_nnmamba_onestage_stream_modes_and_token_counts():
    for mode, expected in (("both", 2 * 2 ** 3), ("spatial", 2 ** 3), ("frequency", 2 ** 3)):
        seen: list[int] = []
        m = _nnmamba(streams=mode)
        handle = m.mamba.register_forward_pre_hook(lambda mod, inp: seen.append(inp[0].shape[1]))
        out = m(torch.randn(2, 3, 32, 32, 32))
        handle.remove()
        assert seen == [expected], f"streams={mode}: {seen} != [{expected}]"
        assert m.pos_embed.shape == (1 if mode != "both" else 2, 2 ** 3, m.d_model)
        assert out.shape == (2, 1)


def test_nnmamba_onestage_frequency_branch_changes_the_output():
    torch.manual_seed(0)
    m = _nnmamba(streams="both").eval()
    x = torch.randn(2, 3, 32, 32, 32)
    with torch.no_grad():
        before = m(x)
        for p in m.freq_proj.parameters():
            p.add_(torch.randn_like(p))
        after = m(x)
    assert not torch.allclose(before, after)


def test_nnmamba_onestage_rejects_indivisible_shapes():
    with pytest.raises(ValueError, match="not divisible by patch_size"):
        NNMamba(input_shape=(30, 32, 32), patch_size=16, d_model=8, mamba_layers=1)
    m = _nnmamba()
    with pytest.raises(ValueError, match="not divisible by patch_size"):
        m(torch.randn(1, 3, 24, 32, 32))


def test_nnmamba_onestage_rejects_wrong_grid_and_channel_count():
    m = _nnmamba()
    with pytest.raises(ValueError, match="was built for input_shape"):
        m(torch.randn(1, 3, 48, 48, 48))          # divisible by 16, but a 3³ grid
    with pytest.raises(ValueError, match="3 spatial channels"):
        m(torch.randn(1, 6, 32, 32, 32))


def test_build_volume_model_dispatch():
    m = build_volume_model("nnmamba", {"patch_size": 16, "d_model": 16, "mamba_layers": 1},
                           streams="spatial", input_shape=(32, 32, 32))
    assert isinstance(m, NNMamba) and m.streams == "spatial"
    c = build_volume_model("cnn_nnmamba", {"base": 8, "blocks": 2, "token_grid": 2,
                                           "mamba_layers": 1}, streams="frequency")
    assert isinstance(c, CNNnnMamba) and c.streams == "frequency"
    with pytest.raises(ValueError, match="unknown model"):
        build_volume_model("cnn_vit", {})


def test_sequence_length_reports_the_scanned_token_count():
    """Both models expose the same cost knob, so the slow-backend warning in
    train_nnmamba can be model-agnostic."""
    assert build_volume_model("cnn_nnmamba", {"token_grid": 4}, streams="both").sequence_length == 128
    assert build_volume_model("cnn_nnmamba", {"token_grid": 4}, streams="spatial").sequence_length == 64
    one = build_volume_model("nnmamba", {"patch_size": 16, "d_model": 16, "mamba_layers": 1},
                             streams="both", input_shape=(128, 128, 128))
    assert one.sequence_length == 2 * 8 ** 3
