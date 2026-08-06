"""Config-driven MixUp: wiring, shared lambda across modalities, label handling."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from als.data.mixup import MixUp, build_mixup


# ── config wiring ──────────────────────────────────────────────────────────

def test_off_by_default_and_when_absent():
    assert build_mixup(None) is None
    assert build_mixup({"enabled": True}) is None                       # no mixup block
    assert build_mixup({"enabled": True, "mixup": {"alpha": 0.4}}) is None   # not enabled


def test_master_switch_disables_mixup():
    cfg = {"enabled": False, "mixup": {"enabled": True, "alpha": 0.4}}
    assert build_mixup(cfg) is None


def test_enabled_reads_alpha_and_prob():
    m = build_mixup({"enabled": True, "mixup": {"enabled": True, "alpha": 0.4, "prob": 0.5}})
    assert isinstance(m, MixUp) and m.alpha == 0.4 and m.prob == 0.5


def test_defaults_when_only_enabled():
    m = build_mixup({"enabled": True, "mixup": {"enabled": True}})
    assert m.alpha == 0.2 and m.prob == 1.0


def test_bad_config_errors_clearly():
    with pytest.raises(ValueError, match="Unknown key"):
        build_mixup({"enabled": True, "mixup": {"enabled": True, "alfa": 0.2}})
    with pytest.raises(ValueError, match="alpha"):
        build_mixup({"enabled": True, "mixup": {"enabled": True, "alpha": 0.0}})
    with pytest.raises(ValueError, match="prob"):
        build_mixup({"enabled": True, "mixup": {"enabled": True, "prob": 1.5}})


# ── behaviour ──────────────────────────────────────────────────────────────

def test_modalities_share_one_lambda_and_one_pairing():
    """T1/T2/FLAIR are one co-registered subject — they must mix identically."""
    torch.manual_seed(0)
    mix = MixUp(alpha=1.0)
    b = 8
    t1 = torch.randn(b, 1, 4, 4, 4)
    t2, fl = t1.clone(), t1.clone()          # identical inputs ...
    y = torch.randint(0, 2, (b, 1)).float()
    (m1, m2, m3), _ = mix((t1, t2, fl), y)
    # ... so identical outputs iff the same lambda and permutation were used.
    assert torch.allclose(m1, m2) and torch.allclose(m2, m3)


def test_labels_are_mixed_with_the_same_lambda_as_the_inputs():
    torch.manual_seed(1)
    mix = MixUp(alpha=1.0)
    b = 16
    x = torch.arange(b, dtype=torch.float32).reshape(b, 1)
    y = torch.arange(b, dtype=torch.float32).reshape(b, 1) * 0.01
    mx, my = mix(x, y)
    # x and y were mixed by the same convex combination, so y must be 1% of x.
    assert torch.allclose(my, mx * 0.01, atol=1e-5)


def test_mixed_labels_stay_in_range_and_shape_is_preserved():
    torch.manual_seed(2)
    mix = MixUp(alpha=0.2)
    x = torch.randn(6, 3, 8, 8, 8)
    y = torch.randint(0, 2, (6, 1)).float()
    mx, my = mix(x, y)
    assert mx.shape == x.shape and my.shape == y.shape
    assert float(my.min()) >= 0.0 and float(my.max()) <= 1.0


def test_batch_of_one_is_left_untouched():
    mix = MixUp(alpha=1.0)
    x = torch.randn(1, 3, 4, 4, 4)
    y = torch.ones(1, 1)
    mx, my = mix(x, y)
    assert torch.equal(mx, x) and torch.equal(my, y)


def test_prob_zero_never_mixes():
    torch.manual_seed(3)
    mix = MixUp(alpha=1.0, prob=0.0)
    x = torch.randn(8, 1, 2, 2, 2)
    y = torch.randint(0, 2, (8, 1)).float()
    for _ in range(10):
        mx, my = mix(x, y)
        assert torch.equal(mx, x) and torch.equal(my, y)


def test_soft_target_bce_equals_the_two_term_mixup_loss():
    """Why the single soft target is used: BCE is linear in the target."""
    torch.manual_seed(4)
    logits = torch.randn(32, 1)
    y_a = torch.randint(0, 2, (32, 1)).float()
    y_b = y_a[torch.randperm(32)]
    pos_weight = torch.tensor([2.5])
    lam = 0.3

    soft = F.binary_cross_entropy_with_logits(
        logits, lam * y_a + (1 - lam) * y_b, pos_weight=pos_weight)
    two_term = (lam * F.binary_cross_entropy_with_logits(logits, y_a, pos_weight=pos_weight)
                + (1 - lam) * F.binary_cross_entropy_with_logits(logits, y_b, pos_weight=pos_weight))
    assert torch.allclose(soft, two_term, atol=1e-6)


def test_forward_adapters_accept_and_apply_mixup():
    from als.stages._common import cnn_forward, vit_forward, volume_forward

    torch.manual_seed(5)
    mix = MixUp(alpha=1.0)
    device = torch.device("cpu")

    tri = torch.nn.Module()
    tri.forward = lambda a, b, c: (a + b + c).flatten(1).mean(1, keepdim=True)
    batch = ((torch.randn(4, 1, 2, 2, 2), torch.randn(4, 1, 2, 2, 2), torch.randn(4, 1, 2, 2, 2)),
             torch.tensor([0.0, 1.0, 1.0, 0.0]))
    logits, labels = cnn_forward(tri, batch, device, mix)
    assert logits.shape == (4, 1) and labels.shape == (4, 1)

    single = torch.nn.Module()
    single.forward = lambda x: x.flatten(1).mean(1, keepdim=True)
    logits, labels = volume_forward(single, (torch.randn(4, 3, 2, 2, 2),
                                             torch.tensor([0.0, 1.0, 1.0, 0.0])), device, mix)
    assert logits.shape == (4, 1) and labels.shape == (4, 1)

    # ViT batches carry a third element (the sample ids) that must pass through.
    logits, labels = vit_forward(single, (torch.randn(4, 3, 2, 2, 2),
                                          torch.tensor([0.0, 1.0, 1.0, 0.0]),
                                          ["a", "b", "c", "d"]), device, mix)
    assert logits.shape == (4, 1) and labels.shape == (4, 1)
