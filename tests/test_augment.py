"""Config-driven augmentation builder: enable/disable, params, add/remove, groups."""

from __future__ import annotations

import pytest
import torch

from als.data.augment import build_transforms

monai = pytest.importorskip("monai")  # the builder is a no-op without MONAI


def _apply(compose, x):
    """Run a MONAI Compose and return a plain tensor."""
    return torch.as_tensor(compose(x)) if compose is not None else x


def test_master_switch_off_disables_everything():
    geom, inten = build_transforms({"enabled": False, "transforms": [
        {"name": "RandFlip", "group": "geometric", "params": {"prob": 1.0, "spatial_axis": 0}},
    ]})
    assert geom is None and inten is None


def test_enabled_flag_per_transform():
    cfg = {"enabled": True, "transforms": [
        {"name": "RandFlip", "enabled": False, "group": "geometric", "params": {"prob": 1.0, "spatial_axis": 0}},
        {"name": "RandGaussianNoise", "enabled": True, "group": "intensity", "params": {"prob": 1.0, "std": 0.1}},
    ]}
    geom, inten = build_transforms(cfg)
    assert geom is None            # the only geometric transform was disabled
    assert inten is not None       # the intensity one survived


def test_groups_route_correctly_and_params_applied():
    # A deterministic (prob=1) flip on axis 0 must actually flip the volume.
    cfg = {"enabled": True, "transforms": [
        {"name": "RandFlip", "group": "geometric", "params": {"prob": 1.0, "spatial_axis": 0}},
    ]}
    geom, inten = build_transforms(cfg)
    assert geom is not None and inten is None
    x = torch.arange(2 * 4 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4, 4)
    out = _apply(geom, x)
    assert torch.allclose(out, torch.flip(x, dims=[1]))   # spatial_axis=0 -> tensor dim 1 (after channel)


def test_add_arbitrary_monai_transform_without_code_change():
    # RandZoom is not in any preset; naming it in config is enough.
    cfg = {"enabled": True, "transforms": [
        {"name": "RandZoom", "group": "geometric", "params": {"prob": 1.0, "min_zoom": 0.9, "max_zoom": 1.1}},
    ]}
    geom, _ = build_transforms(cfg)
    out = _apply(geom, torch.randn(3, 8, 8, 8))
    assert out.shape == (3, 8, 8, 8)


def test_unknown_transform_name_errors_clearly():
    with pytest.raises(ValueError, match="Unknown augmentation 'NotARealTransform'"):
        build_transforms({"enabled": True, "transforms": [
            {"name": "NotARealTransform", "group": "intensity", "params": {}},
        ]})


def test_bad_params_error_names_the_transform():
    with pytest.raises(ValueError, match="Bad params for augmentation 'RandGaussianNoise'"):
        build_transforms({"enabled": True, "transforms": [
            {"name": "RandGaussianNoise", "group": "intensity", "params": {"not_a_kwarg": 1}},
        ]})


def test_bad_group_errors():
    with pytest.raises(ValueError, match="group="):
        build_transforms({"enabled": True, "transforms": [
            {"name": "RandFlip", "group": "nonsense", "params": {"prob": 0.5}},
        ]})


def test_preset_fallback_when_no_transforms_list():
    # aug_config with no 'transforms' key -> named preset is used.
    geom, inten = build_transforms({"enabled": True}, aug_level="heavy")
    assert geom is not None and inten is not None


def test_none_config_uses_preset():
    geom, inten = build_transforms(None, aug_level="light")
    assert geom is not None and inten is not None
