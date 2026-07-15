"""
Config-driven data-augmentation builder.

The single source of truth for augmentations is the root ``config.yaml``
(``augmentations:`` section). This module turns that config into the two MONAI
``Compose`` pipelines the dataset needs, so **no training code has to change to
add / remove / retune an augmentation** — you edit ``config.yaml`` only.

Why two pipelines (this split is not optional for multi-modal MRI):
  * **geometric** — transforms that MOVE voxels (flip / affine / rotate / zoom).
    They are applied ONCE to the stacked ``(3, D, H, W)`` volume so T1/T2/FLAIR
    stay co-registered, and BEFORE the frequency (FFT) channels are computed.
  * **intensity** — transforms that change voxel VALUES only (noise / contrast /
    bias field / blur). Applied AFTER the frequency channels are computed so they
    never corrupt the spectrum.
Each config entry declares its ``group`` so the builder routes it correctly.

Any class in ``monai.transforms`` can be named in the config — the builder looks
it up by name — so new augmentations need no code here either.

If MONAI is not installed the builder returns ``(None, None)`` and the dataset
trains without augmentation (matches the previous behaviour).
"""

from __future__ import annotations

GEOMETRIC = "geometric"
INTENSITY = "intensity"

# Built-in named presets, used only as a FALLBACK when the root config has no
# ``augmentations`` section (keeps older configs / the ``data.aug_level`` flag
# working). Each preset is expressed in the *same* spec format as config.yaml.
_PRESETS: dict[str, list[dict]] = {
    "light": [
        {"name": "RandFlip", "group": GEOMETRIC, "params": {"prob": 0.5, "spatial_axis": 0}},
        {"name": "RandAffine", "group": GEOMETRIC,
         "params": {"prob": 0.5, "rotate_range": [0.1, 0.1, 0.1],
                    "scale_range": [0.05, 0.05, 0.05], "padding_mode": "border"}},
        {"name": "RandGaussianNoise", "group": INTENSITY, "params": {"prob": 0.2, "std": 0.03}},
        {"name": "RandBiasField", "group": INTENSITY, "params": {"prob": 0.2, "coeff_range": [0.0, 0.05]}},
    ],
    "medium": [
        {"name": "RandFlip", "group": GEOMETRIC, "params": {"prob": 0.5, "spatial_axis": 0}},
        {"name": "RandAffine", "group": GEOMETRIC,
         "params": {"prob": 0.7, "translate_range": [3, 3, 3], "rotate_range": [0.087, 0.087, 0.087],
                    "scale_range": [0.1, 0.1, 0.1], "padding_mode": "border"}},
        {"name": "RandGaussianNoise", "group": INTENSITY, "params": {"prob": 0.3, "std": 0.05}},
        {"name": "RandAdjustContrast", "group": INTENSITY, "params": {"prob": 0.3, "gamma": [0.7, 1.5]}},
        {"name": "RandBiasField", "group": INTENSITY, "params": {"prob": 0.3, "coeff_range": [0.0, 0.1]}},
        {"name": "RandGaussianSmooth", "group": INTENSITY,
         "params": {"prob": 0.15, "sigma_x": [0.5, 1.0], "sigma_y": [0.5, 1.0], "sigma_z": [0.5, 1.0]}},
    ],
    "heavy": [
        {"name": "RandFlip", "group": GEOMETRIC, "params": {"prob": 0.5, "spatial_axis": 0}},
        {"name": "RandFlip", "group": GEOMETRIC, "params": {"prob": 0.5, "spatial_axis": 1}},
        {"name": "RandFlip", "group": GEOMETRIC, "params": {"prob": 0.5, "spatial_axis": 2}},
        {"name": "RandAffine", "group": GEOMETRIC,
         "params": {"prob": 0.8, "translate_range": [4, 4, 4], "rotate_range": [0.2, 0.2, 0.2],
                    "scale_range": [0.15, 0.15, 0.15], "padding_mode": "border"}},
        {"name": "RandGaussianNoise", "group": INTENSITY, "params": {"prob": 0.4, "std": 0.07}},
        {"name": "RandAdjustContrast", "group": INTENSITY, "params": {"prob": 0.4, "gamma": [0.6, 1.6]}},
        {"name": "RandBiasField", "group": INTENSITY, "params": {"prob": 0.4, "coeff_range": [0.0, 0.15]}},
        {"name": "RandGaussianSmooth", "group": INTENSITY,
         "params": {"prob": 0.2, "sigma_x": [0.5, 1.5], "sigma_y": [0.5, 1.5], "sigma_z": [0.5, 1.5]}},
    ],
}


def _split_by_group(specs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Filter to enabled specs and partition into (geometric, intensity)."""
    geom: list[dict] = []
    inten: list[dict] = []
    for s in specs:
        if not isinstance(s, dict) or "name" not in s:
            raise ValueError(f"Each augmentation must be a mapping with a 'name'; got {s!r}.")
        if not s.get("enabled", True):
            continue
        group = str(s.get("group", INTENSITY)).lower()
        if group not in (GEOMETRIC, INTENSITY):
            raise ValueError(
                f"Augmentation {s['name']!r} has group={group!r}; must be "
                f"'{GEOMETRIC}' or '{INTENSITY}' (see config.yaml comments)."
            )
        (geom if group == GEOMETRIC else inten).append(s)
    return geom, inten


def _instantiate(specs: list[dict], monai_transforms) -> list:
    """Build MONAI transform instances from spec dicts, with clear errors."""
    out = []
    for s in specs:
        name = s["name"]
        params = s.get("params") or {}
        cls = getattr(monai_transforms, name, None)
        if cls is None:
            raise ValueError(
                f"Unknown augmentation '{name}' in config.yaml — it must be a class in "
                f"monai.transforms (e.g. RandFlip, RandAffine, RandGaussianNoise)."
            )
        try:
            out.append(cls(**params))
        except TypeError as exc:
            raise ValueError(
                f"Bad params for augmentation '{name}' in config.yaml: {exc}. "
                f"Check the 'params' block for {name}."
            ) from exc
    return out


def build_transforms(aug_config: dict | None = None, *, aug_level: str = "medium"):
    """Return ``(geometric_compose, intensity_compose)`` for the training dataset.

    Parameters
    ----------
    aug_config : dict | None
        The ``augmentations`` section from ``config.yaml``. When present it is the
        single source of truth. Recognised keys:
          * ``enabled`` (bool)   — master switch; False → no augmentation.
          * ``transforms`` (list) — the per-augmentation specs (name/enabled/
            group/params). If omitted, the ``aug_level`` preset is used.
    aug_level : str
        Fallback preset name (``light`` / ``medium`` / ``heavy``) used only when
        ``aug_config`` is None or has no ``transforms`` list. Preserves the older
        ``data.aug_level`` behaviour.

    Either element may be ``None`` (that group has no active transforms, or MONAI
    is not installed).
    """
    try:
        from monai import transforms as monai_transforms
        from monai.transforms import Compose
    except ImportError:
        return None, None

    if aug_config is not None:
        if not aug_config.get("enabled", True):
            return None, None  # augmentation switched off entirely
        specs = aug_config.get("transforms")
        if specs is None:
            specs = _PRESETS.get(aug_level, _PRESETS["medium"])
    else:
        specs = _PRESETS.get(aug_level, _PRESETS["medium"])

    geom_specs, inten_specs = _split_by_group(specs)
    geom = _instantiate(geom_specs, monai_transforms)
    inten = _instantiate(inten_specs, monai_transforms)
    return (Compose(geom) if geom else None, Compose(inten) if inten else None)
