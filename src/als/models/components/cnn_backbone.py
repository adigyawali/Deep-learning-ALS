"""
3D CNN backbone (MONAI ResNet, MedicalNet-pretrained) for one MRI modality.

Med3D / MedicalNet (Chen et al. 2019) pretrained 3D ResNets on 23 medical
datasets including brain MRI. We expose ``resnet{10,18,34,50}`` so the backbone
size is a single configurable knob — the first lever to pull when the tri-stream
encoder (three of these live at once) runs the GPU out of memory.

Contract:
  * input  : ``(B, 1, D, H, W)`` z-scored single-modality volume
  * forward : ``(B, feature_dim)`` global-average-pooled, projected embedding
  * forward_features : ``(B, C', D', H', W')`` layer4 map (for the spatial ViT)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

try:
    from monai.networks.nets import resnet10, resnet18, resnet34, resnet50
    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False

FEATURE_DIM = 512
MEDICALNET_HUB = "Warvito/MedicalNet-models"

# MONAI constructor + MedicalNet hub entrypoint + layer4/avgpool channel width.
_BACKBONES = {
    "resnet10": ("resnet10", "medicalnet_resnet10_23datasets", 512),
    "resnet18": ("resnet18", "medicalnet_resnet18_23datasets", 512),
    "resnet34": ("resnet34", "medicalnet_resnet34_23datasets", 512),
    "resnet50": ("resnet50", "medicalnet_resnet50_23datasets", 2048),
}
_CTORS = {"resnet10": lambda **k: resnet10(**k), "resnet18": lambda **k: resnet18(**k),
          "resnet34": lambda **k: resnet34(**k), "resnet50": lambda **k: resnet50(**k)} if _MONAI_AVAILABLE else {}


def _require_pretrained() -> bool:
    """True ⇒ a failed MedicalNet download aborts instead of training random weights.

    Set ``ALS_REQUIRE_PRETRAINED=1`` for the real lab run so a blocked hub
    download (offline box, firewall, trust prompt) hard-fails loudly.
    """
    return os.environ.get("ALS_REQUIRE_PRETRAINED", "").strip().lower() in {"1", "true", "yes"}


class SingleModalityEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet50", feature_dim: int = FEATURE_DIM, freeze_backbone: bool = True):
        super().__init__()
        if not _MONAI_AVAILABLE:
            raise ImportError("MONAI is required: pip install monai")
        if backbone not in _BACKBONES:
            raise ValueError(f"backbone must be one of {list(_BACKBONES)}, got {backbone!r}")
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        ctor_name, self._hub_model, self._backbone_out = _BACKBONES[backbone]

        self._backbone = _CTORS[ctor_name](
            pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=1,
        )
        self._backbone.fc = nn.Identity()
        self._load_medicalnet_weights()

        if freeze_backbone:
            for p in self._backbone.parameters():
                p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(self._backbone_out, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def _fetch_medicalnet_state_dict(self) -> dict:
        local_path = os.environ.get("ALS_MEDICALNET_WEIGHTS", "").strip()
        if local_path:
            p = Path(local_path).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"ALS_MEDICALNET_WEIGHTS={p} does not exist")
            print(f"  Loading MedicalNet weights from local file: {p}")
            raw = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(raw, dict) and "state_dict" in raw:
                return raw["state_dict"]
            return raw if isinstance(raw, dict) else raw.state_dict()
        print(f"  Loading MedicalNet weights ({self._hub_model}) via torch.hub...")
        pretrained = torch.hub.load(MEDICALNET_HUB, self._hub_model, verbose=False, trust_repo=True)
        return pretrained.state_dict()

    def _load_medicalnet_weights(self) -> None:
        require = _require_pretrained()
        # Offline escape hatch for smoke tests / CI: skip the hub download and
        # keep random init. Ignored when ALS_REQUIRE_PRETRAINED is set.
        if not require and os.environ.get("ALS_SKIP_PRETRAINED", "").strip().lower() in {"1", "true", "yes"}:
            print("  ALS_SKIP_PRETRAINED set — skipping MedicalNet download (random init).")
            return
        try:
            pretrained_state = self._fetch_medicalnet_state_dict()
            backbone_state = self._backbone.state_dict()
            matched, skipped = {}, []
            for k, v in pretrained_state.items():
                clean_k = k.replace("module.", "")
                if clean_k in backbone_state and backbone_state[clean_k].shape == v.shape:
                    matched[clean_k] = v
                else:
                    skipped.append(clean_k)
            n_backbone = len(backbone_state)
            coverage = len(matched) / max(1, n_backbone)
            if coverage < 0.5:
                msg = (f"MedicalNet weights loaded but only {len(matched)}/{n_backbone} "
                       f"backbone tensors matched ({coverage:.0%}) — likely a layer-naming "
                       f"mismatch; the backbone would be mostly random.")
                if require:
                    raise RuntimeError(msg)
                print(f"  WARNING: {msg}\n  Continuing anyway.")
            backbone_state.update(matched)
            self._backbone.load_state_dict(backbone_state, strict=False)
            print(f"  MedicalNet weights: {len(matched)}/{n_backbone} matched ({coverage:.0%}), {len(skipped)} skipped.")
        except Exception as exc:
            if require:
                raise RuntimeError(
                    f"Could not load MedicalNet pretrained weights ({exc}). "
                    "ALS_REQUIRE_PRETRAINED is set, so aborting instead of training a "
                    "randomly-initialised backbone. If the torch.hub download is rate-"
                    "limited, download the weights once and set ALS_MEDICALNET_WEIGHTS."
                ) from exc
            print(f"  WARNING: Could not load MedicalNet weights ({exc}).\n"
                  "  Continuing with random initialisation.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self._backbone(x))          # (B, feature_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """ResNet layer4 spatial map *before* global pooling. ``(B, C', D', H', W')``."""
        b = self._backbone
        x = b.conv1(x)
        x = b.bn1(x)
        # MONAI names the stem activation `act`; older versions used `relu`.
        x = (getattr(b, "act", None) or b.relu)(x)
        if not getattr(b, "no_max_pool", False):
            x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        return x
