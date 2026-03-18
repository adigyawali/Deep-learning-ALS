"""
featureExtractor.py

CNN backbone: MONAI ResNet50 pretrained with MedicalNet weights
(Med3D: Transfer Learning for 3D Medical Image Analysis, Chen et al. 2019).

Upgraded from ResNet10 to ResNet50 — justified by 32 GB VRAM (RTX 5090).
ResNet50 has 4x the depth and ~8x the parameters of ResNet10, giving a much
larger receptive field and richer feature representations.  MedicalNet
ResNet50 was pretrained on 23 diverse 3D medical datasets including brain MRI.

Architecture contract:
  - Input per modality : (B, 1, D, H, W)  float32, Z-score normalised
  - Output per modality: (B, FEATURE_DIM)  global-average-pooled embedding
  - FEATURE_DIM        : 512  (ResNet50 layer4 outputs 2048 -> projected to 512)
  - Final classifier   : CascadedMixingTransformer -> single logit (BCE loss)
"""

import torch
import torch.nn as nn

try:
    from monai.networks.nets import resnet50
    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False

FEATURE_DIM      = 512
MEDICALNET_HUB   = "Warvito/MedicalNet-models"
MEDICALNET_MODEL = "medicalnet_resnet50_23datasets"


class SingleModalityEncoder(nn.Module):
    """
    MONAI ResNet50 backbone with MedicalNet pretrained weights, producing a
    fixed-length feature vector from one 3D MRI volume.

    Backbone is frozen by default (Phase 1 training).
    Set freeze_backbone=False for Phase 2 fine-tuning.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, freeze_backbone: bool = True):
        super().__init__()
        self.feature_dim = feature_dim

        if not _MONAI_AVAILABLE:
            raise ImportError("MONAI is required: pip install monai")

        self._backbone = resnet50(
            pretrained=False,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1,          # placeholder — replaced below
        )
        self._backbone.fc = nn.Identity()

        self._load_medicalnet_weights()

        if freeze_backbone:
            for param in self._backbone.parameters():
                param.requires_grad = False

        # ResNet50 avgpool output is 2048-d
        self.projection = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def _load_medicalnet_weights(self) -> None:
        print(f"  Loading MedicalNet pretrained weights ({MEDICALNET_MODEL})...")
        try:
            pretrained       = torch.hub.load(MEDICALNET_HUB, MEDICALNET_MODEL,
                                              pretrained=True, verbose=False)
            pretrained_state = pretrained.state_dict()
            backbone_state   = self._backbone.state_dict()

            matched, skipped = {}, []
            for k, v in pretrained_state.items():
                clean_k = k.replace("module.", "")
                if clean_k in backbone_state and backbone_state[clean_k].shape == v.shape:
                    matched[clean_k] = v
                else:
                    skipped.append(clean_k)

            backbone_state.update(matched)
            self._backbone.load_state_dict(backbone_state, strict=False)
            print(f"  MedicalNet weights: {len(matched)} matched, {len(skipped)} skipped.")
        except Exception as exc:
            print(f"  WARNING: Could not load MedicalNet weights ({exc}).\n"
                  "  Continuing with random initialisation.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self._backbone(x))      # (B, feature_dim)


class CascadedMixingTransformer(nn.Module):
    """
    Three SingleModalityEncoders (T1 / T2 / FLAIR) fused by a Transformer,
    then classified by a linear head.

    Scaled up for 32 GB VRAM:
      - feature_dim: 512  (was 256 with ResNet10)
      - transformer: 4 layers, 8 heads, FFN dim 2048  (was 2 layers, 4 heads)
      - dropout: 0.1 throughout (unchanged — dataset is small)

    Output: single logit — use BCEWithLogitsLoss during training,
            sigmoid(logit) >= 0.5 for ALS-positive prediction at inference.
    """

    def __init__(
        self,
        feature_dim: int    = FEATURE_DIM,
        num_classes: int    = 1,
        freeze_backbone: bool = True,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.t1Encoder    = SingleModalityEncoder(feature_dim, freeze_backbone)
        self.t2Encoder    = SingleModalityEncoder(feature_dim, freeze_backbone)
        self.flairEncoder = SingleModalityEncoder(feature_dim, freeze_backbone)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,                         # 8 heads (was 4)
            dim_feedforward=feature_dim * 4, # 2048 (was feature_dim * 2)
            dropout=dropout_prob,
            batch_first=True,
            norm_first=True,                 # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)  # 4 layers (was 2)

        assert num_classes == 1, "num_classes must be 1. Use BCEWithLogitsLoss."
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(feature_dim * 3, 1),
        )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor, flair: torch.Tensor) -> torch.Tensor:
        feat_t1    = self.t1Encoder(t1)
        feat_t2    = self.t2Encoder(t2)
        feat_flair = self.flairEncoder(flair)

        sequence = torch.stack([feat_t1, feat_t2, feat_flair], dim=1)  # (B, 3, D)
        mixed    = self.transformer(sequence)                            # (B, 3, D)
        flat     = mixed.flatten(1)                                      # (B, 3*D)
        return self.classifier(flat)                                     # (B, 1)


if __name__ == "__main__":
    print("Smoke test...")
    B     = 2
    dummy = lambda: torch.randn(B, 1, 128, 128, 128)
    model = CascadedMixingTransformer(freeze_backbone=False)
    model.train()
    out = model(dummy(), dummy(), dummy())
    assert out.shape == (B, 1)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Output     : {out.shape}")
    print(f"Total      : {total:,}")
    print(f"Trainable  : {trainable:,}")
    print("Passed.")