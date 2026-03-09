import torch
import torch.nn as nn
import random

class ResidualBlock3D(nn.Module):
    """Basic 3D residual block used inside each modality encoder."""

    def __init__(self, inputChannels, outputChannels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(inputChannels, outputChannels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm3d(outputChannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(outputChannels, outputChannels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(outputChannels)
        
        self.downsample = None
        if stride != 1 or inputChannels != outputChannels:
            self.downsample = nn.Sequential(
                nn.Conv3d(inputChannels, outputChannels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(outputChannels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

class SingleModalityEncoder(nn.Module):
    """Extract one compact feature vector from one MRI modality volume.
    
    Reduced from 4 residual blocks to 2 to lower parameter count
    for small datasets (~340 samples). Added spatial dropout for regularization.
    """

    def __init__(self, feature_dim=128):
        super(SingleModalityEncoder, self).__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # Reduced: 2 residual blocks instead of 4
        self.layer1 = ResidualBlock3D(32, 64, stride=2)
        self.layer2 = ResidualBlock3D(64, feature_dim, stride=2)
        
        # Spatial dropout for 3D feature maps (regularization)
        self.dropout = nn.Dropout3d(p=0.3)
        
        self.globalPool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.pre_conv(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        
        return torch.flatten(self.globalPool(x), 1)


class CascadedMixingTransformer(nn.Module):
    """Three-branch encoder + transformer mixer + binary classification head."""

    def __init__(self, feature_dim=128, num_classes=2, dropout_prob=0.2):
        super(CascadedMixingTransformer, self).__init__()
        
        # Independent encoders per modality
        self.t1Encoder = SingleModalityEncoder(feature_dim=feature_dim)
        self.t2Encoder = SingleModalityEncoder(feature_dim=feature_dim)
        self.flairEncoder = SingleModalityEncoder(feature_dim=feature_dim)
        
        self.dropout_prob = dropout_prob

        # Transformer mixes cross-modality context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=4, batch_first=True, dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification head with dropout
        self.head_dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(feature_dim * 3, num_classes)

    def forward(self, t1, t2, flair):
        feat_t1 = self.t1Encoder(t1)
        feat_t2 = self.t2Encoder(t2)
        feat_flair = self.flairEncoder(flair)

        if self.training:
            feat_t1 = self.applyModalityDropout(feat_t1)
            feat_t2 = self.applyModalityDropout(feat_t2)
            feat_flair = self.applyModalityDropout(feat_flair)

        sequence = torch.stack([feat_t1, feat_t2, feat_flair], dim=1)
        mixed_features = self.transformer(sequence)

        flat_features = torch.flatten(mixed_features, 1)
        flat_features = self.head_dropout(flat_features)
        output = self.classifier(flat_features)
        
        return output

    def applyModalityDropout(self, x):
        if self.dropout_prob <= 0:
            return x
        original = x
        keep_mask = (torch.rand(x.size(0), 1, device=x.device) >= self.dropout_prob).float()
        x = x * keep_mask
        if keep_mask.sum() == 0:
            return original
        return x


if __name__ == "__main__":
    dummyT1 = torch.randn(2, 1, 64, 64, 64) 
    dummyT2 = torch.randn(2, 1, 64, 64, 64)
    dummyFlair = torch.randn(2, 1, 64, 64, 64)
    
    model = CascadedMixingTransformer(num_classes=2)
    model.train()
    
    output = model(dummyT1, dummyT2, dummyFlair)
    
    print("Model Output Shape:", output.shape) 
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("Success!")