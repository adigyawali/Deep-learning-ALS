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
        
        # Project identity connection if dimensions differ
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
    """Extract one compact feature vector from one MRI modality volume."""

    def __init__(self):
        super(SingleModalityEncoder, self).__init__()
        # Initial convolution
        self.pre_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks for feature extraction
        # Gradually increase channels and downsample (stride=2)
        self.layer1 = ResidualBlock3D(32, 32)
        self.layer2 = ResidualBlock3D(32, 64, stride=2)
        self.layer3 = ResidualBlock3D(64, 128, stride=2)
        self.layer4 = ResidualBlock3D(128, 256, stride=2)
        
        # Global pooling to flatten 3D maps to vector
        self.globalPool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.pre_conv(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return torch.flatten(self.globalPool(x), 1)


class CascadedMixingTransformer(nn.Module):
    """Three-branch encoder + transformer mixer + binary classification head."""

    def __init__(self, feature_dim=256, num_classes=2, dropout_prob=0.2):
        super(CascadedMixingTransformer, self).__init__()
        
        # Independent encoders per modality
        self.t1Encoder = SingleModalityEncoder()
        self.t2Encoder = SingleModalityEncoder()
        self.flairEncoder = SingleModalityEncoder()
        
        self.dropout_prob = dropout_prob

        # Transformer mixes cross-modality context between [T1, T2, FLAIR] tokens.
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification head
        self.classifier = nn.Linear(feature_dim * 3, num_classes)

    def forward(self, t1, t2, flair):
        feat_t1 = self.t1Encoder(t1)
        feat_t2 = self.t2Encoder(t2)
        feat_flair = self.flairEncoder(flair)

        # During training, randomly mask modality vectors so the model does not overfit to one scan type.
        if self.training:
            feat_t1 = self.applyModalityDropout(feat_t1)
            feat_t2 = self.applyModalityDropout(feat_t2)
            feat_flair = self.applyModalityDropout(feat_flair)

        # Stack features for transformer processing
        sequence = torch.stack([feat_t1, feat_t2, feat_flair], dim=1)

        # Mix features via transformer
        mixed_features = self.transformer(sequence)

        # Flatten and classify
        flat_features = torch.flatten(mixed_features, 1)
        output = self.classifier(flat_features)
        
        return output

    def applyModalityDropout(self, x):
        # Drop per sample (not whole batch) so augmentation is more diverse.
        if self.dropout_prob <= 0:
            return x
        original = x
        keep_mask = (torch.rand(x.size(0), 1, device=x.device) >= self.dropout_prob).float()
        x = x * keep_mask
        # Safety fallback: if every sample was dropped, keep original features.
        if keep_mask.sum() == 0:
            return original
        return x


# --- 3. VERIFICATION ---

# this block runs only when the script is executed directly
if __name__ == "__main__":
    # create sample 3d inputs with a small size for quick testing
    dummyT1 = torch.randn(2, 1, 64, 64, 64) 
    dummyT2 = torch.randn(2, 1, 64, 64, 64)
    dummyFlair = torch.randn(2, 1, 64, 64, 64)
    
    # create an instance of the full model
    model = CascadedMixingTransformer(num_classes=2)
    
    # set the model to training mode
    model.train()
    
    # perform a forward pass to see the output
    output = model(dummyT1, dummyT2, dummyFlair)
    
    # print the shape of the output to verify correctness
    print("Model Output Shape:", output.shape) 
    # confirm the model worked as expected
    print("Success! The model processed inputs and generated an output.")
