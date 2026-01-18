import torch
import torch.nn as nn
import random

# this function builds a standard 3d convolutional block
# it processes spatial features within a 3d volume
def buildEncoderBlock(inputChannels, outputChannels):
    # a sequence of operations that form one block
    return nn.Sequential(
        # perform 3d convolution to extract patterns
        nn.Conv3d(inputChannels, outputChannels, kernel_size=3, padding=1),
        # normalize features across the batch for stable training
        nn.BatchNorm3d(outputChannels),
        # apply a non-linear activation function
        nn.ReLU(inplace=True),
        # reduce the spatial dimensions to summarize information
        nn.MaxPool3d(kernel_size=2, stride=2)
    )

# this class defines a single encoder responsible for one modality
# it learns unique features for either t1, t2, or flair scans
class SingleModalityEncoder(nn.Module):
    def __init__(self):
        super(SingleModalityEncoder, self).__init__()
        # first block for initial feature learning
        self.block1 = buildEncoderBlock(1, 32)
        # second block for deeper feature extraction
        self.block2 = buildEncoderBlock(32, 64)
        # third block to capture more complex patterns
        self.block3 = buildEncoderBlock(64, 128)
        # fourth block for very high level semantic features
        self.block4 = buildEncoderBlock(128, 256)
        # global pooling to convert 3d feature maps into a single vector
        self.globalPool = nn.AdaptiveAvgPool3d(1)

    # defines how data flows through this encoder
    def forward(self, x):
        # pass through each convolutional block
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # flatten the pooled result into a one dimensional feature vector
        return torch.flatten(self.globalPool(x), 1)


# this class defines the main transformer based model
# it mixes features from different modalities for a final prediction
class CascadedMixingTransformer(nn.Module):
    def __init__(self, feature_dim=256, num_classes=2, dropout_prob=0.2):
        super(CascadedMixingTransformer, self).__init__()
        
        # create independent cnn encoders for each modality
        self.t1Encoder = SingleModalityEncoder()
        self.t2Encoder = SingleModalityEncoder()
        self.flairEncoder = SingleModalityEncoder()
        
        # store the probability for modality dropout
        self.dropout_prob = dropout_prob

        # define the core transformer encoder layer
        # this layer allows different modalities to talk to each other
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4, batch_first=True)
        # stack multiple transformer layers for robust mixing
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # a simple linear layer for the final classification decision
        self.classifier = nn.Linear(feature_dim * 3, num_classes)

    # defines the complete forward pass of the model
    def forward(self, t1, t2, flair):
        # extract deep features from each 3d scan using their dedicated cnn
        feat_t1 = self.t1Encoder(t1)
        feat_t2 = self.t2Encoder(t2)
        feat_flair = self.flairEncoder(flair)

        # apply modality dropout during training
        # this randomly simulates missing scans to make the model more robust
        if self.training:
            feat_t1 = self.applyModalityDropout(feat_t1)
            feat_t2 = self.applyModalityDropout(feat_t2)
            feat_flair = self.applyModalityDropout(feat_flair)

        # combine the feature vectors into a sequence
        # the transformer will process these three feature vectors as a set
        sequence = torch.stack([feat_t1, feat_t2, feat_flair], dim=1)

        # pass the sequence through the transformer to mix and integrate features
        # this step allows the model to understand inter-modality relationships
        mixed_features = self.transformer(sequence)

        # flatten all mixed features into a single vector
        flat_features = torch.flatten(mixed_features, 1)
        # make the final prediction based on the integrated features
        output = self.classifier(flat_features)
        
        return output

    # randomly zeroes out a modality's feature vector
    # used to simulate data with missing modalities for robustness
    def applyModalityDropout(self, x):
        # decide randomly if this modality should be dropped
        if random.random() < self.dropout_prob:
            # return a tensor of zeros if dropped
            return torch.zeros_like(x) 
        # otherwise return the original features
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