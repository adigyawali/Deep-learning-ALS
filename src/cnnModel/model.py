import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

# Config for the model
IMG_SIZE = 128
PATCH_SIZE = 32
NUM_CLASSES = 2  # ALS vs Control
NUM_CHANNELS = 3 # T1, T2, FLAIR

def mlp(x, hiddenDim, dropoutRate):
    x = layers.Dense(hiddenDim, activation="gelu")(x)
    x = layers.Dropout(dropoutRate)(x)
    x = layers.Dense(hiddenDim)(x)
    x = layers.Dropout(dropoutRate)(x)
    return x

class ClassToken(layers.Layer):
    def __init__(self, **kwargs):
        super(ClassToken, self).__init__(**kwargs)

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = self.add_weight(
            shape=(1, 1, input_shape[-1]), 
            initializer=w_init, 
            trainable=True,
            name="class_token"
        )

    def call(self, inputs):
        batchSize = tf.shape(inputs)[0]
        hiddenDim = self.w.shape[-1]
        cls = tf.broadcast_to(self.w, [batchSize, 1, hiddenDim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls
    
    def get_config(self):
        config = super().get_config()
        return config

def transformerEncoder(x, numHeads, hiddenDim, dropoutRate):
    skip1 = x
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(num_heads=numHeads, key_dim=hiddenDim)(x, x)
    x = layers.Add()([x, skip1])
    x = layers.Dropout(dropoutRate)(x)

    skip2 = x
    x = layers.LayerNormalization()(x)
    x = mlp(x, hiddenDim, dropoutRate)
    x = layers.Add()([x, skip2])
    x = layers.Dropout(dropoutRate)(x)
    return x

def createHybridModel(
    imageSize=IMG_SIZE, 
    numChannels=NUM_CHANNELS, 
    numClasses=NUM_CLASSES, 
    numLayers=6, 
    hiddenDim=64, 
    numHeads=8, 
    dropoutRate=0.2
):
    """
    Creates the Hybrid ResNet50-ViT model adapted for ALS classification.
    Input: (imageSize, imageSize, 3) -> T1, T2, FLAIR slices.
    Output: Logits for 2 classes (Control, Patient).
    """
    inputShape = (imageSize, imageSize, numChannels)
    inputs = layers.Input(shape=inputShape)

    # --- CNN Part (ResNet50 Backbone) ---
    # We use ResNet50 pre-trained on ImageNet as a feature extractor.
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    
    # In the reference code, they take resnet50.output which is (None, 4, 4, 2048) for 128x128 input
    featureMap = resnet50.output 

    # --- Vision Transformer (ViT) Part ---
    # Patch Embeddings using Conv2D
    patchEmbed = layers.Conv2D(hiddenDim, kernel_size=PATCH_SIZE, padding="same")(featureMap) 
    patchEmbed = layers.BatchNormalization()(patchEmbed)
    
    # Reshape to sequence: (Batch, H*W, Dim)
    # For 128x128 input -> ResNet output 4x4 -> 16 patches
    _, h, w, f = patchEmbed.shape
    numPatches = h * w
    patchEmbed = layers.Reshape((numPatches, f))(patchEmbed)

    # Position Embeddings
    positions = tf.range(start=0, limit=numPatches, delta=1)
    posEmbed = layers.Embedding(input_dim=numPatches, output_dim=hiddenDim)(positions)
    
    # Add Position to Patch
    embed = patchEmbed + posEmbed

    # Add Class Token
    token = ClassToken()(embed)
    x = layers.Concatenate(axis=1)([token, embed])

    # Transformer Encoder Blocks
    for _ in range(numLayers):
        x = transformerEncoder(x, numHeads, hiddenDim, dropoutRate)

    # Classification Head
    x = layers.LayerNormalization()(x)
    # Take the class token (first token)
    x = x[:, 0, :]
    
    logits = layers.Dense(numClasses)(x)

    model = Model(inputs=inputs, outputs=logits, name="Hybrid_RViT_ALS")
    return model
