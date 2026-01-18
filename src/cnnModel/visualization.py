import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

def plot_training_history(history, save_path=None):
    """
    Plots training and validation accuracy/loss.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out", pred_index=None):
    """
    Generates a Grad-CAM heatmap for a specific input image.
    """
    # 1. Create a model that maps the input image to the activations of the last conv layer
    #    as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Compute the gradient of the top predicted class for our input image
    #    with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. This is the gradient of the output neuron (top predicted or chosen)
    #    with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Vector of weights: mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. We multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    
    # 6. Squeeze to remove single dimensions
    heatmap = tf.squeeze(heatmap)

    # 7. Apply ReLU to keep only features that have a positive influence
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, alpha=0.4):
    """
    Superimposes the heatmap on the original image.
    Since our input is 3-channel (T1, T2, FLAIR), we will visualize 
    it on the first channel (T1) for clarity, or an average.
    """
    # Load the original image
    img = cv2.imread(img_path)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Resize the heatmap to the size of the original image
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    return superimposed_img

def visualize_prediction(model, img_array, label, layer_name="conv5_block3_out", save_path=None):
    """
    Wrapper to generate and plot Grad-CAM for a single sample.
    img_array: (1, 128, 128, 3)
    """
    # Generate Heatmap
    heatmap = make_gradcam_heatmap(img_array, model, layer_name)
    
    # Prepare display
    # Extract the T1 channel (index 0) for the background grayscale image
    t1_slice = img_array[0, :, :, 0]
    
    # Rescale heatmap to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB) # Matplotlib uses RGB

    # Resize heatmap to match image size (128x128)
    jet = cv2.resize(jet, (t1_slice.shape[1], t1_slice.shape[0]))
    
    # Overlay
    # We convert T1 to RGB so we can overlay color
    t1_rgb = np.stack([t1_slice, t1_slice, t1_slice], axis=-1)
    # Normalize T1 to 0-255
    t1_rgb = (t1_rgb - t1_rgb.min()) / (t1_rgb.max() - t1_rgb.min() + 1e-8)
    t1_rgb = np.uint8(255 * t1_rgb)
    
    superimposed = cv2.addWeighted(t1_rgb, 0.6, jet, 0.4, 0)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original T1
    axes[0].imshow(t1_slice, cmap='gray')
    axes[0].set_title(f"Input T1 Slice\nTrue Label: {label}")
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(superimposed)
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)
    conf = np.max(prediction)
    
    axes[2].set_title(f"Overlay\nPred: {pred_class} ({conf:.2f})")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Grad-CAM saved to {save_path}")
    else:
        plt.show()
    plt.close()
