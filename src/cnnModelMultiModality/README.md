# Multi-Modal Tri-Stream CNN for ALS Classification

**Note:** This implementation uses **PyTorch** for its modern ecosystem and dynamic computation graph, offering greater flexibility than the previous TensorFlow version.

## Scientific Objective
The primary goal of this architecture is to enhance the **interpretability** and **diagnostic granularity** of the deep learning model. 

### The Problem: Early Fusion
Traditional approaches often "stack" T1, T2, and FLAIR images into a single 3-channel input (Early Fusion). While computationally efficient, this approach entangles the features of different MRI contrasts early in the network. This makes it mathematically difficult to determine whether a specific prediction was driven by T2 hyperintensity, T1 atrophy, or FLAIR lesions, rendering "Heatmaps" (Grad-CAM) ambiguous.

### The Solution: Late Fusion (Tri-Stream Architecture)
We employ a **Tri-Stream Convolutional Neural Network**. 
1.  **Independent Encoders:** We utilize three distinct parallel CNN branches. Each branch is dedicated solely to one modality (T1, T2, or FLAIR).
2.  **Feature Space Fusion:** The modalities meet only after they have been abstracted into high-level feature vectors (embeddings).
3.  **Preserved Interpretability:** This allows us to backpropagate gradients to specific branches, generating clean, modality-specific attention maps.

## Architecture Pipeline
1.  **Input:** Three separate 3D volumes (T1, T2, FLAIR).
2.  **Feature Extraction:** 
    *   `T1 Encoder` (3D CNN) -> Vector $v_{t1}$
    *   `T2 Encoder` (3D CNN) -> Vector $v_{t2}$
    *   `FLAIR Encoder` (3D CNN) -> Vector $v_{flair}$
3.  **Fusion:** The vectors are concatenated: $V_{fused} = [v_{t1}, v_{t2}, v_{flair}]$.
4.  **Future Integration:** This fused vector represents the entire 3D scan and serves as the input token for the Vision Transformer (ViT).

## Clean Output Structure
All scripts now write to a single location:

- `src/cnn_features/cnn_multimodal/checkpoints/encoder_weights.pth`
- `src/cnn_features/cnn_multimodal/features/<subject_id>_features.pt`
- `src/cnn_features/cnn_multimodal/metrics/*.json`

This keeps checkpoints, extracted features, and evaluation metrics separated.

## Run Order
1. Train CNN encoder and classifier:
   - `python cnnModelMultiModality/train_multimodal.py`
2. Evaluate checkpoint:
   - `python cnnModelMultiModality/evaluate.py`
3. Export modality-wise features for ViT:
   - `python cnnModelMultiModality/generate_features.py`
