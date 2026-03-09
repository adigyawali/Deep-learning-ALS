import os
import glob
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from scipy.ndimage import zoom

class MultiModalALSDataset(Dataset):
    """Loads T1/T2/FLAIR triplets and returns aligned tensors plus binary ALS label."""

    def __init__(self, rootDirectory, transform=None, targetShape=(128, 128, 128)):
        self.rootDirectory = rootDirectory
        # `transform` is treated as a boolean toggle for built-in augmentations.
        self.transform = transform
        self.targetShape = targetShape

        # Each entry stores one subject/visit triplet and its binary label.
        self.samples = []
        self._prepareDataset()

    def _prepareDataset(self):
        # Find all subject folders in the processed dataset directory.
        subjectFolders = sorted(glob.glob(os.path.join(self.rootDirectory, "*")))
        
        for folder in subjectFolders:
            if not os.path.isdir(folder):
                continue
                
            folderName = os.path.basename(folder)
            
            # Expected format: {SubjectID}_{VisitID}, for example C005_V1 or P010_V2.
            parts = folderName.split('_')
            if len(parts) < 2:
                continue
            subject_id = parts[0]
            
            if subject_id.startswith("C"):
                label = 0.0 # Control
            elif subject_id.startswith("P"):
                label = 1.0 # ALS Patient
            elif "_C" in folderName: # Fallback for other conventions
                label = 0.0
            elif "_P" in folderName: # Fallback
                label = 1.0
            else:
                # skip if convention matches neither
                continue
                
            # Build expected modality paths under the subject/visit folder.
            t1Path = os.path.join(folder, f"{folderName}_T1.nii.gz")
            t2Path = os.path.join(folder, f"{folderName}_T2.nii.gz")
            flairPath = os.path.join(folder, f"{folderName}_FLAIR.nii.gz")
            
            # ensure all modalities exist before adding to dataset
            if os.path.exists(t1Path) and os.path.exists(t2Path) and os.path.exists(flairPath):
                self.samples.append({
                    'id': folderName,
                    't1': t1Path,
                    't2': t2Path,
                    'flair': flairPath,
                    'label': label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and preprocess each modality with identical spatial target size.
        t1Volume = self._loadVolume(sample['t1'])
        t2Volume = self._loadVolume(sample['t2'])
        flairVolume = self._loadVolume(sample['flair'])

        # Apply synchronized spatial augmentation across modalities.
        if self.transform:
            # Tensor shape is (1, D, H, W), so data axes are 1/2/3.
            if np.random.rand() > 0.5:
                axis = np.random.choice([1, 2, 3])
                t1Volume = torch.flip(t1Volume, [axis])
                t2Volume = torch.flip(t2Volume, [axis])
                flairVolume = torch.flip(flairVolume, [axis])

            # Rotate all modalities together so voxel correspondence stays intact.
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                dims = np.random.choice([1, 2, 3], size=2, replace=False)
                dims = tuple(dims.tolist())
                t1Volume = torch.rot90(t1Volume, k, dims)
                t2Volume = torch.rot90(t2Volume, k, dims)
                flairVolume = torch.rot90(flairVolume, k, dims)

        label = torch.tensor(sample['label'], dtype=torch.float32)

        return (t1Volume, t2Volume, flairVolume), label

    def _loadVolume(self, path):
        # Read the 3D NIfTI array from disk.
        proxy = nib.load(path)
        data = proxy.get_fdata()

        # Normalize each volume to [0, 1] to stabilize training.
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

        # Resize to one common shape so batches can be stacked.
        currentShape = data.shape
        zoomFactors = [
            self.targetShape[0] / currentShape[0],
            self.targetShape[1] / currentShape[1],
            self.targetShape[2] / currentShape[2]
        ]

        # Linear interpolation keeps preprocessing fast and memory-friendly.
        dataResized = zoom(data, zoomFactors, order=1)

        dataResized = np.expand_dims(dataResized, axis=0)

        return torch.from_numpy(dataResized).float()
