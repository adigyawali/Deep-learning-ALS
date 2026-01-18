import os
import glob
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from scipy.ndimage import zoom

# this class manages the loading and preprocessing of 3d mri data
# it inherits from the standard pytorch dataset class
class MultiModalALSDataset(Dataset):
    def __init__(self, rootDirectory, transform=None, targetShape=(128, 128, 128)):
        # path to the processed data folders
        self.rootDirectory = rootDirectory
        # optional transformations (augmentation)
        self.transform = transform
        # desired output shape for the 3d volumes
        self.targetShape = targetShape
        
        # list to store sample information: (subjectId, t1Path, t2Path, flairPath, label)
        self.samples = []
        
        # scan the directory to populate the samples list
        self._prepareDataset()

    # this internal method scans the folder structure to find matched sets of scans
    def _prepareDataset(self):
        # find all subject folders
        subjectFolders = sorted(glob.glob(os.path.join(self.rootDirectory, "*")))
        
        for folder in subjectFolders:
            if not os.path.isdir(folder):
                continue
                
            folderName = os.path.basename(folder)
            
            # determine label based on naming convention
            # starts with 'C' -> control (0), 'P' -> patient (ALS) (1)
            # example: CALSNIC2_EDM_C005_V1
            if "_C" in folderName:
                label = 0.0 # Control
            elif "_P" in folderName:
                label = 1.0 # ALS Patient
            else:
                # skip if convention matches neither
                continue
                
            # construct expected file paths
            # filenames format: {SubjectID}_{VisitID}_{Modality}.nii.gz
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

    # this method returns the total number of samples
    def __len__(self):
        return len(self.samples)

    # this method loads and processes a single sample
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # load the raw nifti files
        t1Volume = self._loadVolume(sample['t1'])
        t2Volume = self._loadVolume(sample['t2'])
        flairVolume = self._loadVolume(sample['flair'])
        
        # convert label to tensor
        label = torch.tensor(sample['label'], dtype=torch.float32)
        
        # return tuple of inputs and the label
        # inputs are (channels, depth, height, width) -> (1, 128, 128, 128)
        return (t1Volume, t2Volume, flairVolume), label

    # this helper method loads a nifti file, normalizes, and resizes it
    def _loadVolume(self, path):
        # load image data using nibabel
        proxy = nib.load(path)
        data = proxy.get_fdata()
        
        # normalize intensity to range [0, 1]
        # this is critical for neural network convergence
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        
        # resize volume to target shape
        # calculate zoom factors for each dimension
        currentShape = data.shape
        zoomFactors = [
            self.targetShape[0] / currentShape[0],
            self.targetShape[1] / currentShape[1],
            self.targetShape[2] / currentShape[2]
        ]
        
        # perform spline interpolation (order 1 for speed, 3 for quality)
        # we use order 1 (linear) to keep preprocessing fast
        dataResized = zoom(data, zoomFactors, order=1)
        
        # add channel dimension: (D, H, W) -> (1, D, H, W)
        dataResized = np.expand_dims(dataResized, axis=0)
        
        # convert to pytorch float tensor
        return torch.from_numpy(dataResized).float()
