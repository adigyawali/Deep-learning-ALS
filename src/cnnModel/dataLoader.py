import os
import numpy as np
import cv2
import nibabel as nib
from pathlib import Path
from tensorflow.keras.utils import to_categorical

# Config matching the model
IMG_SIZE = 128

def getMiddleSlice(volume):
    """
    Extracts the middle axial slice from a 3D volume.
    Assumes volume is (H, W, D).
    """
    depth = volume.shape[2]
    mid = depth // 2
    sliceImg = volume[:, :, mid]
    
    # Rotate if necessary (ANTs/NIfTI often loads rotated relative to standard view)
    # We rotate 90 degrees to align with standard viewing for visual consistency
    sliceImg = np.rot90(sliceImg)
    return sliceImg

def resizeSlice(sliceImg, targetSize=IMG_SIZE):
    """
    Resizes the 2D slice to the target dimensions.
    """
    # cv2.resize expects (width, height)
    resized = cv2.resize(sliceImg, (targetSize, targetSize))
    return resized

def loadALSData(processedDir):
    """
    Loads processed data for ALS classification.
    
    Structure:
    Data/processed/
      Subject_Visit/
          Subject_Visit_T1.nii.gz
          Subject_Visit_T2.nii.gz
          Subject_Visit_FLAIR.nii.gz
          
    Returns:
        X: Numpy array of shape (N, 128, 128, 3)
        y: Numpy array of shape (N, 2) [One-hot encoded]
    """
    dataRoot = Path(processedDir)
    
    images = []
    labels = []
    
    # Get all subject folders first to count them
    subjectFolders = [f for f in dataRoot.iterdir() if f.is_dir() and not f.name.startswith('_')]
    totalFolders = len(subjectFolders)
    
    for i, subjectFolder in enumerate(subjectFolders):
        print(f"Loading subject data: {subjectFolder.name} ({i+1}/{totalFolders})...")
            
        # Parse Label from folder name (e.g., C005_V1 or P108_V1)
        folderName = subjectFolder.name
        if folderName.startswith('C'):
            label = 0 # Control
        elif folderName.startswith('P'):
            label = 1 # Patient (ALS)
        else:
            print(f"Skipping unknown subject type: {folderName}")
            continue
            
        # Paths to scans (No 'features' subdirectory, and using .nii.gz)
        # Filenames are like: {Subject_Visit}_T1.nii.gz
        t1Path = subjectFolder / f"{folderName}_T1.nii.gz"
        t2Path = subjectFolder / f"{folderName}_T2.nii.gz"
        flairPath = subjectFolder / f"{folderName}_FLAIR.nii.gz"
        
        if not (t1Path.exists() and t2Path.exists() and flairPath.exists()):
            print(f"Missing modalities for {folderName}, skipping.")
            # Debug print
            print(f"Expected: {t1Path}")
            continue
            
        try:
            # Load 3D Volumes using nibabel
            t1Vol = nib.load(str(t1Path)).get_fdata()
            t2Vol = nib.load(str(t2Path)).get_fdata()
            flairVol = nib.load(str(flairPath)).get_fdata()
            
            # Extract Middle Slices
            t1Slice = getMiddleSlice(t1Vol)
            t2Slice = getMiddleSlice(t2Vol)
            flairSlice = getMiddleSlice(flairVol)
            
            # Resize
            t1Resized = resizeSlice(t1Slice)
            t2Resized = resizeSlice(t2Slice)
            flairResized = resizeSlice(flairSlice)
            
            # Stack into (128, 128, 3)
            # Channel 0: T1, Channel 1: T2, Channel 2: FLAIR
            combined = np.stack([t1Resized, t2Resized, flairResized], axis=-1)
            
            images.append(combined)
            labels.append(label)
            
        except Exception as e:
            print(f"Error loading {folderName}: {e}")
            
    # Convert to Numpy Arrays
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    # One-hot encode labels
    if len(y) > 0:
        y = to_categorical(y, num_classes=2)
        
    print(f"Loaded {len(X)} subjects. Shape: {X.shape}")
    return X, y