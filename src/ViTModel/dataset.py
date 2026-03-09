import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class FeatureSample:
    sample_id: str
    subject_id: str
    path: Path
    label: float


class ALSFeatureDataset(Dataset):
    """
    Loads CNN features extracted from full 3D MRI volumes.
    Each item returns a tensor with 3 modality tokens: [T1, T2, FLAIR].
    """

    def __init__(self, features_dir: str):
        self.features_dir = Path(features_dir)
        self.samples: List[FeatureSample] = []
        self._scan()

    def _scan(self) -> None:
        if not self.features_dir.exists():
            return

        feature_files = sorted(self.features_dir.glob("*_features.pt"))
        for file_path in feature_files:
            payload = torch.load(file_path, map_location="cpu")

            # Support both new and legacy key names.
            t1 = payload["t1_feat"] if "t1_feat" in payload else payload["t1"]
            t2 = payload["t2_feat"] if "t2_feat" in payload else payload["t2"]
            flair = payload["flair_feat"] if "flair_feat" in payload else payload["fl"]

            sample_id = payload["id"] if "id" in payload else file_path.stem.replace("_features", "")
            label = float(payload["label"])
            subject_id = sample_id.split("_")[0]

            # Validate tensor sizes now so training fails early if files are inconsistent.
            _ = self._stack_modalities(t1, t2, flair)
            self.samples.append(
                FeatureSample(
                    sample_id=sample_id,
                    subject_id=subject_id,
                    path=file_path,
                    label=label,
                )
            )

    def _stack_modalities(self, t1: torch.Tensor, t2: torch.Tensor, flair: torch.Tensor) -> torch.Tensor:
        # Flatten in case features were saved as [1, D] instead of [D].
        t1_vec = t1.float().reshape(-1)
        t2_vec = t2.float().reshape(-1)
        flair_vec = flair.float().reshape(-1)
        return torch.stack([t1_vec, t2_vec, flair_vec], dim=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        payload = torch.load(sample.path, map_location="cpu")

        t1 = payload["t1_feat"] if "t1_feat" in payload else payload["t1"]
        t2 = payload["t2_feat"] if "t2_feat" in payload else payload["t2"]
        flair = payload["flair_feat"] if "flair_feat" in payload else payload["fl"]

        x = self._stack_modalities(t1, t2, flair)
        y = torch.tensor(sample.label, dtype=torch.float32)
        return x, y, sample.sample_id

    @property
    def feature_dim(self) -> int:
        if len(self.samples) == 0:
            return 0
        first_payload = torch.load(self.samples[0].path, map_location="cpu")
        t1 = first_payload["t1_feat"] if "t1_feat" in first_payload else first_payload["t1"]
        return int(t1.float().reshape(-1).numel())


def split_indices_by_subject(
    samples: List[FeatureSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split by subject id so different visits from the same person stay in the same split.
    """
    grouped = {}
    for idx, sample in enumerate(samples):
        grouped.setdefault(sample.subject_id, []).append(idx)

    subject_ids = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    n_subjects = len(subject_ids)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    n_test = n_subjects - n_train - n_val

    if n_subjects >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(1, n_test)
        while n_train + n_val + n_test > n_subjects:
            if n_train >= n_val and n_train >= n_test and n_train > 1:
                n_train -= 1
            elif n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1

    train_subjects = set(subject_ids[:n_train])
    val_subjects = set(subject_ids[n_train : n_train + n_val])
    test_subjects = set(subject_ids[n_train + n_val :])

    train_indices, val_indices, test_indices = [], [], []
    for subject_id, idxs in grouped.items():
        if subject_id in train_subjects:
            train_indices.extend(idxs)
        elif subject_id in val_subjects:
            val_indices.extend(idxs)
        elif subject_id in test_subjects:
            test_indices.extend(idxs)

    return sorted(train_indices), sorted(val_indices), sorted(test_indices)
