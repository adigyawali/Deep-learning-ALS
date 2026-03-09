import random
from collections import defaultdict


def split_indices_by_subject(samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split dataset indices at subject level to avoid leakage between visits of the same person.
    """
    grouped = defaultdict(list)
    for idx, sample in enumerate(samples):
        subject_id = sample["id"].split("_")[0]
        grouped[subject_id].append(idx)

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
    val_subjects = set(subject_ids[n_train:n_train + n_val])
    test_subjects = set(subject_ids[n_train + n_val:])

    train_indices = []
    val_indices = []
    test_indices = []
    for subject_id, idxs in grouped.items():
        if subject_id in train_subjects:
            train_indices.extend(idxs)
        elif subject_id in val_subjects:
            val_indices.extend(idxs)
        elif subject_id in test_subjects:
            test_indices.extend(idxs)

    return sorted(train_indices), sorted(val_indices), sorted(test_indices)
