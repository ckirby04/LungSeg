"""
Weighted sampling strategies for handling difficult cases
Oversamples cases with small lesions
"""

import json
import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler, Sampler
from pathlib import Path
import nibabel as nib
from scipy import ndimage


# Difficult cases are loaded from JSON if available, otherwise empty
DIFFICULT_CASES = []


def load_difficult_cases(json_path=None):
    """Load difficult case IDs from a JSON file."""
    global DIFFICULT_CASES
    if json_path and Path(json_path).exists():
        with open(json_path) as f:
            DIFFICULT_CASES = json.load(f)
    return DIFFICULT_CASES


def calculate_lesion_volume(mask_path):
    """Calculate total lesion volume in a mask"""
    try:
        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata()
        return np.sum(mask > 0)
    except Exception:
        return 0


def calculate_num_lesions(mask_path):
    """Count number of separate lesions"""
    try:
        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata()
        labeled_mask, num_lesions = ndimage.label(mask > 0)
        return num_lesions
    except Exception:
        return 0


def get_case_weights(dataset, strategy='hybrid', difficulty_multiplier=10.0):
    """
    Calculate sampling weights for each case based on difficulty

    Args:
        dataset: Dataset instance with .cases attribute
        strategy: 'volume', 'difficulty', 'hybrid', or 'uniform'
        difficulty_multiplier: How much more to sample difficult cases
    """
    num_cases = len(dataset.cases)
    weights = np.ones(num_cases)

    if strategy == 'uniform':
        return weights.tolist()

    if strategy in ['volume', 'hybrid']:
        volumes = []
        for case_dir in dataset.cases:
            mask_path = case_dir / "seg.nii.gz"
            volume = calculate_lesion_volume(mask_path)
            volumes.append(volume)

        volumes = np.array(volumes)
        volume_weights = 1.0 / (volumes + 100)
        volume_weights = volume_weights / volume_weights.mean()

        if strategy == 'volume':
            weights = volume_weights

    if strategy in ['difficulty', 'hybrid']:
        difficulty_weights = np.ones(num_cases)
        for idx, case_dir in enumerate(dataset.cases):
            case_id = case_dir.name
            if case_id in DIFFICULT_CASES:
                difficulty_weights[idx] = difficulty_multiplier

        if strategy == 'difficulty':
            weights = difficulty_weights
        elif strategy == 'hybrid':
            weights = volume_weights * difficulty_weights

    weights = weights * num_cases / weights.sum()
    return weights.tolist()


def get_stratified_weights(dataset, num_bins=5, boost_small=2.0):
    """Stratified sampling based on lesion size bins"""
    volumes = []
    for case_dir in dataset.cases:
        mask_path = case_dir / "seg.nii.gz"
        volume = calculate_lesion_volume(mask_path)
        volumes.append(volume)

    volumes = np.array(volumes)
    bins = np.percentile(volumes, np.linspace(0, 100, num_bins + 1))
    bin_indices = np.digitize(volumes, bins[1:-1])
    bin_weights = np.linspace(boost_small, 1.0, num_bins)
    weights = bin_weights[bin_indices]

    for idx, case_dir in enumerate(dataset.cases):
        if case_dir.name in DIFFICULT_CASES:
            weights[idx] *= 2.0

    weights = weights * len(dataset) / weights.sum()
    return weights.tolist()


def create_weighted_sampler(dataset, strategy='hybrid', **kwargs):
    """Create a WeightedRandomSampler for the dataset"""
    if strategy == 'stratified':
        weights = get_stratified_weights(dataset, **kwargs)
    else:
        weights = get_case_weights(dataset, strategy=strategy, **kwargs)

    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, difficult_ratio=0.5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.difficult_ratio = difficult_ratio
        self.difficult_indices = []
        self.easy_indices = []
        for idx, case_dir in enumerate(dataset.cases):
            if case_dir.name in DIFFICULT_CASES:
                self.difficult_indices.append(idx)
            else:
                self.easy_indices.append(idx)
        if not self.difficult_indices:
            self.difficult_indices = list(range(len(dataset.cases)))
        self.num_difficult = max(1, int(batch_size * difficult_ratio))
        self.num_easy = batch_size - self.num_difficult
        self.num_batches = len(dataset) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            difficult_batch = np.random.choice(self.difficult_indices, size=self.num_difficult, replace=True)
            easy_batch = np.random.choice(self.easy_indices, size=self.num_easy, replace=True)
            batch = np.concatenate([difficult_batch, easy_batch])
            np.random.shuffle(batch)
            yield from batch.tolist()

    def __len__(self):
        return self.num_batches * self.batch_size
