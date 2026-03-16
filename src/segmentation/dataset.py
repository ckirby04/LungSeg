"""
Dataset loader for lung CT segmentation data
Supports single-channel CT input with HU windowing and 3D patch-based loading
"""

import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.ndimage import zoom


class LungCTDataset(Dataset):
    """
    Dataset for lung CT segmentation

    Args:
        data_dir: Path to train or test directory
        patch_size: Size of 3D patches for training (None = full volume)
        target_size: Target size to resize all volumes to (H, W, D). Required for batching.
        transform: Optional transforms
        augment: Whether to apply data augmentation (default: False)
        augmentation_prob: Probability for each augmentation (default: 0.3)
        hu_min: Lower HU window bound (default: -1000.0)
        hu_max: Upper HU window bound (default: 400.0)
    """

    def __init__(
        self,
        data_dir: str,
        patch_size: Optional[Tuple[int, int, int]] = None,
        target_size: Tuple[int, int, int] = (128, 128, 128),
        transform=None,
        augment: bool = False,
        augmentation_prob: float = 0.3,
        hu_min: float = -1000.0,
        hu_max: float = 400.0,
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.target_size = target_size
        self.transform = transform
        self.hu_min = hu_min
        self.hu_max = hu_max

        # Get all case directories (accept any subdirectory with ct.nii.gz)
        self.cases = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and (d / 'ct.nii.gz').exists()
        ])

        # Check if this is training data (has segmentation masks)
        if self.cases:
            sample = self.cases[0]
            self.has_masks = any((sample / f).exists() for f in ['seg.nii.gz', 'seg.npz', 'seg.npy'])
        else:
            self.has_masks = False

        # Create augmentation pipeline if requested
        self.augmentation = None
        if augment:
            try:
                from segmentation.augmentation import AugmentationPipeline
            except ImportError:
                from augmentation import AugmentationPipeline
            self.augmentation = AugmentationPipeline(augmentation_probability=augmentation_prob)
            print(f"Augmentation enabled with probability: {augmentation_prob}")

        print(f"Loaded {len(self.cases)} cases from {data_dir}")
        print(f"Has segmentation masks: {self.has_masks}")

    def __len__(self):
        return len(self.cases)

    def _load_nifti(self, path: Path) -> np.ndarray:
        """Load NIfTI file and return numpy array"""
        nii = nib.load(str(path))
        return np.asarray(nii.dataobj, dtype=np.float32)

    def _load_numpy(self, path: Path) -> np.ndarray:
        """Load numpy file"""
        if path.suffix == '.npz':
            return np.load(str(path))['data'].astype(np.float32)
        return np.load(str(path)).astype(np.float32)

    def _hu_normalize(self, img: np.ndarray) -> np.ndarray:
        """HU windowing normalization for CT data"""
        img = np.clip(img, self.hu_min, self.hu_max)
        img = (img - self.hu_min) / (self.hu_max - self.hu_min)
        return img

    def _resize_volume(self, img: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """Resize 3D volume to target size"""
        if self.target_size is None:
            return img
        current_shape = img.shape
        target_shape = self.target_size
        zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
        order = 0 if is_mask else 1
        return zoom(img, zoom_factors, order=order, mode='nearest')

    def __getitem__(self, idx: int):
        case_dir = self.cases[idx]
        case_id = case_dir.name

        # Load CT volume (try .npz/.npy first, fall back to .nii.gz)
        npz_path = case_dir / "ct.npz"
        npy_path = case_dir / "ct.npy"
        nii_path = case_dir / "ct.nii.gz"

        if npz_path.exists():
            img = self._load_numpy(npz_path)
        elif npy_path.exists():
            img = self._load_numpy(npy_path)
        elif nii_path.exists():
            img = self._load_nifti(nii_path)
        else:
            raise FileNotFoundError(f"Missing CT volume for case {case_id}")

        img = self._resize_volume(img, is_mask=False)
        img = self._hu_normalize(img)

        # Shape: (1, H, W, D) - single channel
        images = np.expand_dims(img, axis=0)

        # Load segmentation mask if available
        mask = None
        if self.has_masks:
            npz_mask = case_dir / "seg.npz"
            npy_mask = case_dir / "seg.npy"
            nii_mask = case_dir / "seg.nii.gz"

            if npz_mask.exists():
                mask = self._load_numpy(npz_mask)
            elif npy_mask.exists():
                mask = self._load_numpy(npy_mask)
            elif nii_mask.exists():
                mask = self._load_nifti(nii_mask)

            if mask is not None:
                mask = self._resize_volume(mask, is_mask=True)
                mask = (mask > 0).astype(np.float32)
                mask = np.expand_dims(mask, axis=0)

        # Apply transforms if provided
        if self.transform:
            sample = {'image': images, 'mask': mask, 'case_id': case_id}
            sample = self.transform(sample)
            images = sample['image']
            mask = sample['mask']
            if hasattr(images, 'numpy'):
                images = images.numpy()
            elif hasattr(images, 'array'):
                images = images.array
            elif not isinstance(images, np.ndarray):
                images = np.array(images)
            if mask is not None:
                if hasattr(mask, 'numpy'):
                    mask = mask.numpy()
                elif hasattr(mask, 'array'):
                    mask = mask.array
                elif not isinstance(mask, np.ndarray):
                    mask = np.array(mask)

        # Extract patch if specified
        if self.patch_size is not None and mask is not None:
            images, mask = self._extract_random_patch(images, mask)

        # Apply augmentation after patch extraction
        if self.augmentation is not None and mask is not None:
            sample = {'image': images, 'mask': mask}
            sample = self.augmentation(sample)
            images = sample['image']
            mask = sample['mask']
            if hasattr(images, 'array'):
                images = images.array
            elif not isinstance(images, np.ndarray):
                images = np.array(images)
            if hasattr(mask, 'array'):
                mask = mask.array
            elif not isinstance(mask, np.ndarray):
                mask = np.array(mask)

        # Convert to torch tensors
        images = torch.from_numpy(images).float()
        if mask is not None:
            mask = torch.from_numpy(mask).float()
            return images, mask, case_id
        else:
            return images, case_id

    def _extract_random_patch(self, images, mask):
        """Extract random 3D patch, biased towards foreground"""
        C, H, W, D = images.shape
        ph, pw, pd = self.patch_size

        if H < ph or W < pw or D < pd:
            pad_h = max(0, ph - H)
            pad_w = max(0, pw - W)
            pad_d = max(0, pd - D)
            images = np.pad(images, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            mask = np.pad(mask, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            C, H, W, D = images.shape

        foreground = np.where(mask[0] > 0)
        if len(foreground[0]) > 0 and np.random.rand() > 0.1:
            idx = np.random.randint(len(foreground[0]))
            ch, cw, cd = foreground[0][idx], foreground[1][idx], foreground[2][idx]
        else:
            ch = np.random.randint(ph//2, H - ph//2)
            cw = np.random.randint(pw//2, W - pw//2)
            cd = np.random.randint(pd//2, D - pd//2)

        h_start = max(0, min(ch - ph//2, H - ph))
        w_start = max(0, min(cw - pw//2, W - pw))
        d_start = max(0, min(cd - pd//2, D - pd))

        img_patch = images[:, h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd]
        mask_patch = mask[:, h_start:h_start+ph, w_start:w_start+pw, d_start:d_start+pd]

        assert img_patch.shape == (C, ph, pw, pd)
        assert mask_patch.shape == (1, ph, pw, pd)
        return img_patch, mask_patch


def get_train_val_split(data_source, val_ratio: float = 0.15, val_split: float = None, seed: int = 42):
    """
    Create train/validation split

    Args:
        data_source: Either path to training directory (str) or dataset length (int)
        val_ratio: Fraction of data for validation
        val_split: Alternative name for val_ratio
        seed: Random seed
    """
    import random
    random.seed(seed)
    split_fraction = val_split if val_split is not None else val_ratio

    if isinstance(data_source, str):
        data_dir = Path(data_source)
        cases = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / 'ct.nii.gz').exists()])
        cases_shuffled = cases.copy()
        random.shuffle(cases_shuffled)
        val_size = int(len(cases) * split_fraction)
        val_cases = cases_shuffled[:val_size]
        train_cases = cases_shuffled[val_size:]
        print(f"Training cases: {len(train_cases)}")
        print(f"Validation cases: {len(val_cases)}")
        return train_cases, val_cases
    elif isinstance(data_source, int):
        n_samples = data_source
        indices = list(range(n_samples))
        random.shuffle(indices)
        val_size = int(n_samples * split_fraction)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        return train_indices, val_indices
    else:
        raise ValueError(f"data_source must be str or int, got {type(data_source)}")


if __name__ == "__main__":
    print("Testing LungCTDataset...")
    dataset = LungCTDataset(data_dir="data/preprocessed/train", patch_size=(96, 96, 96))
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        images, mask, case_id = dataset[0]
        print(f"First sample: {case_id}")
        print(f"Image shape: {images.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
