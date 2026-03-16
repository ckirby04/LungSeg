"""
Data augmentation pipeline for medical image segmentation
Uses MONAI transforms for 3D medical imaging
"""

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandGaussianSharpend,
)
import numpy as np


class AugmentationPipeline:
    """
    Training augmentation pipeline using MONAI

    Args:
        augmentation_probability: Probability of applying each transform (default: 0.3)
    """

    def __init__(self, augmentation_probability=0.3):
        self.prob = augmentation_probability

        self.train_transforms = Compose([
            # Geometric augmentations (applied to both image and mask)
            RandFlipd(
                keys=["image", "mask"],
                spatial_axis=[0, 1, 2],
                prob=0.5,
            ),
            RandRotate90d(
                keys=["image", "mask"],
                prob=self.prob,
                spatial_axes=(0, 1),
            ),
            RandAffined(
                keys=["image", "mask"],
                prob=self.prob,
                rotate_range=[0.1, 0.1, 0.1],
                scale_range=[0.1, 0.1, 0.1],
                mode=["bilinear", "nearest"],
                padding_mode="border",
            ),
            # Intensity augmentations (only on image)
            RandGaussianNoised(
                keys=["image"],
                prob=0.2,
                mean=0.0,
                std=0.1,
            ),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.2,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(
                keys=["image"],
                factors=0.2,
                prob=self.prob,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.1,
                prob=self.prob,
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=self.prob,
                gamma=(0.8, 1.2),
            ),
            RandGaussianSharpend(
                keys=["image"],
                prob=0.2,
                sigma1_x=(0.5, 1.0),
                sigma1_y=(0.5, 1.0),
                sigma1_z=(0.5, 1.0),
                sigma2_x=(0.5, 1.0),
                sigma2_y=(0.5, 1.0),
                sigma2_z=(0.5, 1.0),
            ),
        ])

    def __call__(self, sample):
        data = {"image": sample["image"], "mask": sample["mask"]}
        augmented = self.train_transforms(data)
        sample["image"] = augmented["image"]
        sample["mask"] = augmented["mask"]
        return sample


class ValidationAugmentation:
    """No augmentation for validation"""
    def __call__(self, sample):
        return sample
