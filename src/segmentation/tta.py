"""
Test-Time Augmentation (TTA) for improved inference
Applies multiple transformations and averages predictions
"""

import torch
import torch.nn.functional as F
import numpy as np


class TestTimeAugmentation:
    def __init__(self, model, device, num_rotations=4, use_flips=True, use_brightness=False):
        self.model = model
        self.device = device
        self.num_rotations = num_rotations
        self.use_flips = use_flips
        self.use_brightness = use_brightness

    def _rotate_3d(self, tensor, k, dims):
        if k == 0:
            return tensor
        return torch.rot90(tensor, k=k, dims=dims)

    def _flip_3d(self, tensor, dim):
        return torch.flip(tensor, dims=[dim])

    def _brightness_adjust(self, tensor, factor):
        return tensor * factor

    @torch.no_grad()
    def predict(self, image, threshold=0.5):
        self.model.eval()
        predictions = []
        pred = torch.sigmoid(self.model(image.to(self.device)))
        predictions.append(pred.cpu())
        for k in range(1, self.num_rotations):
            rotated = self._rotate_3d(image, k, dims=(2, 3))
            pred = torch.sigmoid(self.model(rotated.to(self.device)))
            pred = self._rotate_3d(pred, -k, dims=(2, 3))
            predictions.append(pred.cpu())
        if self.use_flips:
            for dim in [2, 3, 4]:
                flipped = self._flip_3d(image, dim)
                pred = torch.sigmoid(self.model(flipped.to(self.device)))
                pred = self._flip_3d(pred, dim)
                predictions.append(pred.cpu())
        if self.use_brightness:
            for factor in [0.9, 1.1]:
                adjusted = self._brightness_adjust(image, factor)
                pred = torch.sigmoid(self.model(adjusted.to(self.device)))
                predictions.append(pred.cpu())
        avg_pred = torch.stack(predictions).mean(dim=0)
        mask_binary = (avg_pred > threshold).float()
        return avg_pred, mask_binary

    def predict_batch(self, images, threshold=0.5):
        masks = []
        masks_binary = []
        for i in range(images.shape[0]):
            mask, mask_bin = self.predict(images[i:i+1], threshold)
            masks.append(mask)
            masks_binary.append(mask_bin)
        return masks, masks_binary


class MinimalTTA:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, image, threshold=0.5):
        self.model.eval()
        predictions = []
        pred = torch.sigmoid(self.model(image.to(self.device)))
        predictions.append(pred.cpu())
        flipped_h = torch.flip(image, dims=[3])
        pred = torch.sigmoid(self.model(flipped_h.to(self.device)))
        pred = torch.flip(pred, dims=[3])
        predictions.append(pred.cpu())
        flipped_v = torch.flip(image, dims=[2])
        pred = torch.sigmoid(self.model(flipped_v.to(self.device)))
        pred = torch.flip(pred, dims=[2])
        predictions.append(pred.cpu())
        rotated = torch.rot90(image, k=2, dims=(2, 3))
        pred = torch.sigmoid(self.model(rotated.to(self.device)))
        pred = torch.rot90(pred, k=-2, dims=(2, 3))
        predictions.append(pred.cpu())
        avg_pred = torch.stack(predictions).mean(dim=0)
        mask_binary = (avg_pred > threshold).float()
        return avg_pred, mask_binary


class AdaptiveTTA:
    def __init__(self, model, device, uncertainty_threshold=0.3):
        self.model = model
        self.device = device
        self.uncertainty_threshold = uncertainty_threshold
        self.minimal_tta = MinimalTTA(model, device)
        self.full_tta = TestTimeAugmentation(model, device)

    def _compute_uncertainty(self, predictions):
        pred_stack = torch.stack(predictions)
        variance = pred_stack.var(dim=0)
        return variance.mean().item()

    @torch.no_grad()
    def predict(self, image, threshold=0.5):
        mask, mask_bin = self.minimal_tta.predict(image, threshold)
        self.model.eval()
        predictions = []
        pred = torch.sigmoid(self.model(image.to(self.device))).cpu()
        predictions.append(pred)
        flipped_h = torch.flip(image, dims=[3])
        pred = torch.sigmoid(self.model(flipped_h.to(self.device))).cpu()
        pred = torch.flip(pred, dims=[3])
        predictions.append(pred)
        uncertainty = self._compute_uncertainty(predictions)
        if uncertainty > self.uncertainty_threshold:
            mask, mask_bin = self.full_tta.predict(image, threshold)
        return mask, mask_bin


def ensemble_predictions(predictions, method='mean'):
    pred_stack = torch.stack(predictions)
    if method == 'mean':
        return pred_stack.mean(dim=0)
    elif method == 'max':
        return pred_stack.max(dim=0)[0]
    elif method == 'median':
        return pred_stack.median(dim=0)[0]
    elif method == 'weighted_mean':
        confidence = torch.abs(pred_stack - 0.5)
        weights = confidence / confidence.sum(dim=0, keepdim=True)
        return (pred_stack * weights).sum(dim=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
