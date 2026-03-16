"""
Advanced loss functions optimized for small lesion detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        tversky_index = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return 1 - tversky_index


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        bce = F.binary_cross_entropy_with_logits(pred_flat, target_flat, reduction='none')
        pred_prob = torch.sigmoid(pred_flat)
        pt = torch.where(target_flat == 1, pred_prob, 1 - pred_prob)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(target_flat == 1, self.alpha, 1 - self.alpha)
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, ce_ratio=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_ratio = ce_ratio

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + 1) / (pred_flat.sum() + target_flat.sum() + 1)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weights = torch.where(target == 1, self.ce_ratio, 1 - self.ce_ratio)
        weighted_bce = (weights * bce).mean()
        combo = self.alpha * (1 - dice) + self.beta * weighted_bce
        return combo


class SensitivitySpecificityLoss(nn.Module):
    def __init__(self, r=0.7, smooth=1.0):
        super().__init__()
        self.r = r
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        TN = ((1 - target_flat) * (1 - pred_flat)).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        sensitivity = (TP + self.smooth) / (TP + FN + self.smooth)
        specificity = (TN + self.smooth) / (TN + FP + self.smooth)
        loss = 1 - (self.r * sensitivity + (1 - self.r) * specificity)
        return loss


class AsymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        tversky_index = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        focal_tversky = torch.pow(1 - tversky_index, self.gamma)
        return focal_tversky


class SmallLesionOptimizedLoss(nn.Module):
    def __init__(
        self,
        tversky_weight=0.60,
        focal_weight=0.25,
        sensitivity_weight=0.15,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        tversky_gamma=1.5,
        focal_alpha=0.25,
        focal_gamma=2.5,
        sensitivity_r=0.75
    ):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        self.sensitivity_weight = sensitivity_weight
        self.tversky_loss = AsymmetricFocalTverskyLoss(
            alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma
        )
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.sensitivity_loss = SensitivitySpecificityLoss(r=sensitivity_r)

    def forward(self, pred, target):
        tversky = self.tversky_loss(pred, target)
        focal = self.focal_loss(pred, target)
        sensitivity = self.sensitivity_loss(pred, target)
        combined = (
            self.tversky_weight * tversky +
            self.focal_weight * focal +
            self.sensitivity_weight * sensitivity
        )
        return combined


class MultiScaleLoss(nn.Module):
    def __init__(self, loss_fn, weights=None):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights if weights is not None else [1.0, 0.5, 0.25]

    def forward(self, predictions, target):
        if not isinstance(predictions, (list, tuple)):
            return self.loss_fn(predictions, target)
        total_loss = 0
        for i, pred in enumerate(predictions):
            if pred.shape != target.shape:
                target_resized = F.interpolate(target, size=pred.shape[2:], mode='nearest')
            else:
                target_resized = target
            weight = self.weights[i] if i < len(self.weights) else 0.1
            total_loss += weight * self.loss_fn(pred, target_resized)
        return total_loss


def get_loss_function(loss_type, **kwargs):
    losses = {
        'tversky': TverskyLoss,
        'focal': FocalLoss,
        'combo': ComboLoss,
        'sensitivity': SensitivitySpecificityLoss,
        'focal_tversky': AsymmetricFocalTverskyLoss,
        'small_lesion': SmallLesionOptimizedLoss,
    }
    if loss_type not in losses:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(losses.keys())}")
    return losses[loss_type](**kwargs)
