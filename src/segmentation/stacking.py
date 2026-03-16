"""
Stacking classifier for lung segmentation.

The stacking classifier is a lightweight 3D CNN meta-learner that combines
4 base model predictions into a single fused segmentation.

Base models: 4 patch-size models (8, 12, 24, 36)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from pathlib import Path
from scipy.ndimage import label as ndimage_label, zoom

# =============================================================================
# CONFIG
# =============================================================================

STACKING_MODEL_NAMES = [
    'base_8patch', 'base_12patch', 'base_24patch', 'base_36patch',
]
STACKING_IN_CHANNELS = 6   # 4 predictions + variance + range
STACKING_THRESHOLD = 0.9
STACKING_PATCH_SIZE = 32
STACKING_OVERLAP = 0.5


# =============================================================================
# STACKING CLASSIFIER
# =============================================================================

class StackingClassifier(nn.Module):
    """
    3D CNN meta-learner with residual connections.
    Input: 6 channels (4 model predictions + variance + max-min range)
    Output: 1 channel (final segmentation logits)
    ~25K trainable parameters
    """
    def __init__(self, in_channels=6, mid_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.entry = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
        )
        self.head = nn.Conv3d(mid_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.entry(x)
        x = self.relu(x + self.block1(x))
        x = self.relu(x + self.block2(x))
        return self.head(x)


# =============================================================================
# POST-PROCESSING
# =============================================================================

def postprocess_prediction(binary_mask, min_size=20):
    """Remove connected components smaller than min_size voxels."""
    labeled, n_components = ndimage_label(binary_mask)
    if n_components == 0:
        return binary_mask
    result = np.zeros_like(binary_mask)
    for i in range(1, n_components + 1):
        component = (labeled == i)
        if component.sum() >= min_size:
            result[component] = 1
    return result


# =============================================================================
# SLIDING WINDOW INFERENCE
# =============================================================================

def sliding_window_inference(model, volume, patch_size, device, overlap=0.25):
    """Run inference on full volume using sliding window with overlap."""
    model.eval()
    C, H, W, D = volume.shape
    p = patch_size
    stride = max(int(p * (1 - overlap)), 1)

    if p <= 8:
        batch_size = 512
    elif p <= 12:
        batch_size = 256
    elif p <= 24:
        batch_size = 64
    else:
        batch_size = 32

    pad_h = (p - H % p) % p if H % stride != 0 else 0
    pad_w = (p - W % p) % p if W % stride != 0 else 0
    pad_d = (p - D % p) % p if D % stride != 0 else 0
    orig_H, orig_W, orig_D = H, W, D

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        volume = np.pad(volume, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
        C, H, W, D = volume.shape

    output = np.zeros((H, W, D), dtype=np.float32)
    count = np.zeros((H, W, D), dtype=np.float32)

    coords = []
    for h in range(0, H - p + 1, stride):
        for w in range(0, W - p + 1, stride):
            for d in range(0, D - p + 1, stride):
                coords.append((h, w, d))

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            patches = [volume[:, h:h+p, w:w+p, d:d+p] for h, w, d in batch_coords]
            batch = torch.from_numpy(np.stack(patches)).float().to(device)
            if device.type == 'cuda':
                with autocast('cuda'):
                    preds = torch.sigmoid(model(batch)).cpu().numpy()
            else:
                preds = torch.sigmoid(model(batch)).cpu().numpy()
            for j, (h, w, d) in enumerate(batch_coords):
                output[h:h+p, w:w+p, d:d+p] += preds[j, 0]
                count[h:h+p, w:w+p, d:d+p] += 1

    output = output / np.maximum(count, 1)
    return output[:orig_H, :orig_W, :orig_D]


# =============================================================================
# FEATURE BUILDING
# =============================================================================

def build_stacking_features(cache_file, model_names=None):
    """Build full-volume stacking features from cached predictions."""
    if model_names is None:
        model_names = STACKING_MODEL_NAMES
    data = np.load(cache_file)
    mask = data['mask']
    preds = []
    for name in model_names:
        preds.append(data[name])
    preds = np.stack(preds, axis=0)
    variance = preds.var(axis=0, keepdims=True)
    range_map = preds.max(axis=0, keepdims=True) - preds.min(axis=0, keepdims=True)
    features = np.concatenate([preds, variance, range_map], axis=0)
    return features, preds, mask


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_stacking_model(model_dir=None, device=None):
    """Load the stacking classifier from checkpoint."""
    if model_dir is None:
        model_dir = Path(__file__).resolve().parent.parent.parent / 'model'
    else:
        model_dir = Path(model_dir)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = model_dir / 'stacking_classifier.pth'
    if not checkpoint_path.exists():
        return None
    model = StackingClassifier(in_channels=STACKING_IN_CHANNELS).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# =============================================================================
# INFERENCE PIPELINE
# =============================================================================

def run_stacking_inference(cache_path, model, device, target_size=None,
                           model_names=None, patch_size=None, overlap=None):
    """Full stacking inference pipeline."""
    if model_names is None:
        model_names = STACKING_MODEL_NAMES
    if patch_size is None:
        patch_size = STACKING_PATCH_SIZE
    if overlap is None:
        overlap = STACKING_OVERLAP

    features, preds, mask = build_stacking_features(cache_path, model_names)
    stacking_prob = sliding_window_inference(model, features, patch_size, device, overlap=overlap)

    agreement = np.zeros_like(stacking_prob)
    for i in range(preds.shape[0]):
        agreement += (preds[i] > 0.5).astype(np.float32)

    individual = {name: preds[i] for i, name in enumerate(model_names)}

    if target_size is not None and tuple(target_size) != tuple(stacking_prob.shape):
        factors = [t / s for t, s in zip(target_size, stacking_prob.shape)]
        stacking_prob = zoom(stacking_prob.astype(np.float32), factors, order=1)
        agreement = zoom(agreement.astype(np.float32), factors, order=0)
        for name in individual:
            individual[name] = zoom(individual[name].astype(np.float32), factors, order=1)

    return {
        'fused': stacking_prob.astype(np.float32),
        'individual': {k: v.astype(np.float32) for k, v in individual.items()},
        'agreement': agreement.astype(np.float32),
    }
