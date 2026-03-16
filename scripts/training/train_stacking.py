"""
Stacking classifier training for lung CT segmentation.

Trains a lightweight 3D CNN meta-learner on base model predictions.

Usage:
    python scripts/training/train_stacking.py
    python scripts/training/train_stacking.py --models base_24patch,base_36patch
    python scripts/training/train_stacking.py --force-predict --tta

Steps:
    1. Loads selected base models
    2. Generates predictions (sliding window with optional TTA)
    3. Builds stacking features (N predictions + variance + range)
    4. Trains lightweight CNN meta-learner
    5. Evaluates with post-processing
    6. Failure case analysis

Resumable - caches base model predictions to avoid regenerating.
"""

import gc
import json
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, label as ndimage_label
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation.unet import LightweightUNet3D

# =============================================================================
# BASE MODEL CONFIGS
# =============================================================================
BASE_MODELS = {
    'base_8patch': {'patch_size': 8, 'threshold': 0.3},
    'base_12patch': {'patch_size': 12, 'threshold': 0.25},
    'base_24patch': {'patch_size': 24, 'threshold': 0.5},
    'base_36patch': {'patch_size': 36, 'threshold': 0.5},
}

TARGET_SIZE = (128, 128, 128)

# =============================================================================
# STACKING CLASSIFIER
# =============================================================================
class StackingClassifier(nn.Module):
    """3D CNN meta-learner with residual connections."""
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
# LOSSES
# =============================================================================
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

    def focal_loss(self, pred, target, alpha=0.75, gamma=2.0):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (alpha * (1 - pt) ** gamma * bce).mean()

    def forward(self, pred, target):
        return 0.7 * self.tversky(pred, target) + 0.3 * self.focal_loss(pred, target)


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(pred_binary, target):
    pred = pred_binary.flatten()
    target = target.flatten()
    tp = ((pred == 1) & (target == 1)).sum().float()
    tn = ((pred == 0) & (target == 0)).sum().float()
    fp = ((pred == 1) & (target == 0)).sum().float()
    fn = ((pred == 0) & (target == 1)).sum().float()
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    sensitivity = (tp + 1e-6) / (tp + fn + 1e-6)
    specificity = (tn + 1e-6) / (tn + fp + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    return {
        'dice': dice.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
        'precision': precision.item(),
    }


# =============================================================================
# POST-PROCESSING
# =============================================================================
def postprocess_prediction(binary_mask, min_size=20):
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
# VOLUME LOADING (single-channel CT with HU windowing)
# =============================================================================
def hu_normalize(img, hu_min=-1000.0, hu_max=400.0):
    """HU windowing normalization for CT data."""
    img = np.clip(img, hu_min, hu_max)
    img = (img - hu_min) / (hu_max - hu_min)
    return img


def load_volume(case_dir):
    """Load and preprocess a single-channel CT volume."""
    ct_path = case_dir / "ct.nii.gz"
    if not ct_path.exists():
        raise FileNotFoundError(f"Missing ct.nii.gz for {case_dir.name}")
    data = nib.load(str(ct_path)).get_fdata().astype(np.float32)
    factors = [t / s for t, s in zip(TARGET_SIZE, data.shape)]
    data = zoom(data, factors, order=1)
    data = hu_normalize(data)
    return data[np.newaxis]  # (1, H, W, D)


def load_mask(case_dir):
    """Load ground truth mask."""
    for mask_name in ['seg.nii.gz', 'mask.nii.gz', 'label.nii.gz']:
        path = case_dir / mask_name
        if path.exists():
            data = nib.load(str(path)).get_fdata().astype(np.float32)
            factors = [t / s for t, s in zip(TARGET_SIZE, data.shape)]
            data = zoom(data, factors, order=0)
            return (data > 0.5).astype(np.float32)
    return None


# =============================================================================
# SLIDING WINDOW INFERENCE
# =============================================================================
def sliding_window_inference(model, volume, patch_size, device, overlap=0.25):
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
            patches = []
            for h, w, d in batch_coords:
                patches.append(volume[:, h:h+p, w:w+p, d:d+p])
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


def tta_sliding_window_inference(model, volume, patch_size, device, overlap=0.25):
    flip_axes_combos = [
        [], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3],
    ]
    all_preds = []
    for flip_axes in flip_axes_combos:
        vol = volume.copy()
        for ax in flip_axes:
            vol = np.flip(vol, axis=ax).copy()
        pred = sliding_window_inference(model, vol, patch_size, device, overlap=overlap)
        for ax in flip_axes:
            pred = np.flip(pred, axis=ax - 1).copy()
        all_preds.append(pred)
    return np.mean(all_preds, axis=0)


# =============================================================================
# GENERATE BASE MODEL PREDICTIONS
# =============================================================================
def generate_predictions(data_dir, cache_dir, device, selected_models=None, overlap=0.5, tta=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir)
    all_cases = sorted([
        d for d in data_path.iterdir()
        if d.is_dir()
    ])

    model_dir = ROOT / 'model' / 'base_models'

    models_config = BASE_MODELS
    if selected_models:
        models_config = {k: v for k, v in BASE_MODELS.items() if k in selected_models}

    models = {}
    for name, config in models_config.items():
        model_path = model_dir / f'{name}_best.pth'
        if not model_path.exists():
            print(f"WARNING: {model_path} not found, skipping {name}")
            continue
        model = LightweightUNet3D(
            in_channels=1, out_channels=1,
            base_channels=20, use_attention=True, use_residual=True
        ).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models[name] = (model, config['patch_size'])
        print(f"  Loaded {name} (Dice={checkpoint.get('dice', 'N/A'):.4f})")

    infer_fn_name = "TTA sliding window (8 flips)" if tta else "sliding window"
    print(f"\nGenerating predictions for {len(all_cases)} cases with {len(models)} models "
          f"(overlap={overlap}, method={infer_fn_name})...")

    for case_dir in tqdm(all_cases, desc="Predicting"):
        case_id = case_dir.name
        cache_file = cache_dir / f'{case_id}.npz'
        if cache_file.exists():
            continue

        try:
            volume = load_volume(case_dir)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            continue
        mask = load_mask(case_dir)
        if mask is None:
            continue

        preds = {}
        for name, (model, patch_size) in models.items():
            if tta:
                prob_map = tta_sliding_window_inference(model, volume, patch_size, device, overlap=overlap)
            else:
                prob_map = sliding_window_inference(model, volume, patch_size, device, overlap=overlap)
            preds[name] = prob_map

        np.savez_compressed(cache_file, mask=mask, **preds)
        gc.collect()
        torch.cuda.empty_cache()

    del models
    gc.collect()
    torch.cuda.empty_cache()
    return all_cases


# =============================================================================
# STACKING DATASET
# =============================================================================
class StackingDataset(Dataset):
    def __init__(self, case_ids, cache_dir, model_names, patch_size=32, fg_ratio=0.7):
        self.case_ids = case_ids
        self.cache_dir = Path(cache_dir)
        self.model_names = model_names
        self.patch_size = patch_size
        self.fg_ratio = fg_ratio

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        cache_file = self.cache_dir / f'{case_id}.npz'
        data = np.load(cache_file)
        mask = data['mask']

        preds = []
        for name in self.model_names:
            preds.append(data[name])
        preds = np.stack(preds, axis=0)

        variance = preds.var(axis=0, keepdims=True)
        range_map = preds.max(axis=0, keepdims=True) - preds.min(axis=0, keepdims=True)
        features = np.concatenate([preds, variance, range_map], axis=0)

        p = self.patch_size
        C, H, W, D = features.shape
        foreground = np.where(mask > 0)
        if len(foreground[0]) > 0 and np.random.rand() < self.fg_ratio:
            idx_fg = np.random.randint(len(foreground[0]))
            ch, cw, cd = foreground[0][idx_fg], foreground[1][idx_fg], foreground[2][idx_fg]
        else:
            ch = np.random.randint(p // 2, max(H - p // 2, p // 2 + 1))
            cw = np.random.randint(p // 2, max(W - p // 2, p // 2 + 1))
            cd = np.random.randint(p // 2, max(D - p // 2, p // 2 + 1))

        h_start = max(0, min(ch - p // 2, H - p))
        w_start = max(0, min(cw - p // 2, W - p))
        d_start = max(0, min(cd - p // 2, D - p))

        feat_patch = features[:, h_start:h_start+p, w_start:w_start+p, d_start:d_start+p]
        mask_patch = mask[h_start:h_start+p, w_start:w_start+p, d_start:d_start+p]
        mask_patch = mask_patch[np.newaxis]

        return (
            torch.from_numpy(feat_patch).float(),
            torch.from_numpy(mask_patch).float(),
        )


# =============================================================================
# FULL-VOLUME STACKING FEATURES
# =============================================================================
def build_stacking_features(cache_file, model_names):
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
# EVALUATION
# =============================================================================
def get_prob_maps(case_id, cache_dir, model_names, stacking_model, device,
                  stacking_patch_size=32, stacking_overlap=0.5):
    cache_file = Path(cache_dir) / f'{case_id}.npz'
    if not cache_file.exists():
        return None, None

    features, preds, mask = build_stacking_features(cache_file, model_names)

    prob_maps = {}
    for i, name in enumerate(model_names):
        prob_maps[name] = preds[i]
    prob_maps['simple_average'] = preds.mean(axis=0)
    prob_maps['max_fusion'] = preds.max(axis=0)
    prob_maps['stacking'] = sliding_window_inference(
        stacking_model, features, stacking_patch_size, device, overlap=stacking_overlap
    )
    return prob_maps, mask


def tune_thresholds(tune_case_ids, cache_dir, model_names, stacking_model, device,
                    min_component_size=0, stacking_patch_size=32, stacking_overlap=0.5):
    thresholds_to_try = np.arange(0.1, 0.95, 0.05)
    stacking_model.eval()

    all_prob_maps = {}
    method_names = None

    for case_id in tqdm(tune_case_ids, desc="Threshold tuning"):
        prob_maps, mask = get_prob_maps(
            case_id, cache_dir, model_names, stacking_model, device,
            stacking_patch_size=stacking_patch_size, stacking_overlap=stacking_overlap
        )
        if prob_maps is None:
            continue
        if method_names is None:
            method_names = list(prob_maps.keys())
            all_prob_maps = {m: [] for m in method_names}
        for m in method_names:
            all_prob_maps[m].append((prob_maps[m], mask))

    best_thresholds = {}
    print(f"\n{'Method':<25} {'Best Thresh':<14} {'Dice @ Best':<12}")
    print("-" * 55)

    for method_name in method_names:
        best_dice = 0
        best_t = 0.5
        for t in thresholds_to_try:
            dices = []
            for prob_map, mask in all_prob_maps[method_name]:
                pred_binary = (prob_map > t).astype(np.float32)
                if min_component_size > 0:
                    pred_binary = postprocess_prediction(pred_binary, min_size=min_component_size)
                mask_t = torch.from_numpy(mask).float()
                pred_t = torch.from_numpy(pred_binary).float()
                m = compute_metrics(pred_t, mask_t)
                dices.append(m['dice'])
            avg_dice = np.mean(dices)
            if avg_dice > best_dice:
                best_dice = avg_dice
                best_t = t
        best_thresholds[method_name] = float(best_t)
        print(f"{method_name:<25} {best_t:<14.2f} {best_dice:<12.4f}")

    return best_thresholds


def evaluate_all_methods(val_case_ids, cache_dir, model_names, stacking_model, device,
                         thresholds, min_component_size=0,
                         stacking_patch_size=32, stacking_overlap=0.5):
    stacking_model.eval()
    all_metrics = {m: [] for m in thresholds}

    for case_id in tqdm(val_case_ids, desc="Evaluating"):
        prob_maps, mask = get_prob_maps(
            case_id, cache_dir, model_names, stacking_model, device,
            stacking_patch_size=stacking_patch_size, stacking_overlap=stacking_overlap
        )
        if prob_maps is None:
            continue
        mask_t = torch.from_numpy(mask).float()
        for method_name, threshold in thresholds.items():
            pred_binary = (prob_maps[method_name] > threshold).astype(np.float32)
            if min_component_size > 0:
                pred_binary = postprocess_prediction(pred_binary, min_size=min_component_size)
            pred_t = torch.from_numpy(pred_binary).float()
            metrics = compute_metrics(pred_t, mask_t)
            all_metrics[method_name].append((case_id, metrics))

    results = {}
    for method_name, case_metrics in all_metrics.items():
        if not case_metrics:
            continue
        results[method_name] = {
            'dice': np.mean([m['dice'] for _, m in case_metrics]),
            'sensitivity': np.mean([m['sensitivity'] for _, m in case_metrics]),
            'specificity': np.mean([m['specificity'] for _, m in case_metrics]),
            'precision': np.mean([m['precision'] for _, m in case_metrics]),
            'std_dice': np.std([m['dice'] for _, m in case_metrics]),
            'n_cases': len(case_metrics),
            'threshold': thresholds[method_name],
        }
    return results, all_metrics


def analyze_failures(all_metrics, best_method_name, results_path):
    case_metrics = all_metrics.get(best_method_name, [])
    if not case_metrics:
        print("No cases to analyze.")
        return

    sorted_cases = sorted(case_metrics, key=lambda x: x[1]['dice'])
    dices = [m['dice'] for _, m in case_metrics]

    print(f"\n{'='*60}")
    print(f"FAILURE ANALYSIS: {best_method_name}")
    print(f"{'='*60}")

    print(f"\nWorst 10 cases by Dice:")
    print(f"{'Case ID':<30} {'Dice':<10} {'Sens':<10} {'Prec':<10}")
    print("-" * 60)
    for case_id, metrics in sorted_cases[:10]:
        print(f"{case_id:<30} {metrics['dice']:.4f}     {metrics['sensitivity']:.4f}     "
              f"{metrics['precision']:.4f}")

    dices_arr = np.array(dices)
    print(f"\nDice distribution ({len(dices)} cases):")
    print(f"  Min:    {np.min(dices_arr):.4f}")
    print(f"  25th:   {np.percentile(dices_arr, 25):.4f}")
    print(f"  Median: {np.median(dices_arr):.4f}")
    print(f"  75th:   {np.percentile(dices_arr, 75):.4f}")
    print(f"  Max:    {np.max(dices_arr):.4f}")
    print(f"  Mean:   {np.mean(dices_arr):.4f} +/- {np.std(dices_arr):.4f}")

    per_case = {}
    for case_id, metrics in case_metrics:
        per_case[case_id] = {k: float(v) for k, v in metrics.items()}
    per_case_path = results_path.parent / 'stacking_results_per_case.json'
    with open(per_case_path, 'w') as f:
        json.dump(per_case, f, indent=2)
    print(f"\nPer-case metrics saved to: {per_case_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train stacking classifier for lung CT segmentation")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-dir', type=str, default='data/preprocessed/train')
    parser.add_argument('--force-predict', action='store_true')
    parser.add_argument('--models', type=str, default='base_8patch,base_12patch,base_24patch,base_36patch')
    parser.add_argument('--min-component', type=int, default=20)
    parser.add_argument('--regen-overlap', type=float, default=0.5)
    parser.add_argument('--cache-dir', type=str, default='stacking_cache')
    parser.add_argument('--fg-ratio', type=float, default=0.7)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--stacking-patch', type=int, default=32)
    parser.add_argument('--stacking-overlap', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = ROOT / 'model'
    cache_dir = ROOT / 'model' / args.cache_dir
    state_dir = model_dir / 'training_states'
    state_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip() for m in args.models.split(',')]
    in_channels = len(model_names) + 2

    print("=" * 60)
    print("STACKING CLASSIFIER TRAINING (Lung CT)")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Models: {model_names}")
    print(f"Input channels: {in_channels} ({len(model_names)} preds + variance + range)")
    print(f"Cache dir: {cache_dir}")
    print(f"TTA: {args.tta}")
    print(f"FG ratio: {args.fg_ratio}")

    data_dir = str(ROOT / args.data_dir)

    # STEP 1: Generate predictions
    print("\n" + "=" * 60)
    print("STEP 1: Generate base model predictions")
    print("=" * 60)

    if args.force_predict:
        import shutil
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    all_cases = generate_predictions(
        data_dir, cache_dir, device,
        selected_models=model_names, overlap=args.regen_overlap, tta=args.tta
    )

    cached_cases = []
    for case_dir in all_cases:
        cache_file = cache_dir / f'{case_dir.name}.npz'
        if cache_file.exists():
            cached_cases.append(case_dir.name)

    print(f"\nCached predictions: {len(cached_cases)} cases")

    # STEP 2: Train/val split
    print("\n" + "=" * 60)
    print("STEP 2: Train/val split")
    print("=" * 60)

    random.seed(42)
    cases_shuffled = cached_cases.copy()
    random.shuffle(cases_shuffled)

    n_val = int(len(cases_shuffled) * 0.15)
    val_cases = cases_shuffled[:n_val]
    train_cases = cases_shuffled[n_val:]

    n_stack_val = int(len(train_cases) * 0.15)
    stack_val_cases = train_cases[:n_stack_val]
    stack_train_cases = train_cases[n_stack_val:]

    print(f"Base model val set: {n_val} cases (held out)")
    print(f"Stacking train: {len(stack_train_cases)} cases")
    print(f"Stacking val: {len(stack_val_cases)} cases")

    # STEP 3: Train stacking classifier
    print("\n" + "=" * 60)
    print("STEP 3: Train stacking classifier")
    print("=" * 60)

    train_ds = StackingDataset(stack_train_cases, cache_dir, model_names,
                               patch_size=args.patch_size, fg_ratio=args.fg_ratio)
    val_ds = StackingDataset(stack_val_cases, cache_dir, model_names,
                             patch_size=args.patch_size, fg_ratio=args.fg_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    stacking_model = StackingClassifier(in_channels=in_channels).to(device)
    n_params = sum(p.numel() for p in stacking_model.parameters())
    print(f"Stacking model: {n_params:,} parameters ({in_channels} input channels)")

    criterion = CombinedLoss()
    optimizer = optim.AdamW(stacking_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    stacking_state_path = state_dir / 'stacking_finetune_state.pth'
    stacking_best_path = model_dir / 'stacking_classifier.pth'
    start_epoch = 1
    best_dice = 0
    history = {'train_loss': [], 'val_dice': [], 'val_sens': [], 'val_spec': []}

    current_config = {
        'in_channels': in_channels,
        'model_names': model_names,
        'fg_ratio': args.fg_ratio,
        'patch_size': args.patch_size,
        'tta': args.tta,
    }

    if stacking_state_path.exists():
        state = torch.load(stacking_state_path, map_location=device, weights_only=False)
        saved_config = {
            'in_channels': state.get('in_channels', 6),
            'model_names': state.get('model_names', []),
            'fg_ratio': state.get('fg_ratio', 0.9),
            'patch_size': state.get('patch_size', 32),
            'tta': state.get('tta', False),
        }
        if saved_config == current_config:
            print("Resuming from checkpoint...")
            stacking_model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])
            scaler.load_state_dict(state['scaler_state_dict'])
            start_epoch = state['epoch'] + 1
            best_dice = state['best_dice']
            history = state.get('history', history)
            print(f"  Resumed at epoch {start_epoch}, best Dice={best_dice:.4f}")
        else:
            print("Config mismatch, starting fresh training...")
            stacking_state_path.unlink()
            if stacking_best_path.exists():
                stacking_best_path.unlink()

    for epoch in range(start_epoch, args.epochs + 1):
        stacking_model.train()
        train_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"[stacking] Epoch {epoch}/{args.epochs}")
        for features, masks in pbar:
            features, masks = features.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = stacking_model(features)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(stacking_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        stacking_model.eval()
        val_dice_sum = 0
        val_sens_sum = 0
        val_spec_sum = 0
        n_val_batches = 0

        with torch.no_grad():
            for features, masks in val_loader:
                features, masks = features.to(device), masks.to(device)
                with autocast('cuda'):
                    outputs = stacking_model(features)
                pred_binary = (torch.sigmoid(outputs) > 0.5).float()
                m = compute_metrics(pred_binary, masks)
                val_dice_sum += m['dice']
                val_sens_sum += m['sensitivity']
                val_spec_sum += m['specificity']
                n_val_batches += 1

        scheduler.step()

        avg_loss = train_loss / max(n_batches, 1)
        avg_dice = val_dice_sum / max(n_val_batches, 1)
        avg_sens = val_sens_sum / max(n_val_batches, 1)
        avg_spec = val_spec_sum / max(n_val_batches, 1)

        history['train_loss'].append(avg_loss)
        history['val_dice'].append(avg_dice)
        history['val_sens'].append(avg_sens)
        history['val_spec'].append(avg_spec)

        improved = avg_dice > best_dice
        marker = " *NEW BEST*" if improved else ""

        print(f"  Loss={avg_loss:.4f}  Dice={avg_dice:.4f}  "
              f"Sens={avg_sens:.4f}  Spec={avg_spec:.4f}{marker}")

        if improved:
            best_dice = avg_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': stacking_model.state_dict(),
                'dice': best_dice,
                'n_params': n_params,
                'in_channels': in_channels,
                'model_names': model_names,
            }, stacking_best_path)

        torch.save({
            'epoch': epoch,
            'model_state_dict': stacking_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_dice': best_dice,
            'history': history,
            'in_channels': in_channels,
            'model_names': model_names,
            'fg_ratio': args.fg_ratio,
            'patch_size': args.patch_size,
            'tta': args.tta,
            'timestamp': datetime.now().isoformat(),
        }, stacking_state_path)

    print(f"\nStacking training complete! Best patch Dice: {best_dice:.4f}")

    # STEP 4: Threshold tuning
    print("\n" + "=" * 60)
    print(f"STEP 4: Threshold tuning (min_component={args.min_component})")
    print("=" * 60)

    if stacking_best_path.exists():
        checkpoint = torch.load(stacking_best_path, map_location=device, weights_only=False)
        stacking_model.load_state_dict(checkpoint['model_state_dict'])
    stacking_model.eval()

    best_thresholds = tune_thresholds(
        stack_val_cases, cache_dir, model_names, stacking_model, device,
        min_component_size=args.min_component,
        stacking_patch_size=args.stacking_patch,
        stacking_overlap=args.stacking_overlap
    )

    # STEP 5: Full-volume evaluation
    print("\n" + "=" * 60)
    print("STEP 5: Full-volume evaluation")
    print("=" * 60)

    results, all_metrics = evaluate_all_methods(
        val_cases, cache_dir, model_names, stacking_model, device, best_thresholds,
        min_component_size=args.min_component,
        stacking_patch_size=args.stacking_patch,
        stacking_overlap=args.stacking_overlap
    )

    print(f"\n{'Method':<25} {'Dice':<10} {'Sens':<10} {'Spec':<10} {'Prec':<10} {'Thresh':<8}")
    print("-" * 75)
    for method_name, metrics in sorted(results.items(), key=lambda x: -x[1]['dice']):
        print(f"{method_name:<25} {metrics['dice']:.4f}     {metrics['sensitivity']:.4f}     "
              f"{metrics['specificity']:.4f}     {metrics['precision']:.4f}     "
              f"{metrics['threshold']:.2f}")

    results_path = model_dir / 'stacking_results.json'
    json_results = {}
    for k, v in results.items():
        json_results[k] = {kk: float(vv) for kk, vv in v.items()}
    json_results['_thresholds'] = {k: float(v) for k, v in best_thresholds.items()}
    json_results['_config'] = {
        'models': model_names,
        'in_channels': in_channels,
        'cache_dir': args.cache_dir,
    }
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    best_method = max(
        [(k, v) for k, v in results.items() if not k.startswith('_')],
        key=lambda x: x[1]['dice']
    )
    print(f"\nBest method: {best_method[0]} (Dice={best_method[1]['dice']:.4f})")

    # STEP 6: Failure analysis
    analyze_failures(all_metrics, best_method[0], results_path)


if __name__ == '__main__':
    main()
