"""
Train a single base model for lung CT segmentation with:
  1. SmallLesionOptimizedLoss
  2. MONAI augmentation pipeline
  3. Hybrid weighted sampling (oversamples small lesions + difficult cases)

Same LightweightUNet3D architecture as brain met base models, adapted for single-channel CT.

Usage:
    python scripts/training/train_base_model.py --patch-size 8 --gpu 0
    python scripts/training/train_base_model.py --patch-size 12 --gpu 0
    python scripts/training/train_base_model.py --patch-size 24 --gpu 1
    python scripts/training/train_base_model.py --patch-size 36 --gpu 1
"""

import sys
import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation.unet import LightweightUNet3D
from segmentation.dataset import LungCTDataset
from segmentation.advanced_losses import SmallLesionOptimizedLoss
from segmentation.weighted_sampling import get_case_weights


# ============================================================================
# CONFIG
# ============================================================================

MODEL_CONFIG = {
    "in_channels": 1,
    "out_channels": 1,
    "base_channels": 20,
    "use_attention": True,
    "use_residual": True,
}

DEFAULT_EPOCHS = 250
DEFAULT_LR = 0.001
DEFAULT_PATCHES_PER_VOL = 5
VAL_SPLIT = 0.15
SEED = 42


# ============================================================================
# METRICS
# ============================================================================

def compute_size_stratified_dice(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    SIZE_BINS = {
        "tiny": (0, 500), "small": (500, 2000),
        "medium": (2000, 5000), "large": (5000, float("inf")),
    }
    results = {}
    for name, (lo, hi) in SIZE_BINS.items():
        scores = []
        for i in range(target.shape[0]):
            vol = target[i].sum().item()
            if lo <= vol < hi:
                inter = (pred_bin[i] * target[i]).sum()
                dice = (2 * inter + 1e-6) / (pred_bin[i].sum() + target[i].sum() + 1e-6)
                scores.append(dice.item())
        results[name] = sum(scores) / len(scores) if scores else None
    inter = (pred_bin * target).sum()
    results["overall"] = ((2 * inter + 1e-6) / (pred_bin.sum() + target.sum() + 1e-6)).item()
    return results


def compute_sensitivity(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    tp = ((pred_bin == 1) & (target == 1)).sum().float()
    fn = ((pred_bin == 0) & (target == 1)).sum().float()
    return (tp / (tp + fn + 1e-6)).item()


# ============================================================================
# DATA LOADING
# ============================================================================

def get_batch_size(patch_size, gpu_memory_gb):
    scale = (patch_size / 16) ** 3
    mem_per_sample = 0.05 * scale
    bs = int(gpu_memory_gb * 0.6 / mem_per_sample)
    return max(2, min(bs, 64))


def load_patches(dataset, patches_per_volume=5, chunk_size=30):
    """Pre-extract patches. MONAI augmentation applied during dataset.__getitem__."""
    n = len(dataset)
    total = n * patches_per_volume
    all_imgs, all_masks = [], []

    print(f"  Extracting {patches_per_volume} patches x {n} volumes = {total} patches...")
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_imgs, chunk_masks = [], []
        for i in range(start, end):
            for _ in range(patches_per_volume):
                img, mask, _ = dataset[i]
                chunk_imgs.append(img)
                chunk_masks.append(mask)
        all_imgs.append(torch.stack(chunk_imgs))
        all_masks.append(torch.stack(chunk_masks))
        print(f"    {end}/{n} volumes...", end="\r")
        del chunk_imgs, chunk_masks
        gc.collect()

    print(f"  Loaded {total} patches from {n} volumes.           ")
    return torch.cat(all_imgs), torch.cat(all_masks)


class PatchTensorDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{epochs} [Train]", leave=False, ncols=100)
    for img, mask in pbar:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device, epoch, epochs):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{epochs} [Val]  ", leave=False, ncols=100)
    with torch.no_grad():
        for img, mask in pbar:
            img, mask = img.to(device), mask.to(device)
            out = model(img)
            loss = criterion(out, mask)
            total_loss += loss.item()
            all_preds.append(out.cpu())
            all_targets.append(mask.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    dice_scores = compute_size_stratified_dice(all_preds, all_targets)
    sensitivity = compute_sensitivity(all_preds, all_targets)
    return total_loss / len(loader), dice_scores, sensitivity


def train_model(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    gpu_mem = 8.0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(args.gpu).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(args.gpu)
    else:
        gpu_name = "CPU"

    patch_size = args.patch_size
    PATCH = (patch_size, patch_size, patch_size)
    batch_size = args.batch_size or get_batch_size(patch_size, gpu_mem)
    epochs = args.epochs

    model_dir = ROOT / "model" / "base_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# LUNG CT BASE MODEL TRAINING (SmallLesionLoss + MONAI aug + weighted sampling)")
    print(f"# Patch: {patch_size}^3 | Epochs: {epochs} | Batch: {batch_size}")
    print(f"# GPU {args.gpu}: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"# Data: {args.data_dir}")
    print(f"{'#'*70}\n")

    # Build dataset
    full_ds = LungCTDataset(
        data_dir=str(args.data_dir),
        patch_size=None,
        target_size=None,
        augment=False,
    )

    n_cases = len(full_ds.cases)
    np.random.seed(SEED)
    idx = np.random.permutation(n_cases)
    val_size = int(n_cases * VAL_SPLIT)
    train_cases = [full_ds.cases[i] for i in idx[val_size:]]
    val_cases = [full_ds.cases[i] for i in idx[:val_size]]

    print(f"  Cases: {n_cases} total, {len(train_cases)} train, {len(val_cases)} val")

    # Training dataset with MONAI augmentation
    train_ds = LungCTDataset(
        data_dir=str(args.data_dir),
        patch_size=PATCH,
        target_size=None,
        augment=True,
        augmentation_prob=0.3,
    )
    train_ds.cases = train_cases

    # Validation dataset without augmentation
    val_ds = LungCTDataset(
        data_dir=str(args.data_dir),
        patch_size=PATCH,
        target_size=None,
        augment=False,
    )
    val_ds.cases = val_cases

    # Pre-extract patches
    t0 = time.time()
    train_images, train_masks = load_patches(train_ds, patches_per_volume=args.patches_per_volume)
    val_images, val_masks = load_patches(val_ds, patches_per_volume=max(2, args.patches_per_volume // 2))
    mb = (train_images.element_size() * train_images.numel() +
          val_images.element_size() * val_images.numel()) / 1e6
    print(f"  Pre-loaded in {(time.time()-t0)/60:.1f} min ({mb:.0f} MB)")

    # Build weighted sampler
    print("  Computing sampling weights (hybrid: volume-inverse + difficulty)...")
    weights = get_case_weights(train_ds, strategy='hybrid', difficulty_multiplier=10.0)
    patch_weights = []
    for w in weights:
        patch_weights.extend([w] * args.patches_per_volume)
    sampler = WeightedRandomSampler(
        weights=patch_weights, num_samples=len(patch_weights), replacement=True,
    )

    train_tensor_ds = PatchTensorDataset(train_images, train_masks)
    val_tensor_ds = PatchTensorDataset(val_images, val_masks)

    train_loader = DataLoader(
        train_tensor_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_tensor_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print(f"  Train: {len(train_tensor_ds)}, Val: {len(val_tensor_ds)}")

    # Model
    model = LightweightUNet3D(**MODEL_CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = SmallLesionOptimizedLoss()

    prefix = f"base_{patch_size}patch"
    best_path = model_dir / f"{prefix}_best.pth"
    state_path = model_dir / f"{prefix}_state.json"

    best_val_loss = float("inf")
    best_val_dice = 0.0
    best_tiny_dice = 0.0
    best_epoch = 0
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        val_loss, dice_scores, sensitivity = validate_epoch(
            model, val_loader, criterion, device, epoch, epochs
        )
        scheduler.step()

        val_dice = dice_scores["overall"]
        tiny_d = dice_scores.get("tiny")
        if tiny_d and tiny_d > best_tiny_dice:
            best_tiny_dice = tiny_d

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_dice = val_dice
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "dice": val_dice,
                "dice_scores": dice_scores,
                "sensitivity": sensitivity,
                "model_config": MODEL_CONFIG,
                "patch_size": patch_size,
                "training": "SmallLesionLoss + MONAI aug + weighted sampling",
            }, best_path)

        with open(state_path, "w") as f:
            json.dump({
                "epoch": epoch, "epochs_total": epochs,
                "patch_size": patch_size, "gpu": args.gpu,
                "train_loss": train_loss, "val_loss": val_loss,
                "val_dice": val_dice, "tiny_dice": tiny_d,
                "sensitivity": sensitivity,
                "best_val_loss": best_val_loss,
                "best_val_dice": best_val_dice,
                "best_tiny_dice": best_tiny_dice,
                "best_epoch": best_epoch,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

        ep_time = time.time() - ep_start
        tiny_str = f"tiny={tiny_d:.3f}" if tiny_d else "tiny=N/A"
        best_mark = " *" if is_best else ""
        print(f"Epoch {epoch:3d}/{epochs}: loss={train_loss:.4f} val_dice={val_dice:.3f} "
              f"{tiny_str} sens={sensitivity:.3f} ({ep_time:.1f}s){best_mark}")

    total_min = (time.time() - train_start) / 60
    print(f"\n{'='*70}")
    print(f"  {patch_size}^3 COMPLETE in {total_min:.1f} min")
    print(f"  Best Dice: {best_val_dice:.4f} (epoch {best_epoch})")
    print(f"  Best Tiny Dice: {best_tiny_dice:.4f}")
    print(f"  Checkpoint: {best_path}")
    print(f"{'='*70}\n")

    del model, optimizer, scheduler, train_loader, val_loader
    del train_images, train_masks, val_images, val_masks
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_dice


def main():
    parser = argparse.ArgumentParser(description="Lung CT base model: SmallLesionLoss + MONAI aug + weighted sampling")
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patches-per-volume", type=int, default=DEFAULT_PATCHES_PER_VOL)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = ROOT / "data" / "preprocessed" / "train"
    else:
        args.data_dir = Path(args.data_dir)

    if not args.data_dir.exists():
        print(f"ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)

    train_model(args)


if __name__ == "__main__":
    main()
