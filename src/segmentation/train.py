"""
Training script for lung CT segmentation U-Net
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json

from dataset import LungCTDataset, get_train_val_split
from unet import LightweightUNet3D, CombinedLoss, EnhancedCombinedLoss, count_parameters


def dice_coefficient(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-8)
    return dice.item()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, scheduler=None, gradient_clip=None):
    model.train()
    total_loss = 0
    total_dice = 0
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, masks, _) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        if gradient_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        dice = dice_coefficient(outputs, masks)
        total_loss += loss.item()
        total_dice += dice
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    return total_loss / len(loader), total_dice / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    pbar = tqdm(loader, desc="Validation")
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        dice = dice_coefficient(outputs, masks)
        total_loss += loss.item()
        total_dice += dice
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    return total_loss / len(loader), total_dice / len(loader)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading dataset...")
    train_cases, val_cases = get_train_val_split(args.data_dir, val_ratio=args.val_ratio, seed=args.seed)

    train_dataset = LungCTDataset(
        data_dir=args.data_dir, patch_size=tuple(args.patch_size),
        augment=args.use_augmentation, augmentation_prob=args.augmentation_prob
    )
    val_dataset = LungCTDataset(
        data_dir=args.data_dir, patch_size=tuple(args.patch_size), augment=False
    )

    case_to_idx = {case.name: idx for idx, case in enumerate(train_dataset.cases)}
    train_indices = [case_to_idx[case.name] for case in train_cases if case.name in case_to_idx]
    val_indices = [case_to_idx[case.name] for case in val_cases if case.name in case_to_idx]

    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = LightweightUNet3D(
        in_channels=1, out_channels=1, base_channels=args.base_channels,
        depth=args.depth, dropout_p=args.dropout,
        use_attention=args.use_attention, use_residual=args.use_residual
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = EnhancedCombinedLoss() if args.loss_type == 'enhanced_combined' else CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_dice = 0
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [], 'lr': []}

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, gradient_clip=args.gradient_clip
        )
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}\n")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice, 'best_dice': best_dice,
                'history': history, 'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"[SAVED] Best model (Dice: {val_dice:.4f})\n")

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training completed! Best validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lung CT segmentation model")
    parser.add_argument('--data_dir', type=str, default='../../data/preprocessed/train')
    parser.add_argument('--output_dir', type=str, default='../../model')
    parser.add_argument('--base_channels', type=int, default=16)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--use_residual', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--patch_size', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--use_augmentation', action='store_true')
    parser.add_argument('--augmentation_prob', type=float, default=0.3)
    parser.add_argument('--loss_type', type=str, default='combined', choices=['combined', 'enhanced_combined'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
