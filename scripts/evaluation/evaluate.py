"""
Evaluation script for lung CT segmentation.

Evaluates the stacking ensemble on all validation cases with:
- Voxel-level Dice score
- Lesion-level detection metrics (F1, recall, precision)
- Size-stratified analysis

Usage:
    python scripts/evaluation/evaluate.py
    python scripts/evaluation/evaluate.py --data-dir data/preprocessed/train --threshold 0.9
"""

import argparse
import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label as ndimage_label, zoom
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation.stacking import (
    StackingClassifier, load_stacking_model, build_stacking_features,
    sliding_window_inference, postprocess_prediction,
    STACKING_MODEL_NAMES, STACKING_PATCH_SIZE, STACKING_OVERLAP, STACKING_THRESHOLD,
)


def compute_lesion_metrics(pred_binary, mask):
    """Compute lesion-level detection metrics."""
    gt_labeled, n_gt = ndimage_label(mask > 0)
    pred_labeled, n_pred = ndimage_label(pred_binary > 0)

    gt_matched = set()
    pred_matched = set()
    per_lesion_dices = []

    for g in range(1, n_gt + 1):
        gt_region = (gt_labeled == g)
        for p in range(1, n_pred + 1):
            pred_region = (pred_labeled == p)
            if (gt_region & pred_region).any():
                gt_matched.add(g)
                pred_matched.add(p)
                inter = (gt_region & pred_region).sum()
                total = gt_region.sum() + pred_region.sum()
                per_lesion_dices.append(2 * inter / total)

    tp = len(gt_matched)
    fn = n_gt - len(gt_matched)
    fp = n_pred - len(pred_matched)

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'n_gt': n_gt, 'n_pred': n_pred,
        'per_lesion_dices': per_lesion_dices,
    }


def evaluate_case(cache_file, model, device, threshold, min_size=20):
    """Evaluate a single case."""
    features, preds, mask = build_stacking_features(cache_file)

    prob = sliding_window_inference(
        model, features, STACKING_PATCH_SIZE, device, overlap=STACKING_OVERLAP
    )
    pred = (prob > threshold).astype(np.float32)
    pred = postprocess_prediction(pred, min_size=min_size)

    # Voxel dice
    inter = (pred * mask).sum()
    total = pred.sum() + mask.sum()
    voxel_dice = 2 * inter / total if total > 0 else (1.0 if mask.sum() == 0 else 0.0)

    # Lesion metrics
    lesion_metrics = compute_lesion_metrics(pred, mask)

    # Size-stratified dice
    SIZE_BINS = {
        "tiny": (0, 500), "small": (500, 2000),
        "medium": (2000, 5000), "large": (5000, float("inf")),
    }
    gt_labeled, n_gt = ndimage_label(mask > 0)
    size_dices = {}
    for bin_name, (lo, hi) in SIZE_BINS.items():
        bin_dices = []
        for g in range(1, n_gt + 1):
            gt_region = (gt_labeled == g)
            vol = gt_region.sum()
            if lo <= vol < hi:
                pred_in_region = pred[gt_region].sum()
                gt_vol = gt_region.sum()
                d = 2 * pred_in_region / (pred_in_region + gt_vol) if (pred_in_region + gt_vol) > 0 else 0
                bin_dices.append(d)
        size_dices[bin_name] = np.mean(bin_dices) if bin_dices else None

    return {
        'voxel_dice': float(voxel_dice),
        'lesion_metrics': lesion_metrics,
        'size_dices': size_dices,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate lung CT segmentation")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = ROOT / "model" / "stacking_cache"

    threshold = args.threshold or STACKING_THRESHOLD

    # Load stacking model
    model = load_stacking_model(device=device)
    if model is None:
        print("ERROR: Stacking model not found")
        sys.exit(1)

    # Get validation cases
    cached_cases = sorted([f.stem for f in cache_dir.glob("*.npz")])
    random.seed(42)
    cases_shuffled = cached_cases.copy()
    random.shuffle(cases_shuffled)
    n_val = int(len(cases_shuffled) * 0.15)
    val_cases = cases_shuffled[:n_val]

    print(f"Evaluating {len(val_cases)} validation cases")
    print(f"Threshold: {threshold}, Min component size: {args.min_size}")

    # Evaluate
    all_voxel_dices = []
    all_lesion_dices = []
    total_tp, total_fp, total_fn = 0, 0, 0
    size_results = {k: [] for k in ["tiny", "small", "medium", "large"]}

    for case_id in tqdm(val_cases, desc="Evaluating"):
        cache_file = cache_dir / f"{case_id}.npz"
        if not cache_file.exists():
            continue

        result = evaluate_case(cache_file, model, device, threshold, args.min_size)

        all_voxel_dices.append(result['voxel_dice'])
        lm = result['lesion_metrics']
        total_tp += lm['tp']
        total_fp += lm['fp']
        total_fn += lm['fn']
        all_lesion_dices.extend(lm['per_lesion_dices'])

        for bin_name, dice in result['size_dices'].items():
            if dice is not None:
                size_results[bin_name].append(dice)

    # Aggregate
    voxel_dices = np.array(all_voxel_dices)
    lesion_f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-6)
    lesion_recall = total_tp / (total_tp + total_fn + 1e-6)
    lesion_precision = total_tp / (total_tp + total_fp + 1e-6)

    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"\nVoxel-level Dice:")
    print(f"  Mean:   {voxel_dices.mean():.4f} +/- {voxel_dices.std():.4f}")
    print(f"  Median: {np.median(voxel_dices):.4f}")
    print(f"  Min:    {voxel_dices.min():.4f}")
    print(f"  Max:    {voxel_dices.max():.4f}")

    print(f"\nLesion-level Detection:")
    print(f"  F1:        {lesion_f1:.4f}")
    print(f"  Recall:    {lesion_recall:.4f}")
    print(f"  Precision: {lesion_precision:.4f}")
    print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")

    if all_lesion_dices:
        print(f"\nPer-Lesion Dice:")
        print(f"  Mean: {np.mean(all_lesion_dices):.4f}")

    print(f"\nSize-stratified Dice:")
    for bin_name in ["tiny", "small", "medium", "large"]:
        vals = size_results[bin_name]
        if vals:
            print(f"  {bin_name:>8}: {np.mean(vals):.4f} ({len(vals)} lesions)")
        else:
            print(f"  {bin_name:>8}: N/A")

    print(f"{'='*70}")

    # Save results
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'timestamp': datetime.now().isoformat(),
        'n_cases': len(val_cases),
        'threshold': threshold,
        'min_size': args.min_size,
        'voxel_dice_mean': float(voxel_dices.mean()),
        'voxel_dice_std': float(voxel_dices.std()),
        'voxel_dice_median': float(np.median(voxel_dices)),
        'lesion_f1': float(lesion_f1),
        'lesion_recall': float(lesion_recall),
        'lesion_precision': float(lesion_precision),
        'per_lesion_dice': float(np.mean(all_lesion_dices)) if all_lesion_dices else None,
    }
    report_path = results_dir / "evaluation_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to: {report_path}")


if __name__ == "__main__":
    main()
