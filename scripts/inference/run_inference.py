"""
Inference script for lung CT segmentation.

Runs the full stacking ensemble pipeline on new CT volumes:
  1. Load CT volume
  2. Run 4 base models (sliding window)
  3. Run stacking classifier
  4. Post-process and save result

Usage:
    python scripts/inference/run_inference.py --input path/to/ct.nii.gz --output pred.nii.gz
    python scripts/inference/run_inference.py --input-dir data/test --output-dir results/predictions
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation.unet import LightweightUNet3D
from segmentation.stacking import (
    StackingClassifier, load_stacking_model,
    sliding_window_inference, postprocess_prediction,
    STACKING_MODEL_NAMES, STACKING_PATCH_SIZE, STACKING_OVERLAP, STACKING_THRESHOLD,
)

# Base model configs
BASE_MODELS = {
    'base_8patch': {'patch_size': 8},
    'base_12patch': {'patch_size': 12},
    'base_24patch': {'patch_size': 24},
    'base_36patch': {'patch_size': 36},
}

TARGET_SIZE = (128, 128, 128)


def hu_normalize(img, hu_min=-1000.0, hu_max=400.0):
    """HU windowing normalization for CT data."""
    img = np.clip(img, hu_min, hu_max)
    img = (img - hu_min) / (hu_max - hu_min)
    return img


def load_base_models(device):
    """Load all base models."""
    model_dir = ROOT / 'model' / 'base_models'
    models = {}

    for name, config in BASE_MODELS.items():
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
        dice = checkpoint.get('dice', checkpoint.get('val_dice', 'N/A'))
        print(f"  Loaded {name} (Dice={dice})")

    return models


def predict_single(ct_path, base_models, stacking_model, device,
                   threshold=None, min_size=20):
    """Run full inference pipeline on a single CT volume."""
    if threshold is None:
        threshold = STACKING_THRESHOLD

    # Load and preprocess CT
    nii = nib.load(str(ct_path))
    ct_data = nii.get_fdata().astype(np.float32)
    original_shape = ct_data.shape

    # Resize to model input size
    factors = [t / s for t, s in zip(TARGET_SIZE, ct_data.shape)]
    ct_resized = zoom(ct_data, factors, order=1)
    ct_norm = hu_normalize(ct_resized)
    volume = ct_norm[np.newaxis]  # (1, H, W, D)

    # Generate base model predictions
    preds = {}
    for name, (model, patch_size) in base_models.items():
        prob = sliding_window_inference(model, volume, patch_size, device, overlap=0.5)
        preds[name] = prob

    # Build stacking features
    pred_stack = np.stack([preds[name] for name in STACKING_MODEL_NAMES if name in preds], axis=0)
    variance = pred_stack.var(axis=0, keepdims=True)
    range_map = pred_stack.max(axis=0, keepdims=True) - pred_stack.min(axis=0, keepdims=True)
    features = np.concatenate([pred_stack, variance, range_map], axis=0)

    # Run stacking inference
    stacking_prob = sliding_window_inference(
        stacking_model, features, STACKING_PATCH_SIZE, device, overlap=STACKING_OVERLAP
    )

    # Threshold and post-process
    pred_binary = (stacking_prob > threshold).astype(np.float32)
    pred_binary = postprocess_prediction(pred_binary, min_size=min_size)

    # Upsample back to original size
    if pred_binary.shape != original_shape:
        up_factors = [o / p for o, p in zip(original_shape, pred_binary.shape)]
        pred_binary = zoom(pred_binary, up_factors, order=0)

    return pred_binary, stacking_prob


def main():
    parser = argparse.ArgumentParser(description="Lung CT segmentation inference")
    parser.add_argument("--input", type=str, default=None,
                        help="Single CT NIfTI file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output segmentation NIfTI file")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory with CT volumes (ct.nii.gz per subdirectory)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for predictions")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save-prob", action="store_true",
                        help="Also save probability map")
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir is required")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading models...")
    base_models = load_base_models(device)
    stacking_model = load_stacking_model(device=device)

    if stacking_model is None:
        print("ERROR: Stacking model not found")
        sys.exit(1)
    if not base_models:
        print("ERROR: No base models found")
        sys.exit(1)

    print(f"Loaded {len(base_models)} base models + stacking classifier")

    if args.input:
        # Single file mode
        ct_path = Path(args.input)
        output_path = Path(args.output) if args.output else ct_path.parent / "pred_seg.nii.gz"

        print(f"\nProcessing: {ct_path}")
        t0 = time.time()
        pred, prob = predict_single(ct_path, base_models, stacking_model, device,
                                     args.threshold, args.min_size)

        # Save with same header as input
        nii = nib.load(str(ct_path))
        pred_nii = nib.Nifti1Image(pred.astype(np.uint8), nii.affine, nii.header)
        nib.save(pred_nii, str(output_path))
        print(f"Saved: {output_path} ({time.time()-t0:.1f}s)")

        if args.save_prob:
            prob_path = output_path.parent / output_path.name.replace(".nii.gz", "_prob.nii.gz")
            prob_nii = nib.Nifti1Image(prob.astype(np.float32), nii.affine, nii.header)
            nib.save(prob_nii, str(prob_path))
            print(f"Saved: {prob_path}")

    elif args.input_dir:
        # Batch mode
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all cases
        cases = []
        for case_dir in sorted(input_dir.iterdir()):
            if case_dir.is_dir():
                ct_path = case_dir / "ct.nii.gz"
                if ct_path.exists():
                    cases.append((case_dir.name, ct_path))

        if not cases:
            # Try flat files
            for f in sorted(input_dir.glob("*.nii.gz")):
                cases.append((f.stem, f))

        print(f"\nProcessing {len(cases)} cases...")

        for case_id, ct_path in cases:
            out_path = output_dir / f"{case_id}_pred.nii.gz"
            if out_path.exists():
                continue

            print(f"  {case_id}...", end=" ", flush=True)
            t0 = time.time()

            try:
                pred, prob = predict_single(ct_path, base_models, stacking_model, device,
                                             args.threshold, args.min_size)

                nii = nib.load(str(ct_path))
                pred_nii = nib.Nifti1Image(pred.astype(np.uint8), nii.affine, nii.header)
                nib.save(pred_nii, str(out_path))

                if args.save_prob:
                    prob_path = output_dir / f"{case_id}_prob.nii.gz"
                    prob_nii = nib.Nifti1Image(prob.astype(np.float32), nii.affine, nii.header)
                    nib.save(prob_nii, str(prob_path))

                print(f"done ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"ERROR: {e}")

            gc.collect()
            torch.cuda.empty_cache()

        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
