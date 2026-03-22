"""
Re-run stacking with all 6 base models (4 custom + nnU-Net 3D + nnU-Net 2D).

Steps:
  1. Generate nnU-Net predictions for all tumor-only cases
  2. Merge nnU-Net predictions into existing stacking cache
  3. Train stacking classifier with 8 input channels
  4. Evaluate

Usage:
    python scripts/training/restacking_with_nnunet.py
    python scripts/training/restacking_with_nnunet.py --skip-predict   # if nnU-Net preds already cached
    python scripts/training/restacking_with_nnunet.py --epochs 500
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable

TUMOR_DIR = ROOT / "data" / "preprocessed" / "tumor_only"
MODEL_DIR = ROOT / "model"
CACHE_DIR = MODEL_DIR / "stacking_cache_6model"
NNUNET_BASE = ROOT / "nnUNet"
NNUNET_RESULTS = NNUNET_BASE / "nnUNet_results"
NNUNET_RAW = NNUNET_BASE / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_BASE / "nnUNet_preprocessed"
DATASET_ID = "001"
DATASET_NAME = "LungTumor"

# All 6 model names for the stacking
ALL_MODELS = [
    "base_8patch", "base_12patch", "base_24patch", "base_36patch",
    "nnunet_3d", "nnunet_2d"
]


def set_nnunet_env():
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)


def load_case_mapping():
    """Load the mapping from nnU-Net case IDs back to original case names."""
    mapping_file = NNUNET_RAW / f"Dataset{DATASET_ID}_{DATASET_NAME}" / "case_mapping.json"
    with open(mapping_file) as f:
        return json.load(f)  # {"case_0000": "nsclc_LUNG1-001", ...}


def generate_nnunet_predictions():
    """Generate nnU-Net predictions for all cases using trained models."""
    print("=" * 60)
    print("STEP 1: Generate nnU-Net predictions")
    print("=" * 60)

    set_nnunet_env()

    case_mapping = load_case_mapping()
    # Reverse mapping: original_name -> nnunet_case_id
    reverse_mapping = {v: k for k, v in case_mapping.items()}

    # We need to run nnU-Net inference on the preprocessed data
    # and save probability maps for each case

    for config_name, model_key in [("3d_fullres", "nnunet_3d"), ("2d", "nnunet_2d")]:
        print(f"\n  Generating {model_key} predictions...")

        trainer_dir = (NNUNET_RESULTS / f"Dataset{DATASET_ID}_{DATASET_NAME}" /
                       f"nnUNetTrainer__nnUNetPlans__{config_name}" / "fold_0")

        if not (trainer_dir / "checkpoint_final.pth").exists():
            print(f"  WARNING: {config_name} not trained, skipping")
            continue

        # Load the nnU-Net trainer to run inference
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,  # no TTA for speed
            perform_everything_on_device=True,
            device=torch.device('cuda'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )

        predictor.initialize_from_trained_model_folder(
            str(NNUNET_RESULTS / f"Dataset{DATASET_ID}_{DATASET_NAME}" /
                f"nnUNetTrainer__nnUNetPlans__{config_name}"),
            use_folds=(0,),
            checkpoint_name="checkpoint_final.pth",
        )

        # Process each tumor case
        tumor_cases = sorted([d for d in TUMOR_DIR.iterdir() if d.is_dir()])
        print(f"  Processing {len(tumor_cases)} cases...")

        for case_dir in tqdm(tumor_cases, desc=f"  {model_key}"):
            case_name = case_dir.name
            cache_file = CACHE_DIR / f"{case_name}.npz"

            # Check if this prediction already exists in cache
            if cache_file.exists():
                data = np.load(cache_file, allow_pickle=True)
                if model_key in data.files:
                    continue

            ct_path = case_dir / "ct.nii.gz"
            if not ct_path.exists():
                continue

            try:
                # Load CT volume
                ct_nib = nib.load(str(ct_path))
                ct_data = ct_nib.get_fdata().astype(np.float32)

                # nnU-Net expects [C, H, W, D] with channel first
                ct_input = ct_data[np.newaxis]  # [1, H, W, D]

                # Run prediction - returns softmax probabilities
                # predict_single_npy_array returns [C, H, W, D] softmax
                prediction = predictor.predict_single_npy_array(
                    ct_input,
                    {'spacing': ct_nib.header.get_zooms()[:3]},
                    None,
                    None,
                    True,  # return probabilities
                )

                # prediction shape is [num_classes, H, W, D]
                # We want the foreground (tumor) probability = class 1
                if prediction.ndim == 4 and prediction.shape[0] >= 2:
                    prob_map = prediction[1]  # foreground class
                elif prediction.ndim == 3:
                    prob_map = prediction
                else:
                    prob_map = prediction[0]

                # Resize to match the stacking target size (128, 128, 128)
                from scipy.ndimage import zoom as scipy_zoom
                target = (128, 128, 128)
                if prob_map.shape != target:
                    factors = [t / s for t, s in zip(target, prob_map.shape)]
                    prob_map = scipy_zoom(prob_map, factors, order=1)

                # Save to cache - merge with existing cache if present
                if cache_file.exists():
                    existing = dict(np.load(cache_file, allow_pickle=True))
                    existing[model_key] = prob_map.astype(np.float16)
                    np.savez_compressed(cache_file, **existing)
                else:
                    # Need mask too
                    seg_path = case_dir / "seg.nii.gz"
                    if seg_path.exists():
                        mask = nib.load(str(seg_path)).get_fdata()
                        if mask.shape != target:
                            mask = scipy_zoom(mask, [t / s for t, s in zip(target, mask.shape)], order=0)
                        mask = (mask > 0.5).astype(np.uint8)
                        np.savez_compressed(cache_file, mask=mask,
                                            **{model_key: prob_map.astype(np.float16)})

            except Exception as e:
                print(f"  ERROR {case_name}: {e}")

            gc.collect()
            torch.cuda.empty_cache()

        del predictor
        gc.collect()
        torch.cuda.empty_cache()


def merge_old_cache():
    """Merge existing 4-model stacking cache into the new 6-model cache."""
    print("\n" + "=" * 60)
    print("STEP 2: Merge existing custom model predictions")
    print("=" * 60)

    old_cache = MODEL_DIR / "stacking_cache"
    if not old_cache.exists():
        print("  No existing cache found — will need to regenerate custom model predictions")
        return False

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    custom_models = ["base_8patch", "base_12patch", "base_24patch", "base_36patch"]

    old_files = sorted(old_cache.glob("*.npz"))
    print(f"  Merging {len(old_files)} cases from old cache...")

    merged = 0
    for old_file in tqdm(old_files, desc="  Merging"):
        new_file = CACHE_DIR / old_file.name

        old_data = dict(np.load(old_file, allow_pickle=True))

        if new_file.exists():
            # Merge: keep nnU-Net preds from new, add custom from old
            new_data = dict(np.load(new_file, allow_pickle=True))
            for key in custom_models:
                if key in old_data and key not in new_data:
                    new_data[key] = old_data[key]
            np.savez_compressed(new_file, **new_data)
        else:
            # Copy old cache as starting point
            np.savez_compressed(new_file, **old_data)

        merged += 1

    print(f"  Merged {merged} cases")
    return True


def verify_cache():
    """Check which cases have all 6 model predictions."""
    print("\n" + "=" * 60)
    print("STEP 3: Verify cache completeness")
    print("=" * 60)

    cache_files = sorted(CACHE_DIR.glob("*.npz"))
    complete = 0
    incomplete = []

    for f in cache_files:
        data = np.load(f, allow_pickle=True)
        available = set(data.files) - {"mask"}
        missing = set(ALL_MODELS) - available
        if not missing:
            complete += 1
        else:
            incomplete.append((f.stem, missing))

    print(f"  Total cached: {len(cache_files)}")
    print(f"  Complete (all 6 models): {complete}")
    print(f"  Incomplete: {len(incomplete)}")

    if incomplete[:5]:
        print(f"  Sample incomplete cases:")
        for name, missing in incomplete[:5]:
            print(f"    {name}: missing {missing}")

    return complete, len(cache_files)


def train_stacking(epochs, fg_ratio, lr, batch_size, stacking_patch, stacking_overlap):
    """Train stacking classifier with all 6 models."""
    print("\n" + "=" * 60)
    print(f"STEP 4: Train stacking classifier ({epochs} epochs, 8 channels)")
    print("=" * 60)

    # Use only complete cases
    model_list = ",".join(ALL_MODELS)

    cmd = [
        PYTHON, str(ROOT / "scripts" / "training" / "train_stacking.py"),
        "--data-dir", str(TUMOR_DIR),
        "--epochs", str(epochs),
        "--models", model_list,
        "--cache-dir", str(CACHE_DIR),
        "--fg-ratio", str(fg_ratio),
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--stacking-patch", str(stacking_patch),
        "--stacking-overlap", str(stacking_overlap),
        "--regen-overlap", "0.5",
    ]

    print(f"  Models: {ALL_MODELS}")
    print(f"  Input channels: {len(ALL_MODELS) + 2} (6 preds + variance + range)")
    print(f"  $ {' '.join(str(c) for c in cmd)}\n")

    result = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Re-run stacking with nnU-Net + custom models (8-channel)"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--fg-ratio", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stacking-patch", type=int, default=32)
    parser.add_argument("--stacking-overlap", type=float, default=0.5)
    parser.add_argument("--skip-predict", action="store_true",
                        help="Skip nnU-Net prediction generation")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merging old cache")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip stacking training (just generate predictions)")
    args = parser.parse_args()

    print(f"""
################################################################################
#  RE-STACKING WITH nnU-Net (6 base models, 8-channel fusion)                  #
#                                                                              #
#  Models: 4 custom + nnU-Net 3D + nnU-Net 2D                                 #
#  Channels: 6 predictions + variance + range = 8                              #
#  Epochs: {args.epochs:<60}#
#  Cache: {str(CACHE_DIR):<62}#
################################################################################
""")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge old custom model cache
    if not args.skip_merge:
        merge_old_cache()

    # Step 2: Generate nnU-Net predictions
    if not args.skip_predict:
        generate_nnunet_predictions()

    # Step 3: Verify
    complete, total = verify_cache()

    if complete == 0:
        print("\n  ERROR: No complete cases found. Need both custom + nnU-Net predictions.")
        print("  Run without --skip-predict and --skip-merge flags.")
        sys.exit(1)

    # Step 4: Train stacking
    if not args.skip_train:
        # Update stacking script to use new cache dir
        train_stacking(
            args.epochs, args.fg_ratio, args.lr, args.batch_size,
            args.stacking_patch, args.stacking_overlap
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
