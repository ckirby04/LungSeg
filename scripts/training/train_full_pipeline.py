"""
Full training pipeline matching BrainMetScan architecture.

6 base models + stacking classifier:
  1. LightweightUNet3D  8-patch  (custom)
  2. LightweightUNet3D 12-patch  (custom)
  3. LightweightUNet3D 24-patch  (custom)
  4. LightweightUNet3D 36-patch  (custom)
  5. nnU-Net 3D full resolution
  6. nnU-Net 2D
  → Stacking classifier (8-channel: 6 preds + variance + range)

Usage:
    python scripts/training/train_full_pipeline.py
    python scripts/training/train_full_pipeline.py --skip-nnunet    # only custom models
    python scripts/training/train_full_pipeline.py --skip-custom    # only nnU-Net
    python scripts/training/train_full_pipeline.py --skip-base      # only stacking + eval
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable

# Directories
TUMOR_DIR = ROOT / "data" / "preprocessed" / "tumor_only"
COMBINED_DIR = ROOT / "data" / "preprocessed" / "combined"
MODEL_DIR = ROOT / "model"
BASE_MODEL_DIR = MODEL_DIR / "base_models"
RESULTS_DIR = ROOT / "results"
NNUNET_BASE = ROOT / "nnUNet"

# Scripts
SETUP_NNUNET = ROOT / "scripts" / "training" / "setup_nnunet.py"
TRAIN_BASE = ROOT / "scripts" / "training" / "train_base_model.py"
TRAIN_STACKING = ROOT / "scripts" / "training" / "train_stacking.py"
EVALUATE = ROOT / "scripts" / "evaluation" / "evaluate.py"

# Non-tumor prefixes to exclude
EXCLUDE_PREFIXES = ["covid_"]

DATASET_ID = "001"


def banner(text):
    w = 70
    print(f"\n{'#' * w}")
    print(f"# {text:<{w - 4}} #")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<{w - 4}} #")
    print(f"{'#' * w}\n")


def run(cmd, desc, cwd=None):
    print(f"  >> {desc}")
    print(f"  $ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run([str(c) for c in cmd], cwd=str(cwd or ROOT))
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}): {desc}")
        return False
    return True


def run_bg(cmd, desc, cwd=None):
    print(f"  [bg] {desc}")
    return subprocess.Popen([str(c) for c in cmd], cwd=str(cwd or ROOT))


def set_nnunet_env():
    """Set nnU-Net environment variables."""
    os.environ["nnUNet_raw"] = str(NNUNET_BASE / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(NNUNET_BASE / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(NNUNET_BASE / "nnUNet_results")


def build_tumor_dataset():
    """Build tumor-only dataset from combined, excluding COVID."""
    banner("STEP 0: Build tumor-only dataset")

    source = COMBINED_DIR
    if not source.exists():
        print(f"  ERROR: {source} not found")
        return False

    TUMOR_DIR.mkdir(parents=True, exist_ok=True)

    all_cases = sorted([d for d in source.iterdir() if d.is_dir()])
    tumor_cases = [d for d in all_cases if not any(d.name.startswith(p) for p in EXCLUDE_PREFIXES)]

    new = 0
    for src in tumor_cases:
        dst = TUMOR_DIR / src.name
        if not dst.exists():
            try:
                os.symlink(str(src), str(dst), target_is_directory=True)
            except OSError:
                import shutil
                shutil.copytree(str(src), str(dst))
            new += 1

    total = len([d for d in TUMOR_DIR.iterdir() if d.is_dir()])
    print(f"  Tumor-only: {total} cases ({new} newly added)")
    return True


# =========================================================================
# PHASE 1: Train 4 custom base models
# =========================================================================
def train_custom_models(epochs, patches_per_vol, lr, gpu0, gpu1):
    banner(f"PHASE 1: Train 4 custom base models ({epochs} epochs)")

    data_dir = TUMOR_DIR if TUMOR_DIR.exists() else COMBINED_DIR
    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Batch 1: 8-patch + 24-patch in parallel
    print(f"  Batch 1: 8-patch (GPU {gpu0}) + 24-patch (GPU {gpu1})")
    procs = []
    for ps, gpu in [(8, gpu0), (24, gpu1)]:
        p = run_bg([
            PYTHON, str(TRAIN_BASE),
            "--patch-size", str(ps), "--gpu", str(gpu),
            "--epochs", str(epochs), "--patches-per-volume", str(patches_per_vol),
            "--lr", str(lr), "--data-dir", str(data_dir)
        ], f"base_{ps}patch on GPU {gpu}")
        procs.append((f"base_{ps}patch", p))

    for name, p in procs:
        p.wait()
        print(f"  {name}: {'OK' if p.returncode == 0 else 'FAILED'}")

    # Batch 2: 12-patch + 36-patch in parallel
    print(f"\n  Batch 2: 12-patch (GPU {gpu0}) + 36-patch (GPU {gpu1})")
    procs = []
    for ps, gpu in [(12, gpu0), (36, gpu1)]:
        p = run_bg([
            PYTHON, str(TRAIN_BASE),
            "--patch-size", str(ps), "--gpu", str(gpu),
            "--epochs", str(epochs), "--patches-per-volume", str(patches_per_vol),
            "--lr", str(lr), "--data-dir", str(data_dir)
        ], f"base_{ps}patch on GPU {gpu}")
        procs.append((f"base_{ps}patch", p))

    for name, p in procs:
        p.wait()
        print(f"  {name}: {'OK' if p.returncode == 0 else 'FAILED'}")

    # Print results
    print("\n  Results:")
    for ps in [8, 12, 24, 36]:
        state = BASE_MODEL_DIR / f"base_{ps}patch_state.json"
        if state.exists():
            with open(state) as f:
                s = json.load(f)
            print(f"    {ps}-patch: dice={s['best_val_dice']:.4f} (epoch {s['best_epoch']})")


# =========================================================================
# PHASE 2: Setup + train nnU-Net (3D and 2D)
# =========================================================================
def train_nnunet(folds, gpu):
    banner("PHASE 2: nnU-Net training (3D full res + 2D)")

    set_nnunet_env()

    # Step 2a: Convert dataset to nnU-Net format (skip if already done)
    dataset_json = NNUNET_BASE / "nnUNet_raw" / f"Dataset{DATASET_ID}_LungTumor" / "dataset.json"
    if dataset_json.exists():
        print("  Step 2a: nnU-Net dataset already set up, skipping.")
    else:
        print("  Step 2a: Setup nnU-Net dataset...")
        if not run([PYTHON, str(SETUP_NNUNET), "--dataset-id", DATASET_ID], "nnU-Net dataset setup"):
            return False

    # Step 2b: Plan and preprocess (skip if already done)
    plans_json = NNUNET_BASE / "nnUNet_preprocessed" / f"Dataset{DATASET_ID}_LungTumor" / "nnUNetPlans.json"
    preprocessed_3d = NNUNET_BASE / "nnUNet_preprocessed" / f"Dataset{DATASET_ID}_LungTumor" / "nnUNetPlans_3d_fullres"
    if plans_json.exists() and preprocessed_3d.exists():
        print("  Step 2b: Preprocessing already complete, skipping.")
    else:
        print("\n  Step 2b: Plan and preprocess...")
        if not run([PYTHON, "-c",
                    "import sys; sys.argv = ['', '-d', '" + DATASET_ID + "', '--verify_dataset_integrity']; "
                    "from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry; "
                    "plan_and_preprocess_entry()"],
                   "nnU-Net planning and preprocessing"):
            return False

    # Check if there are existing checkpoints to resume from
    results_base = NNUNET_BASE / "nnUNet_results" / f"Dataset{DATASET_ID}_LungTumor"

    # Step 2c: Train 3D fullres then 2D sequentially
    for fold in folds:
        for config, step_label in [("3d_fullres", "2c"), ("2d", "2d")]:
            # Check if already finished
            final_ckpt = results_base / f"nnUNetTrainer__nnUNetPlans__{config}" / f"fold_{fold}" / "checkpoint_final.pth"
            if final_ckpt.exists():
                print(f"\n  Step {step_label}: {config} fold {fold} already complete, skipping.")
                continue

            # Check if there's a checkpoint to resume from
            latest_ckpt = results_base / f"nnUNetTrainer__nnUNetPlans__{config}" / f"fold_{fold}" / "checkpoint_latest.pth"
            resume_flag = "'--c', " if latest_ckpt.exists() else ""
            if latest_ckpt.exists():
                print(f"\n  Step {step_label}: Resuming {config} fold {fold} from checkpoint...")
            else:
                print(f"\n  Step {step_label}: Training {config} fold {fold} from scratch...")

            run([PYTHON, "-c",
                 f"import sys; sys.argv = ['', '{DATASET_ID}', '{config}', '{fold}', '-device', 'cuda'"
                 f"{', ' + chr(39) + '--c' + chr(39) if latest_ckpt.exists() else ''}]; "
                 "from nnunetv2.run.run_training import run_training_entry; "
                 "run_training_entry()"],
                f"nnU-Net {config} fold {fold}")

    print("\n  nnU-Net training complete!")
    return True


# =========================================================================
# PHASE 3: Generate predictions from all models + train stacking
# =========================================================================
def train_stacking(stacking_epochs, gpu):
    banner(f"PHASE 3: Stacking classifier ({stacking_epochs} epochs)")

    set_nnunet_env()
    data_dir = TUMOR_DIR if TUMOR_DIR.exists() else COMBINED_DIR

    # Determine which models are available
    available_models = []
    for ps in [8, 12, 24, 36]:
        if (BASE_MODEL_DIR / f"base_{ps}patch_best.pth").exists():
            available_models.append(f"base_{ps}patch")

    # Check for nnU-Net models
    nnunet_results = NNUNET_BASE / "nnUNet_results"
    has_nnunet_3d = (nnunet_results / f"Dataset{DATASET_ID}_LungTumor" / "nnUNetTrainer__nnUNetPlans__3d_fullres").exists()
    has_nnunet_2d = (nnunet_results / f"Dataset{DATASET_ID}_LungTumor" / "nnUNetTrainer__nnUNetPlans__2d").exists()

    print(f"  Custom models: {available_models}")
    print(f"  nnU-Net 3D: {'YES' if has_nnunet_3d else 'NO'}")
    print(f"  nnU-Net 2D: {'YES' if has_nnunet_2d else 'NO'}")

    models_str = ",".join(available_models)

    # Train stacking with available models
    # Note: nnU-Net predictions need to be generated separately via nnunet_probs.py
    # For now, stack the custom models; nnU-Net can be added later
    cmd = [
        PYTHON, str(TRAIN_STACKING),
        "--data-dir", str(data_dir),
        "--epochs", str(stacking_epochs),
        "--stacking-patch", "32",
        "--stacking-overlap", "0.5",
        "--fg-ratio", "0.8",
        "--lr", "0.0005",
        "--batch-size", "4",
        "--models", models_str,
    ]

    return run(cmd, "Stacking classifier training")


# =========================================================================
# PHASE 4: Evaluate
# =========================================================================
def evaluate():
    banner("PHASE 4: Full evaluation")
    data_dir = TUMOR_DIR if TUMOR_DIR.exists() else COMBINED_DIR
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return run([PYTHON, str(EVALUATE), "--data-dir", str(data_dir)], "Evaluation")


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Full BrainMetScan-style training pipeline for lung tumor segmentation"
    )

    # Epoch configs
    parser.add_argument("--custom-epochs", type=int, default=1000,
                        help="Custom base model epochs (default: 1000)")
    parser.add_argument("--stacking-epochs", type=int, default=300,
                        help="Stacking classifier epochs (default: 300)")
    parser.add_argument("--patches-per-volume", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0005)

    # GPU
    parser.add_argument("--gpu0", type=int, default=0)
    parser.add_argument("--gpu1", type=int, default=1)

    # nnU-Net
    parser.add_argument("--nnunet-folds", type=int, nargs="+", default=[0],
                        help="nnU-Net folds to train (default: [0])")
    parser.add_argument("--nnunet-gpu", type=int, default=0,
                        help="GPU for nnU-Net training")

    # Skip flags
    parser.add_argument("--skip-custom", action="store_true", help="Skip custom model training")
    parser.add_argument("--skip-nnunet", action="store_true", help="Skip nnU-Net training")
    parser.add_argument("--skip-base", action="store_true", help="Skip all base model training")
    parser.add_argument("--skip-stacking", action="store_true", help="Skip stacking")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")

    args = parser.parse_args()

    start = time.time()

    n_base = 0 if args.skip_base else (4 if not args.skip_custom else 0) + (2 if not args.skip_nnunet else 0)

    print(f"""
################################################################################
#                                                                              #
#   LUNG TUMOR SEGMENTATION - FULL PIPELINE (BrainMetScan Architecture)        #
#                                                                              #
#   Base models: {n_base} ({('4 custom + ' if not args.skip_custom else '') + ('nnU-Net 3D+2D' if not args.skip_nnunet else ''):<45}#
#   Custom epochs:      {args.custom_epochs:<48}#
#   Stacking epochs:    {args.stacking_epochs:<48}#
#   Patches/volume:     {args.patches_per_volume:<48}#
#   Learning rate:      {args.lr:<48}#
#   nnU-Net folds:      {str(args.nnunet_folds):<48}#
#   GPUs:               {args.gpu0}, {args.gpu1:<44}#
#                                                                              #
################################################################################
""")

    # Step 0: Build tumor-only dataset
    if not build_tumor_dataset():
        sys.exit(1)

    # Phase 1: Custom base models
    if not args.skip_base and not args.skip_custom:
        train_custom_models(args.custom_epochs, args.patches_per_volume,
                           args.lr, args.gpu0, args.gpu1)

    # Phase 2: nnU-Net
    if not args.skip_base and not args.skip_nnunet:
        train_nnunet(args.nnunet_folds, args.nnunet_gpu)

    # Phase 3: Stacking
    if not args.skip_stacking:
        train_stacking(args.stacking_epochs, args.gpu0)

    # Phase 4: Evaluate
    if not args.skip_eval:
        evaluate()

    hours = (time.time() - start) / 3600
    print(f"""
################################################################################
#                                                                              #
#   PIPELINE COMPLETE - {hours:.1f} hours{'':.<48}#
#                                                                              #
#   Models:  {str(BASE_MODEL_DIR):<59}#
#   nnU-Net: {str(NNUNET_BASE / 'nnUNet_results'):<59}#
#   Results: {str(RESULTS_DIR):<59}#
#                                                                              #
################################################################################
""")

    # Print summary
    for ps in [8, 12, 24, 36]:
        state = BASE_MODEL_DIR / f"base_{ps}patch_state.json"
        if state.exists():
            with open(state) as f:
                s = json.load(f)
            print(f"  {ps}-patch: dice={s['best_val_dice']:.4f} (epoch {s['best_epoch']})")

    stacking_results = MODEL_DIR / "stacking_results.json"
    if stacking_results.exists():
        with open(stacking_results) as f:
            r = json.load(f)
        print("\n  Stacking results:")
        for method, m in sorted(
            [(k, v) for k, v in r.items() if not k.startswith('_')],
            key=lambda x: -x[1].get('dice', 0)
        ):
            print(f"    {method:<25} dice={m['dice']:.4f} sens={m['sensitivity']:.4f}")


if __name__ == "__main__":
    main()
