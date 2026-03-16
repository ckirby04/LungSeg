"""
Unified pipeline: Preprocess → Train → Evaluate for lung CT segmentation.

This script handles the full workflow:
  1. Extract the MSD Task06_Lung dataset (if tar file exists)
  2. Preprocess raw data to isotropic spacing + target size
  3. Train 4 base models (8/12/24/36 patch) across 2 GPUs
  4. Train stacking classifier (meta-learner)
  5. Evaluate on held-out validation set

Usage:
    python run_pipeline.py                        # Full pipeline
    python run_pipeline.py --step preprocess      # Only preprocess
    python run_pipeline.py --step train            # Only train (base + stacking)
    python run_pipeline.py --step train-base       # Only train base models
    python run_pipeline.py --step train-stacking   # Only train stacking
    python run_pipeline.py --step eval             # Only evaluate
    python run_pipeline.py --epochs 100            # Fewer epochs for quick test
    python run_pipeline.py --quick                 # Quick test run (20 epochs)
"""

import argparse
import gc
import json
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# Directories
RAW_DIR = ROOT / "data" / "raw"
MSD_DIR = RAW_DIR / "Task06_Lung"
MSD_TAR = RAW_DIR / "Task06_Lung.tar"
PREPROCESSED_DIR = ROOT / "data" / "preprocessed" / "train"
MODEL_DIR = ROOT / "model"
BASE_MODEL_DIR = MODEL_DIR / "base_models"
RESULTS_DIR = ROOT / "results"

# Scripts
PREPROCESS_SCRIPT = ROOT / "scripts" / "preprocessing" / "preprocess_lung.py"
TRAIN_BASE_SCRIPT = ROOT / "scripts" / "training" / "train_base_model.py"
TRAIN_STACKING_SCRIPT = ROOT / "scripts" / "training" / "train_stacking.py"
EVALUATE_SCRIPT = ROOT / "scripts" / "evaluation" / "evaluate.py"


def banner(text):
    width = 70
    print(f"\n{'#' * width}")
    print(f"# {text:<{width - 4}} #")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<{width - 4}} #")
    print(f"{'#' * width}\n")


def run(cmd, desc):
    """Run a subprocess and stream output."""
    print(f"  → {desc}")
    print(f"  $ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n  ✗ FAILED (exit code {result.returncode}): {desc}")
        return False
    print(f"\n  ✓ {desc}")
    return True


def run_background(cmd, desc):
    """Run a subprocess in background, return the process."""
    print(f"  → [background] {desc}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(ROOT),
    )


# =========================================================================
# STEP 1: Extract dataset
# =========================================================================
def step_extract():
    banner("STEP 1: Extract MSD Task06_Lung dataset")

    if MSD_DIR.exists() and (MSD_DIR / "dataset.json").exists():
        print("  Dataset already extracted, skipping.")
        return True

    if not MSD_TAR.exists():
        print(f"  ERROR: {MSD_TAR} not found.")
        print("  Please download the MSD Task06_Lung dataset first.")
        print("  The download should be running in the background.")
        return False

    print(f"  Extracting {MSD_TAR}...")
    with tarfile.open(str(MSD_TAR), "r") as tar:
        tar.extractall(path=str(RAW_DIR))
    print("  Extraction complete.")
    return True


# =========================================================================
# STEP 2: Preprocess
# =========================================================================
def step_preprocess(target_size=(256, 256, 256)):
    banner("STEP 2: Preprocess data")

    if PREPROCESSED_DIR.exists():
        cases = [d for d in PREPROCESSED_DIR.iterdir() if d.is_dir()]
        if len(cases) > 0:
            print(f"  Found {len(cases)} already preprocessed cases.")
            print("  To re-preprocess, delete data/preprocessed/train/ first.")
            return True

    if not MSD_DIR.exists():
        print("  ERROR: MSD dataset not found. Run extract step first.")
        return False

    size_args = [str(s) for s in target_size]
    return run(
        [PYTHON, str(PREPROCESS_SCRIPT),
         "--input-dir", str(MSD_DIR),
         "--output-dir", str(PREPROCESSED_DIR),
         "--format", "msd",
         "--target-size", *size_args],
        "Preprocessing MSD Task06_Lung data"
    )


# =========================================================================
# STEP 3: Train base models
# =========================================================================
def step_train_base(epochs=250, patches_per_volume=5, gpu0=0, gpu1=1):
    banner("STEP 3: Train base models")

    if not PREPROCESSED_DIR.exists():
        print("  ERROR: Preprocessed data not found. Run preprocess step first.")
        return False

    cases = [d for d in PREPROCESSED_DIR.iterdir() if d.is_dir()]
    print(f"  Training data: {len(cases)} cases")
    print(f"  Epochs: {epochs}")
    print(f"  GPUs: {gpu0}, {gpu1}")

    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Batch 1: 8-patch (GPU 0) + 24-patch (GPU 1) in parallel
    print(f"\n  Batch 1: 8-patch (GPU {gpu0}) + 24-patch (GPU {gpu1})")
    procs = []
    for ps, gpu in [(8, gpu0), (24, gpu1)]:
        p = run_background(
            [PYTHON, str(TRAIN_BASE_SCRIPT),
             "--patch-size", str(ps),
             "--gpu", str(gpu),
             "--epochs", str(epochs),
             "--patches-per-volume", str(patches_per_volume),
             "--data-dir", str(PREPROCESSED_DIR)],
            f"Training base_{ps}patch on GPU {gpu}"
        )
        procs.append((f"base_{ps}patch", p))

    for name, proc in procs:
        proc.wait()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  {name}: {status}")

    # Batch 2: 12-patch (GPU 0) + 36-patch (GPU 1) in parallel
    print(f"\n  Batch 2: 12-patch (GPU {gpu0}) + 36-patch (GPU {gpu1})")
    procs = []
    for ps, gpu in [(12, gpu0), (36, gpu1)]:
        p = run_background(
            [PYTHON, str(TRAIN_BASE_SCRIPT),
             "--patch-size", str(ps),
             "--gpu", str(gpu),
             "--epochs", str(epochs),
             "--patches-per-volume", str(patches_per_volume),
             "--data-dir", str(PREPROCESSED_DIR)],
            f"Training base_{ps}patch on GPU {gpu}"
        )
        procs.append((f"base_{ps}patch", p))

    for name, proc in procs:
        proc.wait()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  {name}: {status}")

    # Summary
    print("\n  Base model training results:")
    for ps in [8, 12, 24, 36]:
        state_path = BASE_MODEL_DIR / f"base_{ps}patch_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            print(f"    {ps}-patch: best_dice={state['best_val_dice']:.4f} "
                  f"(epoch {state['best_epoch']})")
        else:
            print(f"    {ps}-patch: NOT TRAINED")

    return True


# =========================================================================
# STEP 4: Train stacking classifier
# =========================================================================
def step_train_stacking(epochs=150):
    banner("STEP 4: Train stacking classifier")

    # Check that base models exist
    missing = []
    for ps in [8, 12, 24, 36]:
        path = BASE_MODEL_DIR / f"base_{ps}patch_best.pth"
        if not path.exists():
            missing.append(f"base_{ps}patch_best.pth")

    if missing:
        print(f"  WARNING: Missing base models: {', '.join(missing)}")
        print("  Stacking will use whatever models are available.")

    return run(
        [PYTHON, str(TRAIN_STACKING_SCRIPT),
         "--data-dir", str(PREPROCESSED_DIR),
         "--epochs", str(epochs),
         "--stacking-patch", "32",
         "--stacking-overlap", "0.5"],
        "Training stacking classifier"
    )


# =========================================================================
# STEP 5: Evaluate
# =========================================================================
def step_evaluate():
    banner("STEP 5: Evaluate")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stacking_path = MODEL_DIR / "stacking_classifier.pth"
    if not stacking_path.exists():
        print("  WARNING: Stacking model not found. Evaluation may fail.")

    return run(
        [PYTHON, str(EVALUATE_SCRIPT),
         "--data-dir", str(PREPROCESSED_DIR)],
        "Full evaluation"
    )


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Lung CT segmentation: full pipeline (preprocess → train → eval)"
    )
    parser.add_argument(
        "--step", type=str, default="all",
        choices=["all", "extract", "preprocess", "train", "train-base",
                 "train-stacking", "eval"],
        help="Which pipeline step to run (default: all)"
    )
    parser.add_argument("--epochs", type=int, default=250,
                        help="Training epochs for base models (default: 250)")
    parser.add_argument("--stacking-epochs", type=int, default=150,
                        help="Training epochs for stacking classifier (default: 150)")
    parser.add_argument("--patches-per-volume", type=int, default=5,
                        help="Patches to extract per volume (default: 5)")
    parser.add_argument("--gpu0", type=int, default=0, help="First GPU ID")
    parser.add_argument("--gpu1", type=int, default=1, help="Second GPU ID")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run (20 base epochs, 30 stacking epochs)")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 20
        args.stacking_epochs = 30

    start = time.time()

    print(f"""
================================================================================
  LUNG CT SEGMENTATION PIPELINE

  Step:              {args.step}
  Base epochs:       {args.epochs}
  Stacking epochs:   {args.stacking_epochs}
  Patches/volume:    {args.patches_per_volume}
  GPUs:              {args.gpu0}, {args.gpu1}
  Quick mode:        {args.quick}
================================================================================
""")

    steps = {
        "extract": step_extract,
        "preprocess": step_preprocess,
        "train-base": lambda: step_train_base(args.epochs, args.patches_per_volume,
                                               args.gpu0, args.gpu1),
        "train-stacking": lambda: step_train_stacking(args.stacking_epochs),
        "eval": step_evaluate,
    }

    if args.step == "all":
        order = ["extract", "preprocess", "train-base", "train-stacking", "eval"]
    elif args.step == "train":
        order = ["train-base", "train-stacking"]
    else:
        order = [args.step]

    for step_name in order:
        ok = steps[step_name]()
        if not ok:
            print(f"\n  Pipeline stopped at step: {step_name}")
            sys.exit(1)

    elapsed = time.time() - start
    hours = elapsed / 3600
    mins = elapsed / 60

    print(f"""
================================================================================
  PIPELINE COMPLETE

  Total time: {hours:.1f} hours ({mins:.0f} minutes)

  Models:  {BASE_MODEL_DIR}
  Results: {RESULTS_DIR}

  Next steps:
    - Run inference: python scripts/inference/run_inference.py --input <ct.nii.gz>
    - Re-evaluate:   python run_pipeline.py --step eval
================================================================================
""")


if __name__ == "__main__":
    main()
