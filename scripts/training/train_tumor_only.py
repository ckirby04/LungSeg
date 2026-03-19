"""
Tumor-only long training pipeline.

Excludes COVID infection cases and runs extended training:
  - Base models: 500 epochs, 10 patches/volume, cosine annealing
  - Stacking classifier: 300 epochs, higher fg_ratio
  - Full evaluation with threshold tuning

Usage:
    python scripts/training/train_tumor_only.py
    python scripts/training/train_tumor_only.py --base-epochs 750 --stacking-epochs 500
    python scripts/training/train_tumor_only.py --resume  # resume from last checkpoint
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable

COMBINED_DIR = ROOT / "data" / "preprocessed" / "combined"
TUMOR_DIR = ROOT / "data" / "preprocessed" / "tumor_only"
MODEL_DIR = ROOT / "model"
BASE_MODEL_DIR = MODEL_DIR / "base_models"
RESULTS_DIR = ROOT / "results"

TRAIN_BASE_SCRIPT = ROOT / "scripts" / "training" / "train_base_model.py"
TRAIN_STACKING_SCRIPT = ROOT / "scripts" / "training" / "train_stacking.py"
EVALUATE_SCRIPT = ROOT / "scripts" / "evaluation" / "evaluate.py"

# Prefixes to EXCLUDE (non-tumor lesion types)
EXCLUDE_PREFIXES = ["covid_"]


def banner(text):
    w = 70
    print(f"\n{'#' * w}")
    print(f"# {text:<{w - 4}} #")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<{w - 4}} #")
    print(f"{'#' * w}\n")


def run_bg(cmd, desc):
    print(f"  [bg] {desc}")
    print(f"       $ {' '.join(str(c) for c in cmd)}")
    return subprocess.Popen([str(c) for c in cmd], cwd=str(ROOT))


# =========================================================================
# STEP 1: Build tumor-only dataset (symlinks to avoid copying)
# =========================================================================
def build_tumor_dataset():
    banner("STEP 1: Build tumor-only dataset")

    TUMOR_DIR.mkdir(parents=True, exist_ok=True)

    if not COMBINED_DIR.exists():
        print("  ERROR: Combined dataset not found.")
        return False

    all_cases = sorted([d for d in COMBINED_DIR.iterdir() if d.is_dir()])
    tumor_cases = [
        d for d in all_cases
        if not any(d.name.startswith(p) for p in EXCLUDE_PREFIXES)
    ]

    excluded = len(all_cases) - len(tumor_cases)
    print(f"  Combined dataset: {len(all_cases)} cases")
    print(f"  Excluded (non-tumor): {excluded} cases")
    print(f"  Tumor-only: {len(tumor_cases)} cases")

    # Breakdown
    prefixes = {}
    for d in tumor_cases:
        p = d.name.split("_")[0]
        prefixes[p] = prefixes.get(p, 0) + 1
    for p, n in sorted(prefixes.items()):
        print(f"    {p}: {n}")

    # Copy (or symlink) cases
    new_count = 0
    for src_dir in tumor_cases:
        dst_dir = TUMOR_DIR / src_dir.name
        if dst_dir.exists():
            continue
        # Use junction/symlink on Windows, copy as fallback
        try:
            os.symlink(str(src_dir), str(dst_dir), target_is_directory=True)
        except (OSError, NotImplementedError):
            shutil.copytree(str(src_dir), str(dst_dir))
        new_count += 1

    existing = len([d for d in TUMOR_DIR.iterdir() if d.is_dir()])
    print(f"  Tumor dataset ready: {existing} cases ({new_count} newly linked)")
    return True


# =========================================================================
# STEP 2: Train base models (extended)
# =========================================================================
def train_base_models(epochs, patches_per_vol, gpu0, gpu1):
    banner(f"STEP 2: Train base models ({epochs} epochs, {patches_per_vol} patches/vol)")

    cases = [d for d in TUMOR_DIR.iterdir() if d.is_dir()]
    print(f"  Training data: {len(cases)} tumor-only cases")
    print(f"  Epochs: {epochs}")
    print(f"  Patches/volume: {patches_per_vol}")
    print(f"  GPUs: {gpu0}, {gpu1}")

    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Batch 1: 8-patch (GPU 0) + 24-patch (GPU 1)
    print(f"\n  Batch 1: 8-patch (GPU {gpu0}) + 24-patch (GPU {gpu1})")
    procs = []
    for ps, gpu in [(8, gpu0), (24, gpu1)]:
        p = run_bg(
            [PYTHON, str(TRAIN_BASE_SCRIPT),
             "--patch-size", str(ps),
             "--gpu", str(gpu),
             "--epochs", str(epochs),
             "--patches-per-volume", str(patches_per_vol),
             "--lr", "0.0005",
             "--data-dir", str(TUMOR_DIR)],
            f"base_{ps}patch on GPU {gpu}"
        )
        procs.append((f"base_{ps}patch", p))

    for name, proc in procs:
        print(f"\n  Waiting for {name} (PID {proc.pid})...")
        proc.wait()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  {name}: {status}")

    # Batch 2: 12-patch (GPU 0) + 36-patch (GPU 1)
    print(f"\n  Batch 2: 12-patch (GPU {gpu0}) + 36-patch (GPU {gpu1})")
    procs = []
    for ps, gpu in [(12, gpu0), (36, gpu1)]:
        p = run_bg(
            [PYTHON, str(TRAIN_BASE_SCRIPT),
             "--patch-size", str(ps),
             "--gpu", str(gpu),
             "--epochs", str(epochs),
             "--patches-per-volume", str(patches_per_vol),
             "--lr", "0.0005",
             "--data-dir", str(TUMOR_DIR)],
            f"base_{ps}patch on GPU {gpu}"
        )
        procs.append((f"base_{ps}patch", p))

    for name, proc in procs:
        print(f"\n  Waiting for {name} (PID {proc.pid})...")
        proc.wait()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  {name}: {status}")

    # Print results
    print("\n  Base model results:")
    for ps in [8, 12, 24, 36]:
        state_path = BASE_MODEL_DIR / f"base_{ps}patch_state.json"
        if state_path.exists():
            with open(state_path) as f:
                s = json.load(f)
            print(f"    {ps}-patch: best_dice={s['best_val_dice']:.4f} "
                  f"(epoch {s['best_epoch']}/{s['epochs_total']}), "
                  f"sens={s['sensitivity']:.3f}, tiny={s.get('best_tiny_dice','N/A')}")

    return True


# =========================================================================
# STEP 3: Train stacking classifier (extended)
# =========================================================================
def train_stacking(epochs):
    banner(f"STEP 3: Train stacking classifier ({epochs} epochs)")

    missing = []
    for ps in [8, 12, 24, 36]:
        if not (BASE_MODEL_DIR / f"base_{ps}patch_best.pth").exists():
            missing.append(str(ps))
    if missing:
        print(f"  WARNING: Missing base models for patch sizes: {', '.join(missing)}")

    cmd = [
        PYTHON, str(TRAIN_STACKING_SCRIPT),
        "--data-dir", str(TUMOR_DIR),
        "--epochs", str(epochs),
        "--stacking-patch", "32",
        "--stacking-overlap", "0.5",
        "--fg-ratio", "0.8",
        "--lr", "0.0005",
        "--batch-size", "4",
        "--regen-overlap", "0.5",
    ]

    print(f"  $ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    return result.returncode == 0


# =========================================================================
# STEP 4: Evaluate
# =========================================================================
def evaluate():
    banner("STEP 4: Full evaluation")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(EVALUATE_SCRIPT),
        "--data-dir", str(TUMOR_DIR),
    ]
    print(f"  $ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    return result.returncode == 0


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Tumor-only extended training pipeline"
    )
    parser.add_argument("--base-epochs", type=int, default=1000,
                        help="Base model training epochs (default: 1000)")
    parser.add_argument("--stacking-epochs", type=int, default=300,
                        help="Stacking classifier epochs (default: 300)")
    parser.add_argument("--patches-per-volume", type=int, default=10,
                        help="Patches per volume (default: 10)")
    parser.add_argument("--gpu0", type=int, default=0)
    parser.add_argument("--gpu1", type=int, default=1)
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip base model training (use existing checkpoints)")
    parser.add_argument("--skip-stacking", action="store_true",
                        help="Skip stacking training")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation")
    args = parser.parse_args()

    start = time.time()

    print(f"""
################################################################################
#                                                                              #
#   TUMOR-ONLY EXTENDED TRAINING PIPELINE                                      #
#                                                                              #
#   Base model epochs:     {args.base_epochs:<46}#
#   Stacking epochs:       {args.stacking_epochs:<46}#
#   Patches per volume:    {args.patches_per_volume:<46}#
#   Learning rate:         0.0005 (lower for longer training)                  #
#   FG ratio (stacking):   0.8                                                #
#   GPUs:                  {args.gpu0}, {args.gpu1:<44}#
#                                                                              #
################################################################################
""")

    # Step 1: Build tumor-only dataset
    if not build_tumor_dataset():
        sys.exit(1)

    # Step 2: Train base models
    if not args.skip_base:
        if not train_base_models(args.base_epochs, args.patches_per_volume,
                                  args.gpu0, args.gpu1):
            print("  Base model training had errors, continuing anyway...")

    # Step 3: Train stacking
    if not args.skip_stacking:
        if not train_stacking(args.stacking_epochs):
            print("  Stacking training had errors, continuing anyway...")

    # Step 4: Evaluate
    if not args.skip_eval:
        evaluate()

    elapsed = time.time() - start
    hours = elapsed / 3600

    # Final summary
    print(f"""
################################################################################
#                                                                              #
#   TUMOR-ONLY TRAINING COMPLETE                                               #
#   Total time: {hours:.1f} hours{'':.<54}#
#                                                                              #
#   Base models:  {str(BASE_MODEL_DIR):<54}#
#   Results:      {str(RESULTS_DIR):<54}#
#                                                                              #
#   Next: python scripts/inference/run_inference.py --input <ct.nii.gz>        #
#                                                                              #
################################################################################
""")

    # Print final results
    stacking_results = MODEL_DIR / "stacking_results.json"
    if stacking_results.exists():
        with open(stacking_results) as f:
            results = json.load(f)
        print("\nFinal method comparison:")
        print(f"{'Method':<25} {'Dice':<10} {'Sens':<10} {'Prec':<10}")
        print("-" * 55)
        for method, metrics in sorted(
            [(k, v) for k, v in results.items() if not k.startswith('_')],
            key=lambda x: -x[1].get('dice', 0)
        ):
            print(f"{method:<25} {metrics['dice']:.4f}     "
                  f"{metrics['sensitivity']:.4f}     {metrics['precision']:.4f}")

    eval_results = RESULTS_DIR / "evaluation_results.json"
    if eval_results.exists():
        with open(eval_results) as f:
            ev = json.load(f)
        print(f"\nEvaluation (stacking ensemble):")
        print(f"  Voxel Dice:  {ev['voxel_dice_mean']:.4f} +/- {ev['voxel_dice_std']:.4f}")
        print(f"  Lesion F1:   {ev.get('lesion_f1', 'N/A')}")
        print(f"  Lesion Recall: {ev.get('lesion_recall', 'N/A')}")


if __name__ == "__main__":
    main()
