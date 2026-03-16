"""
Overnight pipeline for lung CT segmentation.

Phase 1 (~6-8h): Train 4 base models
Phase 2 (~3-4h): Build stacking cache + train stacking classifier
Phase 3 (~1h):   Full evaluation

GPU assignment:
  GPU 0: 8-patch + 12-patch (sequential)
  GPU 1: 24-patch + 36-patch (sequential)
  Phases 2/3: GPU 0 (or GPU 1 if available)

Usage:
    python scripts/training/overnight.py
    python scripts/training/overnight.py --epochs 250
    python scripts/training/overnight.py --skip-phase1  # just rebuild stacking
"""

import argparse
import gc
import json
import subprocess
import sys
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SCRIPT = ROOT / "scripts" / "training" / "train_base_model.py"
STACKING_SCRIPT = ROOT / "scripts" / "training" / "train_stacking.py"
DATA_DIR = ROOT / "data" / "preprocessed" / "train"
MODEL_DIR = ROOT / "model"
BASE_MODEL_DIR = ROOT / "model" / "base_models"
RESULTS_DIR = ROOT / "results"

PYTHON = sys.executable

BASE_MODELS = {
    8:  "base_8patch",
    12: "base_12patch",
    24: "base_24patch",
    36: "base_36patch",
}


def run_cmd(cmd, desc, wait=True):
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(ROOT),
        stdout=None,
        stderr=subprocess.STDOUT,
    )
    if wait:
        proc.wait()
        return proc.returncode
    return proc


def main():
    parser = argparse.ArgumentParser(description="Lung CT overnight training pipeline")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patches-per-volume", type=int, default=5)
    parser.add_argument("--skip-phase1", action="store_true", help="Skip base model training")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip stacking training")
    parser.add_argument("--skip-phase3", action="store_true", help="Skip final evaluation")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--gpu0", type=int, default=0)
    parser.add_argument("--gpu1", type=int, default=1)
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = DATA_DIR

    start_time = time.time()

    print(f"""
################################################################################
#                                                                              #
#   LUNG CT SEGMENTATION OVERNIGHT PIPELINE                                    #
#                                                                              #
#   Phase 1: Train 4 base models (8/12/24/36 patch)                           #
#   Phase 2: Build stacking cache + train stacking classifier                  #
#   Phase 3: Full evaluation                                                   #
#                                                                              #
#   Epochs: {args.epochs}                                                           #
#   Data: {data_dir}
#                                                                              #
################################################################################
""")

    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # PHASE 1: Train 4 base models in parallel
    # ========================================================================
    if not args.skip_phase1:
        print(f"\n{'#'*70}")
        print(f"# PHASE 1: Train base models")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}")

        # Batch 1: 8-patch (GPU 0) + 24-patch (GPU 1) in parallel
        print(f"\n  Batch 1: 8-patch (GPU {args.gpu0}) + 24-patch (GPU {args.gpu1})")
        procs = []
        for ps, gpu in [(8, args.gpu0), (24, args.gpu1)]:
            p = run_cmd(
                [PYTHON, str(TRAIN_SCRIPT),
                 "--patch-size", str(ps), "--gpu", str(gpu),
                 "--epochs", str(args.epochs),
                 "--patches-per-volume", str(args.patches_per_volume),
                 "--data-dir", str(data_dir)],
                f"Base {ps}-patch on GPU {gpu}",
                wait=False,
            )
            procs.append((f"{ps}-patch", p))

        for name, proc in procs:
            print(f"\n  Waiting for {name} (PID {proc.pid})...")
            proc.wait()
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"  {name}: {status}")

        # Batch 2: 12-patch (GPU 0) + 36-patch (GPU 1) in parallel
        print(f"\n  Batch 2: 12-patch (GPU {args.gpu0}) + 36-patch (GPU {args.gpu1})")
        procs = []
        for ps, gpu in [(12, args.gpu0), (36, args.gpu1)]:
            p = run_cmd(
                [PYTHON, str(TRAIN_SCRIPT),
                 "--patch-size", str(ps), "--gpu", str(gpu),
                 "--epochs", str(args.epochs),
                 "--patches-per-volume", str(args.patches_per_volume),
                 "--data-dir", str(data_dir)],
                f"Base {ps}-patch on GPU {gpu}",
                wait=False,
            )
            procs.append((f"{ps}-patch", p))

        for name, proc in procs:
            print(f"\n  Waiting for {name} (PID {proc.pid})...")
            proc.wait()
            status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
            print(f"  {name}: {status}")

        # Print training results summary
        print(f"\n  Phase 1 training results:")
        for ps in [8, 12, 24, 36]:
            state_path = BASE_MODEL_DIR / f"base_{ps}patch_state.json"
            if state_path.exists():
                with open(state_path) as f:
                    state = json.load(f)
                print(f"    {ps}-patch: best_dice={state['best_val_dice']:.4f} "
                      f"(ep {state['best_epoch']}), tiny={state.get('best_tiny_dice', 'N/A')}")

    # ========================================================================
    # PHASE 2: Build stacking cache + train stacking classifier
    # ========================================================================
    if not args.skip_phase2:
        print(f"\n{'#'*70}")
        print(f"# PHASE 2: Build stacking cache + train stacking classifier")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}")

        run_cmd(
            [PYTHON, str(STACKING_SCRIPT),
             "--data-dir", f"data/preprocessed/train",
             "--epochs", "150",
             "--stacking-patch", "32",
             "--stacking-overlap", "0.5"],
            "Stacking classifier training",
        )

    # ========================================================================
    # PHASE 3: Full evaluation
    # ========================================================================
    if not args.skip_phase3:
        print(f"\n{'#'*70}")
        print(f"# PHASE 3: Full evaluation")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}")

        eval_script = ROOT / "scripts" / "evaluation" / "evaluate.py"
        if eval_script.exists():
            run_cmd(
                [PYTHON, str(eval_script),
                 "--data-dir", str(data_dir)],
                "Full evaluation",
            )
        else:
            print("  Evaluation script not found, skipping Phase 3")

    total_hours = (time.time() - start_time) / 3600
    print(f"""
################################################################################
#                                                                              #
#   OVERNIGHT PIPELINE COMPLETE                                                #
#   Total time: {total_hours:.1f} hours                                              #
#   Base models: {BASE_MODEL_DIR}
#   Results: {RESULTS_DIR}
#                                                                              #
################################################################################
""")


if __name__ == "__main__":
    main()
