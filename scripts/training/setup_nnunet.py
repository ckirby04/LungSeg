"""
Convert tumor-only dataset to nnU-Net v2 format and set up environment.

Creates:
    nnUNet_raw/Dataset001_LungTumor/
        dataset.json
        imagesTr/
            case_001_0000.nii.gz   # CT (single channel)
        labelsTr/
            case_001.nii.gz        # Binary tumor mask

Usage:
    python scripts/training/setup_nnunet.py
    python scripts/training/setup_nnunet.py --dataset-id 001
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

TUMOR_DIR = ROOT / "data" / "preprocessed" / "tumor_only"
COMBINED_DIR = ROOT / "data" / "preprocessed" / "combined"
NNUNET_BASE = ROOT / "nnUNet"
EXCLUDE_PREFIXES = ["covid_"]


def setup_environment(dataset_id):
    """Set nnU-Net environment variables."""
    raw_dir = NNUNET_BASE / "nnUNet_raw"
    preprocessed_dir = NNUNET_BASE / "nnUNet_preprocessed"
    results_dir = NNUNET_BASE / "nnUNet_results"

    for d in [raw_dir, preprocessed_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(raw_dir)
    os.environ["nnUNet_preprocessed"] = str(preprocessed_dir)
    os.environ["nnUNet_results"] = str(results_dir)

    print(f"  nnUNet_raw:          {raw_dir}")
    print(f"  nnUNet_preprocessed: {preprocessed_dir}")
    print(f"  nnUNet_results:      {results_dir}")

    return raw_dir, preprocessed_dir, results_dir


def convert_to_nnunet(dataset_id="001", dataset_name="LungTumor"):
    """Convert tumor-only cases to nnU-Net format."""

    raw_dir, _, _ = setup_environment(dataset_id)

    # Find tumor cases
    source_dir = TUMOR_DIR if TUMOR_DIR.exists() else COMBINED_DIR
    if not source_dir.exists():
        print(f"  ERROR: No data found at {source_dir}")
        sys.exit(1)

    all_cases = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    tumor_cases = [
        d for d in all_cases
        if not any(d.name.startswith(p) for p in EXCLUDE_PREFIXES)
    ]
    print(f"  Source: {source_dir}")
    print(f"  Tumor cases: {len(tumor_cases)}")

    # Create nnU-Net dataset directory
    dataset_dir = raw_dir / f"Dataset{dataset_id}_{dataset_name}"
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Link/copy cases
    training_list = []
    skipped = 0

    for i, case_dir in enumerate(tumor_cases):
        case_id = f"case_{i:04d}"
        ct_src = case_dir / "ct.nii.gz"
        seg_src = case_dir / "seg.nii.gz"

        if not ct_src.exists() or not seg_src.exists():
            skipped += 1
            continue

        # nnU-Net naming: case_XXXX_0000.nii.gz for channel 0 (CT)
        ct_dst = images_dir / f"{case_id}_0000.nii.gz"
        seg_dst = labels_dir / f"{case_id}.nii.gz"

        # Use hard links to save space, fall back to symlinks then copy
        for src, dst in [(ct_src, ct_dst), (seg_src, seg_dst)]:
            if dst.exists():
                continue
            try:
                os.link(str(src), str(dst))
            except (OSError, NotImplementedError):
                try:
                    os.symlink(str(src), str(dst))
                except (OSError, NotImplementedError):
                    import shutil
                    shutil.copy2(str(src), str(dst))

        training_list.append({
            "image": f"./imagesTr/{case_id}_0000.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz"
        })

    # Create dataset.json
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "tumor": 1
        },
        "numTraining": len(training_list),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "Lung tumor segmentation from CT (MSD + NSCLC-Radiomics + RIDER)",
        "reference": "Combined multi-source dataset",
        "licence": "Research use",
        "tensorImageSize": "3D",
    }

    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # Also save a mapping file for reference
    mapping = {}
    for i, case_dir in enumerate(tumor_cases):
        ct_src = case_dir / "ct.nii.gz"
        seg_src = case_dir / "seg.nii.gz"
        if ct_src.exists() and seg_src.exists():
            mapping[f"case_{i:04d}"] = case_dir.name

    with open(dataset_dir / "case_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\n  Dataset: {dataset_dir}")
    print(f"  Training cases: {len(training_list)}")
    print(f"  Skipped: {skipped}")
    print(f"  dataset.json written")
    print(f"  case_mapping.json written")

    return dataset_dir, len(training_list)


def main():
    parser = argparse.ArgumentParser(description="Setup nnU-Net dataset for lung tumor segmentation")
    parser.add_argument("--dataset-id", type=str, default="001")
    parser.add_argument("--dataset-name", type=str, default="LungTumor")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  nnU-Net Dataset Setup")
    print(f"{'='*60}\n")

    dataset_dir, n_cases = convert_to_nnunet(args.dataset_id, args.dataset_name)

    print(f"\n{'='*60}")
    print(f"  Next steps:")
    print(f"{'='*60}")
    print(f"""
  1. Plan and preprocess:
     nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity

  2. Train 3D full resolution (fold 0):
     nnUNetv2_train {args.dataset_id} 3d_fullres 0

  3. Train 2D (fold 0):
     nnUNetv2_train {args.dataset_id} 2d 0

  4. Or use the full training script:
     python scripts/training/train_full_pipeline.py
""")


if __name__ == "__main__":
    main()
