"""
Preprocessing pipeline for lung CT segmentation data.

Supports multiple input formats:
  - NIfTI: Directory with ct.nii.gz + seg.nii.gz per case
  - MSD: Medical Segmentation Decathlon format (dataset.json + imagesTr/labelsTr)
  - DICOM: DICOM series with optional RTSTRUCT

Steps:
  1. Load CT volume and segmentation mask
  2. Resample to isotropic spacing (default: 1.0mm)
  3. Resize to target size (default: 256^3)
  4. Save as case_id/ct.nii.gz + case_id/seg.nii.gz

Usage:
    python scripts/preprocessing/preprocess_lung.py \
        --input-dir data/raw/Task06_Lung --output-dir data/preprocessed/train --format msd

    python scripts/preprocessing/preprocess_lung.py \
        --input-dir data/raw/my_nifti --output-dir data/preprocessed/train --format nifti

    python scripts/preprocessing/preprocess_lung.py \
        --input-dir data/raw/dicom_series --output-dir data/preprocessed/train --format dicom
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def resample_image(image, target_spacing=(1.0, 1.0, 1.0), is_mask=False):
    """Resample a SimpleITK image to target spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1024)

    return resampler.Execute(image)


def resize_image(image, target_size=(256, 256, 256), is_mask=False):
    """Resize a SimpleITK image to target size."""
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    new_spacing = [
        ospc * osz / tsz
        for ospc, osz, tsz in zip(original_spacing, original_size, target_size)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1024)

    return resampler.Execute(image)


def process_case(ct_image, seg_image, target_spacing, target_size):
    """Process a single case: resample + resize."""
    # Resample to isotropic spacing
    ct_resampled = resample_image(ct_image, target_spacing, is_mask=False)
    seg_resampled = resample_image(seg_image, target_spacing, is_mask=True)

    # Resize to target size
    if target_size is not None:
        ct_resampled = resize_image(ct_resampled, target_size, is_mask=False)
        seg_resampled = resize_image(seg_resampled, target_size, is_mask=True)

    # Cast to appropriate types
    ct_resampled = sitk.Cast(ct_resampled, sitk.sitkFloat32)
    seg_resampled = sitk.Cast(seg_resampled, sitk.sitkUInt8)

    # Binarize mask
    seg_resampled = sitk.BinaryThreshold(seg_resampled, lowerThreshold=1)

    return ct_resampled, seg_resampled


# =============================================================================
# FORMAT: NIfTI
# =============================================================================
def preprocess_nifti(input_dir, output_dir, target_spacing, target_size):
    """
    Process NIfTI format: each subdirectory has ct.nii.gz + seg.nii.gz,
    or input_dir contains paired files directly.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Try subdirectory format first
    case_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])

    if case_dirs:
        cases = []
        for case_dir in case_dirs:
            ct_candidates = ['ct.nii.gz', 'CT.nii.gz', 'image.nii.gz', 'img.nii.gz']
            seg_candidates = ['seg.nii.gz', 'mask.nii.gz', 'label.nii.gz', 'segmentation.nii.gz']

            ct_path = None
            seg_path = None
            for name in ct_candidates:
                p = case_dir / name
                if p.exists():
                    ct_path = p
                    break
            for name in seg_candidates:
                p = case_dir / name
                if p.exists():
                    seg_path = p
                    break

            if ct_path and seg_path:
                cases.append((case_dir.name, ct_path, seg_path))

        print(f"Found {len(cases)} NIfTI cases in subdirectories")
    else:
        # Try flat file pairs: xxx_ct.nii.gz + xxx_seg.nii.gz
        ct_files = sorted(input_path.glob("*_ct.nii.gz"))
        cases = []
        for ct_file in ct_files:
            case_id = ct_file.name.replace("_ct.nii.gz", "")
            seg_file = input_path / f"{case_id}_seg.nii.gz"
            if seg_file.exists():
                cases.append((case_id, ct_file, seg_file))
        print(f"Found {len(cases)} NIfTI file pairs")

    for case_id, ct_path, seg_path in tqdm(cases, desc="Processing"):
        out_dir = output_path / case_id
        if (out_dir / "ct.nii.gz").exists() and (out_dir / "seg.nii.gz").exists():
            continue

        try:
            ct_img = sitk.ReadImage(str(ct_path))
            seg_img = sitk.ReadImage(str(seg_path))

            ct_proc, seg_proc = process_case(ct_img, seg_img, target_spacing, target_size)

            out_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(ct_proc, str(out_dir / "ct.nii.gz"))
            sitk.WriteImage(seg_proc, str(out_dir / "seg.nii.gz"))
        except Exception as e:
            print(f"ERROR processing {case_id}: {e}")


# =============================================================================
# FORMAT: MSD (Medical Segmentation Decathlon)
# =============================================================================
def preprocess_msd(input_dir, output_dir, target_spacing, target_size):
    """
    Process MSD format: dataset.json + imagesTr/ + labelsTr/
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    dataset_json = input_path / "dataset.json"
    if not dataset_json.exists():
        print(f"ERROR: dataset.json not found in {input_dir}")
        sys.exit(1)

    with open(dataset_json) as f:
        dataset = json.load(f)

    training = dataset.get("training", [])
    print(f"Found {len(training)} cases in MSD dataset")

    for entry in tqdm(training, desc="Processing"):
        image_path = input_path / entry["image"].lstrip("./")
        label_path = input_path / entry["label"].lstrip("./")

        # Extract case ID from filename
        case_id = Path(entry["image"]).stem
        if case_id.endswith(".nii"):
            case_id = case_id[:-4]

        out_dir = output_path / case_id
        if (out_dir / "ct.nii.gz").exists() and (out_dir / "seg.nii.gz").exists():
            continue

        if not image_path.exists():
            print(f"WARNING: Image not found: {image_path}")
            continue
        if not label_path.exists():
            print(f"WARNING: Label not found: {label_path}")
            continue

        try:
            ct_img = sitk.ReadImage(str(image_path))
            seg_img = sitk.ReadImage(str(label_path))

            # MSD images may be 4D (with time/channel dim) - extract first volume
            if ct_img.GetDimension() == 4:
                ct_img = ct_img[:, :, :, 0]
            if seg_img.GetDimension() == 4:
                seg_img = seg_img[:, :, :, 0]

            ct_proc, seg_proc = process_case(ct_img, seg_img, target_spacing, target_size)

            out_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(ct_proc, str(out_dir / "ct.nii.gz"))
            sitk.WriteImage(seg_proc, str(out_dir / "seg.nii.gz"))
        except Exception as e:
            print(f"ERROR processing {case_id}: {e}")


# =============================================================================
# FORMAT: DICOM
# =============================================================================
def preprocess_dicom(input_dir, output_dir, target_spacing, target_size):
    """
    Process DICOM format: each subdirectory is a patient with DICOM series
    and optional RTSTRUCT file.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    case_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    print(f"Found {len(case_dirs)} DICOM case directories")

    for case_dir in tqdm(case_dirs, desc="Processing"):
        case_id = case_dir.name
        out_dir = output_path / case_id

        if (out_dir / "ct.nii.gz").exists() and (out_dir / "seg.nii.gz").exists():
            continue

        try:
            # Read DICOM series
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(str(case_dir))

            if not dicom_files:
                # Check for subdirectories containing DICOM files
                for subdir in case_dir.iterdir():
                    if subdir.is_dir():
                        dicom_files = reader.GetGDCMSeriesFileNames(str(subdir))
                        if dicom_files:
                            break

            if not dicom_files:
                print(f"WARNING: No DICOM series found in {case_dir}")
                continue

            reader.SetFileNames(dicom_files)
            ct_img = reader.Execute()

            # Look for RTSTRUCT or segmentation mask
            seg_img = None

            # Try NIfTI mask
            for seg_name in ['seg.nii.gz', 'mask.nii.gz', 'label.nii.gz']:
                seg_path = case_dir / seg_name
                if seg_path.exists():
                    seg_img = sitk.ReadImage(str(seg_path))
                    break

            if seg_img is None:
                print(f"WARNING: No segmentation found for {case_id}, skipping")
                continue

            ct_proc, seg_proc = process_case(ct_img, seg_img, target_spacing, target_size)

            out_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(ct_proc, str(out_dir / "ct.nii.gz"))
            sitk.WriteImage(seg_proc, str(out_dir / "seg.nii.gz"))
        except Exception as e:
            print(f"ERROR processing {case_id}: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Preprocess lung CT data for segmentation")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Input directory with raw data")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for preprocessed data")
    parser.add_argument("--format", type=str, choices=["nifti", "msd", "dicom"],
                        default="nifti", help="Input data format (default: nifti)")
    parser.add_argument("--target-spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Target isotropic spacing in mm (default: 1.0 1.0 1.0)")
    parser.add_argument("--target-size", type=int, nargs=3, default=[256, 256, 256],
                        help="Target volume size (default: 256 256 256). Use 0 0 0 to skip resize.")
    args = parser.parse_args()

    target_spacing = tuple(args.target_spacing)
    target_size = tuple(args.target_size)
    if target_size == (0, 0, 0):
        target_size = None

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Input:   {args.input_dir}")
    print(f"Output:  {args.output_dir}")
    print(f"Format:  {args.format}")
    print(f"Spacing: {target_spacing}")
    print(f"Size:    {target_size}")

    if args.format == "nifti":
        preprocess_nifti(args.input_dir, args.output_dir, target_spacing, target_size)
    elif args.format == "msd":
        preprocess_msd(args.input_dir, args.output_dir, target_spacing, target_size)
    elif args.format == "dicom":
        preprocess_dicom(args.input_dir, args.output_dir, target_spacing, target_size)

    # Print summary
    output_cases = sorted([d for d in output_path.iterdir() if d.is_dir()])
    print(f"\nPreprocessing complete: {len(output_cases)} cases in {args.output_dir}")


if __name__ == "__main__":
    main()
