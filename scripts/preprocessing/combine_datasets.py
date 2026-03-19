"""
Combine multiple lung CT segmentation datasets into a unified format.

Handles:
  1. MSD Task06_Lung (already preprocessed)
  2. COVID-19 CT Seg Benchmark (Zenodo, NIfTI)
  3. NSCLC-Radiomics (TCIA, DICOM CT + DICOM SEG)
  4. RIDER Lung CT (TCIA, DICOM CT + DICOM SEG)

All datasets are converted to:
    data/preprocessed/combined/
        <dataset>_<case_id>/
            ct.nii.gz       # Float32, resampled to 1mm isotropic, 256^3
            seg.nii.gz      # Binary mask (uint8, 0/1)

Usage:
    python scripts/preprocessing/combine_datasets.py
    python scripts/preprocessing/combine_datasets.py --skip-download
    python scripts/preprocessing/combine_datasets.py --dataset nsclc  # single dataset
"""

import argparse
import gc
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "preprocessing"))
from preprocess_lung import resample_image, resize_image, process_case

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "preprocessed" / "combined"
TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_SIZE = (256, 256, 256)


def case_exists(output_dir, case_id):
    d = output_dir / case_id
    return (d / "ct.nii.gz").exists() and (d / "seg.nii.gz").exists()


def save_case(ct_sitk, seg_sitk, output_dir, case_id):
    """Preprocess and save a single case."""
    try:
        ct_proc, seg_proc = process_case(ct_sitk, seg_sitk, TARGET_SPACING, TARGET_SIZE)
        out_dir = output_dir / case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(ct_proc, str(out_dir / "ct.nii.gz"))
        sitk.WriteImage(seg_proc, str(out_dir / "seg.nii.gz"))
        return True
    except Exception as e:
        print(f"  ERROR {case_id}: {e}")
        return False


# =============================================================================
# 1. MSD Task06_Lung (copy already preprocessed)
# =============================================================================
def process_msd(output_dir):
    """Copy already-preprocessed MSD cases."""
    src = ROOT / "data" / "preprocessed" / "train"
    if not src.exists():
        print("  MSD: Not found, skipping")
        return 0

    cases = sorted([d for d in src.iterdir() if d.is_dir()])
    count = 0
    for case_dir in tqdm(cases, desc="MSD Task06"):
        case_id = f"msd_{case_dir.name}"
        if case_exists(output_dir, case_id):
            count += 1
            continue
        out = output_dir / case_id
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(case_dir / "ct.nii.gz", out / "ct.nii.gz")
        shutil.copy2(case_dir / "seg.nii.gz", out / "seg.nii.gz")
        count += 1
    return count


# =============================================================================
# 2. COVID-19 CT Seg Benchmark
# =============================================================================
def process_covid(output_dir):
    """Process COVID-19 CT Seg Benchmark (infection masks as lesion proxy)."""
    covid_dir = RAW_DIR / "covid19_ct_seg"

    # Extract zips if needed
    for zf_name in ["COVID-19-CT-Seg_20cases.zip", "Infection_Mask.zip"]:
        zf_path = covid_dir / zf_name
        if zf_path.exists():
            extract_dir = covid_dir / zf_name.replace(".zip", "")
            if not extract_dir.exists():
                print(f"  Extracting {zf_name}...")
                with zipfile.ZipFile(str(zf_path), "r") as z:
                    z.extractall(str(covid_dir))

    # Find CT volumes and infection masks
    ct_dir = None
    mask_dir = None

    # The zip may extract with different structures
    for candidate in [
        covid_dir / "COVID-19-CT-Seg_20cases",
        covid_dir,
    ]:
        nii_files = list(candidate.glob("*.nii.gz"))
        if len(nii_files) >= 20:
            ct_dir = candidate
            break

    for candidate in [
        covid_dir / "infection_masks",
        covid_dir / "Infection_Mask",
        covid_dir / "infection_mask",
    ]:
        if candidate.exists() and list(candidate.glob("*.nii.gz")):
            mask_dir = candidate
            break

    # If masks extracted flat, extract into subfolder
    if mask_dir is None:
        mask_subdir = covid_dir / "infection_masks"
        mask_zip = covid_dir / "Infection_Mask.zip"
        if mask_zip.exists():
            mask_subdir.mkdir(exist_ok=True)
            with zipfile.ZipFile(str(mask_zip), "r") as z:
                z.extractall(str(mask_subdir))
            if list(mask_subdir.glob("*.nii.gz")):
                mask_dir = mask_subdir

    if ct_dir is None or mask_dir is None:
        # Try flat structure - all in covid_dir
        ct_files = sorted(covid_dir.glob("coronacases_*.nii.gz")) + sorted(covid_dir.glob("radiopaedia_*.nii.gz"))
        if not ct_files:
            print("  COVID-19: CT volumes not found, skipping")
            return 0
        ct_dir = covid_dir

    ct_files = sorted([f for f in ct_dir.glob("*.nii.gz")
                       if not f.name.startswith(".")])
    mask_files = sorted([f for f in mask_dir.glob("*.nii.gz")]) if mask_dir else []

    # Match by filename
    ct_by_name = {f.stem.replace(".nii", ""): f for f in ct_files}
    mask_by_name = {f.stem.replace(".nii", ""): f for f in mask_files}

    count = 0
    for name, ct_path in tqdm(ct_by_name.items(), desc="COVID-19"):
        mask_path = mask_by_name.get(name)
        if mask_path is None:
            continue

        case_id = f"covid_{name}"
        if case_exists(output_dir, case_id):
            count += 1
            continue

        try:
            ct_img = sitk.ReadImage(str(ct_path))
            seg_img = sitk.ReadImage(str(mask_path))

            # Binarize infection mask (any label > 0 = lesion)
            seg_arr = sitk.GetArrayFromImage(seg_img)
            seg_binary = (seg_arr > 0).astype(np.uint8)
            seg_bin_img = sitk.GetImageFromArray(seg_binary)
            seg_bin_img.CopyInformation(seg_img)

            if save_case(ct_img, seg_bin_img, output_dir, case_id):
                count += 1
        except Exception as e:
            print(f"  ERROR {case_id}: {e}")

    return count


# =============================================================================
# 3. NSCLC-Radiomics (DICOM CT + DICOM SEG)
# =============================================================================
def dicom_seg_to_sitk(seg_dcm_path, ct_image):
    """Convert a DICOM SEG file to a SimpleITK binary mask aligned to CT."""
    try:
        import pydicom
        dcm = pydicom.dcmread(str(seg_dcm_path))

        # Try pydicom-seg first
        try:
            import pydicom_seg
            reader = pydicom_seg.SegmentReader()
            result = reader.read(dcm)
            # Get first segment
            for seg_num in result.available_segments:
                seg_image = result.segment_image(seg_num)
                arr = sitk.GetArrayFromImage(seg_image)
                if arr.sum() > 0:
                    return seg_image
        except Exception:
            pass

        # Fallback: manual DICOM SEG extraction
        if hasattr(dcm, 'PixelData') and hasattr(dcm, 'NumberOfFrames'):
            n_frames = int(dcm.NumberOfFrames)
            rows = int(dcm.Rows)
            cols = int(dcm.Columns)

            pixel_data = dcm.pixel_array
            if pixel_data.ndim == 2:
                pixel_data = pixel_data.reshape(n_frames, rows, cols)

            # Create mask volume matching CT
            ct_arr = sitk.GetArrayFromImage(ct_image)
            mask = np.zeros(ct_arr.shape, dtype=np.uint8)

            # Map frames to CT slices using PerFrameFunctionalGroupsSequence
            if hasattr(dcm, 'PerFrameFunctionalGroupsSequence'):
                for i, frame_fg in enumerate(dcm.PerFrameFunctionalGroupsSequence):
                    if i >= n_frames:
                        break
                    try:
                        plane_pos = frame_fg.PlanePositionSequence[0]
                        pos = [float(x) for x in plane_pos.ImagePositionPatient]
                        # Find matching CT slice
                        ct_origin = ct_image.GetOrigin()
                        ct_spacing = ct_image.GetSpacing()
                        slice_idx = int(round((pos[2] - ct_origin[2]) / ct_spacing[2]))
                        if 0 <= slice_idx < mask.shape[0]:
                            frame_data = pixel_data[i]
                            if frame_data.shape == (rows, cols):
                                mask[slice_idx] = np.maximum(
                                    mask[slice_idx],
                                    (frame_data > 0).astype(np.uint8)
                                )
                    except Exception:
                        continue

            if mask.sum() > 0:
                mask_img = sitk.GetImageFromArray(mask)
                mask_img.CopyInformation(ct_image)
                return mask_img

    except Exception as e:
        pass

    return None


def process_nsclc_radiomics(output_dir):
    """Process NSCLC-Radiomics DICOM CT + SEG."""
    dicom_dir = RAW_DIR / "nsclc_radiomics" / "dicom"
    manifest_path = RAW_DIR / "nsclc_radiomics" / "manifest.csv"

    if not dicom_dir.exists() or not manifest_path.exists():
        print("  NSCLC-Radiomics: Not downloaded yet, skipping")
        return 0

    import pandas as pd
    manifest = pd.read_csv(manifest_path)

    count = 0
    errors = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="NSCLC-Radiomics"):
        patient_id = row["PatientID"]
        case_id = f"nsclc_{patient_id}"

        if case_exists(output_dir, case_id):
            count += 1
            continue

        ct_uid = row["CT_SeriesUID"]
        seg_uid = row["SEG_SeriesUID"]

        # Find DICOM directories
        ct_dcm_dir = dicom_dir / ct_uid
        seg_dcm_dir = dicom_dir / seg_uid

        if not ct_dcm_dir.exists() or not seg_dcm_dir.exists():
            continue

        try:
            # Read CT
            reader = sitk.ImageSeriesReader()
            ct_files = reader.GetGDCMSeriesFileNames(str(ct_dcm_dir))
            if not ct_files:
                continue
            reader.SetFileNames(ct_files)
            ct_img = reader.Execute()

            # Read SEG
            seg_files = list(seg_dcm_dir.glob("*.dcm"))
            if not seg_files:
                seg_files = list(seg_dcm_dir.glob("*"))
                seg_files = [f for f in seg_files if f.is_file()]

            seg_img = None
            for seg_file in seg_files:
                seg_img = dicom_seg_to_sitk(seg_file, ct_img)
                if seg_img is not None:
                    break

            if seg_img is None:
                # Try reading as regular image
                try:
                    seg_img = sitk.ReadImage(str(seg_files[0]))
                except Exception:
                    errors += 1
                    continue

            if save_case(ct_img, seg_img, output_dir, case_id):
                count += 1

        except Exception as e:
            errors += 1

        gc.collect()

    if errors > 0:
        print(f"  NSCLC-Radiomics: {errors} conversion errors")
    return count


# =============================================================================
# 4. RIDER Lung CT (DICOM CT + DICOM SEG)
# =============================================================================
def process_rider(output_dir):
    """Process RIDER Lung CT DICOM CT + SEG."""
    dicom_dir = RAW_DIR / "rider_lung_ct" / "dicom"
    manifest_path = RAW_DIR / "rider_lung_ct" / "manifest.csv"

    if not dicom_dir.exists() or not manifest_path.exists():
        print("  RIDER Lung CT: Not downloaded yet, skipping")
        return 0

    import pandas as pd
    manifest = pd.read_csv(manifest_path)

    count = 0
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="RIDER"):
        patient_id = row["PatientID"]
        case_id = f"rider_{patient_id}"

        if case_exists(output_dir, case_id):
            count += 1
            continue

        ct_uid = row["CT_SeriesUID"]
        seg_uid = row["SEG_SeriesUID"]

        ct_dcm_dir = dicom_dir / ct_uid
        seg_dcm_dir = dicom_dir / seg_uid

        if not ct_dcm_dir.exists() or not seg_dcm_dir.exists():
            continue

        try:
            reader = sitk.ImageSeriesReader()
            ct_files = reader.GetGDCMSeriesFileNames(str(ct_dcm_dir))
            if not ct_files:
                continue
            reader.SetFileNames(ct_files)
            ct_img = reader.Execute()

            seg_files = list(seg_dcm_dir.glob("*.dcm")) + list(seg_dcm_dir.glob("*"))
            seg_files = [f for f in seg_files if f.is_file()]

            seg_img = None
            for seg_file in seg_files:
                seg_img = dicom_seg_to_sitk(seg_file, ct_img)
                if seg_img is not None:
                    break

            if seg_img is None:
                continue

            if save_case(ct_img, seg_img, output_dir, case_id):
                count += 1

        except Exception as e:
            print(f"  ERROR {case_id}: {e}")

        gc.collect()

    return count


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Combine lung CT segmentation datasets")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "msd", "covid", "nsclc", "rider", "interobserver"],
                        help="Which dataset to process")
    parser.add_argument("--target-size", type=int, nargs=3, default=[256, 256, 256])
    args = parser.parse_args()

    global TARGET_SIZE
    TARGET_SIZE = tuple(args.target_size)

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {output_dir}")
    print(f"Target size: {TARGET_SIZE}")
    print()

    def process_interobserver(output_dir):
        """Process NSCLC-Radiomics-Interobserver1 (same format as NSCLC-Radiomics)."""
        dicom_dir = RAW_DIR / "nsclc_interobserver" / "dicom"
        manifest_path = RAW_DIR / "nsclc_interobserver" / "manifest.csv"
        if not dicom_dir.exists() or not manifest_path.exists():
            print("  NSCLC-Interobserver1: Not downloaded yet, skipping")
            return 0
        import pandas as pd
        manifest = pd.read_csv(manifest_path)
        count = 0
        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Interobserver"):
            patient_id = row["PatientID"]
            case_id = f"interobs_{patient_id}"
            if case_exists(output_dir, case_id):
                count += 1
                continue
            ct_dcm_dir = dicom_dir / row["CT_SeriesUID"]
            seg_dcm_dir = dicom_dir / row["SEG_SeriesUID"]
            if not ct_dcm_dir.exists() or not seg_dcm_dir.exists():
                continue
            try:
                reader = sitk.ImageSeriesReader()
                ct_files = reader.GetGDCMSeriesFileNames(str(ct_dcm_dir))
                if not ct_files:
                    continue
                reader.SetFileNames(ct_files)
                ct_img = reader.Execute()
                seg_files = [f for f in seg_dcm_dir.iterdir() if f.is_file()]
                seg_img = None
                for sf in seg_files:
                    seg_img = dicom_seg_to_sitk(sf, ct_img)
                    if seg_img is not None:
                        break
                if seg_img is None:
                    continue
                if save_case(ct_img, seg_img, output_dir, case_id):
                    count += 1
            except Exception as e:
                print(f"  ERROR {case_id}: {e}")
            gc.collect()
        return count

    processors = {
        "msd": ("MSD Task06_Lung", process_msd),
        "covid": ("COVID-19 CT Seg", process_covid),
        "nsclc": ("NSCLC-Radiomics", process_nsclc_radiomics),
        "rider": ("RIDER Lung CT", process_rider),
        "interobserver": ("NSCLC-Interobserver1", process_interobserver),
    }

    total = 0
    if args.dataset == "all":
        order = ["msd", "covid", "nsclc", "rider", "interobserver"]
    else:
        order = [args.dataset]

    for key in order:
        name, func = processors[key]
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")
        n = func(output_dir)
        print(f"  >> {n} cases")
        total += n

    # Summary
    all_cases = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
    dataset_counts = {}
    for c in all_cases:
        prefix = c.split("_")[0]
        dataset_counts[prefix] = dataset_counts.get(prefix, 0) + 1

    print(f"\n{'='*60}")
    print(f"COMBINED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total cases: {len(all_cases)}")
    for prefix, count in sorted(dataset_counts.items()):
        print(f"  {prefix}: {count}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
