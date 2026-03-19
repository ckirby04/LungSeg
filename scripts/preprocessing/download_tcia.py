"""
Resumable TCIA dataset downloader for NSCLC-Radiomics and RIDER Lung CT.

Downloads DICOM CT + SEG series, skipping already downloaded series.
Handles connection errors gracefully with retries.

Usage:
    python scripts/preprocessing/download_tcia.py --dataset nsclc
    python scripts/preprocessing/download_tcia.py --dataset rider
    python scripts/preprocessing/download_tcia.py --dataset all
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tcia_utils import nbia

ROOT = Path(__file__).resolve().parent.parent.parent


def get_downloaded_series(dicom_dir):
    """Get set of already-downloaded series UIDs."""
    if not dicom_dir.exists():
        return set()
    return {d.name for d in dicom_dir.iterdir() if d.is_dir()}


def download_dataset(name, manifest_path, dicom_dir, batch_size=10, max_retries=3):
    """Download a TCIA dataset with resume support."""
    manifest = pd.read_csv(manifest_path)

    # Build full series list
    all_uids = []
    for _, row in manifest.iterrows():
        all_uids.append(row['CT_SeriesUID'])
        all_uids.append(row['SEG_SeriesUID'])

    # Filter out already downloaded
    downloaded = get_downloaded_series(dicom_dir)
    remaining = [uid for uid in all_uids if uid not in downloaded]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Total series: {len(all_uids)}")
    print(f"  Already downloaded: {len(all_uids) - len(remaining)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"{'='*60}\n")

    if not remaining:
        print("  All series already downloaded!")
        return

    # Download in batches
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]
        series_list = [{'SeriesInstanceUID': uid} for uid in batch]

        for attempt in range(max_retries):
            try:
                print(f"  Batch {i//batch_size + 1}/{(len(remaining) + batch_size - 1)//batch_size} "
                      f"({len(batch)} series)...")
                nbia.downloadSeries(
                    series_data=series_list,
                    path=str(dicom_dir),
                    number=5,
                )
                break
            except Exception as e:
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(5)

        # Progress update
        now_downloaded = len(get_downloaded_series(dicom_dir))
        print(f"  Progress: {now_downloaded}/{len(all_uids)} series\n")

    final = len(get_downloaded_series(dicom_dir))
    print(f"\n  {name} complete: {final}/{len(all_uids)} series downloaded")


def main():
    parser = argparse.ArgumentParser(description="Download TCIA datasets")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "nsclc", "rider"])
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    datasets = {
        "nsclc": (
            "NSCLC-Radiomics (422 patients)",
            ROOT / "data" / "raw" / "nsclc_radiomics" / "manifest.csv",
            ROOT / "data" / "raw" / "nsclc_radiomics" / "dicom",
        ),
        "rider": (
            "RIDER Lung CT (32 patients)",
            ROOT / "data" / "raw" / "rider_lung_ct" / "manifest.csv",
            ROOT / "data" / "raw" / "rider_lung_ct" / "dicom",
        ),
    }

    if args.dataset == "all":
        order = ["nsclc", "rider"]
    else:
        order = [args.dataset]

    for key in order:
        name, manifest_path, dicom_dir = datasets[key]
        if not manifest_path.exists():
            print(f"  {key}: manifest not found, skipping")
            continue
        download_dataset(name, manifest_path, dicom_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
