# Data Directory

## Combined Dataset (504 tumor-only cases)

Built from four public sources using `scripts/preprocessing/combine_datasets.py`:

| Dataset | Prefix | Cases | Format |
|---------|--------|-------|--------|
| MSD Task06_Lung | `msd_` | 63 | NIfTI (direct) |
| NSCLC-Radiomics | `nsclc_` | 408 | DICOM CT + SEG (TCIA) |
| RIDER Lung CT | `rider_` | 31 | DICOM CT + SEG (TCIA) |
| NSCLC-Interobserver1 | `interobs_` | 2 | DICOM CT + SEG (TCIA) |
| COVID-19 CT Seg | `covid_` | 20 | NIfTI (Zenodo) — excluded from training |

COVID cases are excluded from the `tumor_only/` directory since they contain infection masks, not tumor annotations.

## Directory Structure

```
data/
├── raw/                          # Original downloads (can be deleted after preprocessing)
│   ├── Task06_Lung/              # MSD dataset (NIfTI)
│   ├── covid19_ct_seg/           # COVID-19 CT Seg (Zenodo zips)
│   ├── nsclc_radiomics/          # NSCLC-Radiomics manifest
│   ├── rider_lung_ct/            # RIDER Lung CT manifest
│   └── nsclc_interobserver/      # NSCLC-Interobserver1 manifest
├── preprocessed/
│   ├── combined/                 # All datasets merged (524 cases)
│   ├── tumor_only/               # Symlinks to combined/, excluding COVID (504 cases)
│   └── train/                    # MSD-only preprocessing (legacy, 63 cases)
└── README.md
```

## Preprocessed Format

Each case directory contains:

```
<dataset>_<case_id>/
    ct.nii.gz       # Float32, resampled to ~1mm isotropic, 256x256x256
    seg.nii.gz      # Binary segmentation mask (uint8), 0 = background, 1 = tumor
```

## Downloading Data

```bash
# Download all datasets (TCIA + Zenodo + MSD)
python scripts/preprocessing/combine_datasets.py

# Download only TCIA datasets (resumable)
python scripts/preprocessing/download_tcia.py --dataset all

# Process a specific dataset
python scripts/preprocessing/combine_datasets.py --dataset nsclc
```

## Supported Raw Formats

The preprocessing pipeline supports:
1. **NIfTI pairs** — `ct.nii.gz` + `seg.nii.gz` per case
2. **MSD format** — Medical Segmentation Decathlon JSON + imagesTr/labelsTr
3. **DICOM + SEG** — DICOM CT series with DICOM Segmentation objects (TCIA)
4. **DICOM + RTSTRUCT** — DICOM CT with RT structure sets
