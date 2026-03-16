# Data Directory

Place your lung CT segmentation datasets here.

## Expected Format (after preprocessing)

```
data/preprocessed/train/
  case_001/
    ct.nii.gz       # Float32 CT volume, resampled to isotropic spacing
    seg.nii.gz      # Binary segmentation mask, uint8, 0/1
  case_002/
    ct.nii.gz
    seg.nii.gz
  ...
```

## Supported Raw Formats

The preprocessing script (`scripts/preprocessing/preprocess_lung.py`) supports:

1. **NIfTI pairs**: Directory with `ct.nii.gz` + `seg.nii.gz` per case
2. **MSD format**: Medical Segmentation Decathlon JSON + imagesTr/labelsTr
3. **DICOM + RTSTRUCT**: DICOM series with RT structure sets

## Recommended Datasets

- **MSD Task06 (Lung)**: ~63 cases with lung tumor annotations
- **NSCLC-Radiomics**: Larger dataset with CT + contours
- **LCTSC**: Lung CT Segmentation Challenge

## Preprocessing

```bash
# MSD format
python scripts/preprocessing/preprocess_lung.py \
    --input-dir data/raw/Task06_Lung \
    --output-dir data/preprocessed/train \
    --format msd

# NIfTI pairs
python scripts/preprocessing/preprocess_lung.py \
    --input-dir data/raw/my_dataset \
    --output-dir data/preprocessed/train \
    --format nifti
```
