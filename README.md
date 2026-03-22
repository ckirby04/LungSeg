# LungSeg

Automated lung tumor segmentation from CT scans using a stacking ensemble of multi-scale 3D U-Nets and nnU-Net. The system trains six base models — four custom `LightweightUNet3D` at different patch scales (8, 12, 24, 36) plus nnU-Net 3D and 2D — and fuses their predictions with a lightweight 3D CNN meta-learner.

Trained on **504 tumor-only cases** from four public datasets (MSD Task06, NSCLC-Radiomics, RIDER Lung CT, NSCLC-Interobserver).

## Architecture

```
                           CT Volume (1-channel)
                                 |
         +----------+----------+-+----------+----------+
         |          |          |            |           |
     8-patch    12-patch   24-patch    36-patch     nnU-Net
      U-Net      U-Net      U-Net      U-Net     3D + 2D
         |          |          |            |           |
         +----------+----------+------------+----------+
                                 |
                       Stacking Features
                (6 preds + variance + range = 8ch)
                                 |
                       Stacking Classifier
                        (3D CNN, ~25K params)
                                 |
                       Final Segmentation Mask
```

**Custom Base Models** — `LightweightUNet3D` with attention gates and residual connections (~85K params each). Each trained at a different patch scale to capture fine-to-coarse tumor features.

**nnU-Net Models** — Auto-configured 3D full-resolution and 2D U-Nets (nnU-Net v2) trained for 1000 epochs with automatic preprocessing, architecture search, and data augmentation.

**Stacking Classifier** — 3D CNN meta-learner with residual blocks. Takes 8-channel input (6 base predictions + inter-model variance + prediction range) and outputs the fused segmentation.

## Results

Trained on 504 tumor-only cases, evaluated on 75 held-out cases:

| Method | Dice | Sensitivity | Precision |
|--------|------|-------------|-----------|
| **Stacking (6 models)** | **0.732** | **0.793** | **0.749** |
| Simple average | 0.562 | 0.652 | 0.582 |
| 36-patch alone | 0.553 | 0.733 | 0.513 |
| 24-patch alone | 0.501 | 0.562 | 0.532 |

- **Median Dice: 0.849** (83% of cases above 0.7)
- **Top cases: 0.93-0.96 Dice**
- Best custom base model: 24-patch at 0.882 patch-level Dice (epoch 627/1000)

## Dataset

Combined from four public sources (tumor annotations only, COVID infection masks excluded):

| Dataset | Cases | Source |
|---------|-------|--------|
| [MSD Task06_Lung](http://medicaldecathlon.com/) | 63 | Medical Segmentation Decathlon |
| [NSCLC-Radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) | 408 | TCIA (DICOM SEG) |
| [RIDER Lung CT](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=46334165) | 31 | TCIA (DICOM SEG) |
| [NSCLC-Interobserver1](https://www.cancerimagingarchive.net/collection/nsclc-radiomics-interobserver1/) | 2 | TCIA (DICOM SEG) |
| **Total** | **504** | |

All cases are preprocessed to 256^3 volumes at ~1mm isotropic spacing with binary tumor masks.

## Project Structure

```
LungSeg/
├── configs/
│   └── models.yaml                     # Model registry & ensemble config
├── data/
│   ├── preprocessed/
│   │   ├── combined/                   # All datasets merged (524 cases)
│   │   ├── tumor_only/                 # Tumor-only subset (504 cases)
│   │   └── train/                      # MSD-only (legacy)
│   └── raw/                            # Raw downloads (DICOM, NIfTI, zips)
├── model/
│   ├── base_models/                    # Custom U-Net checkpoints (.pth + state.json)
│   ├── stacking_cache/                 # Cached base model predictions (.npz)
│   ├── stacking_classifier.pth         # Trained meta-learner
│   └── stacking_results.json           # Per-method evaluation
├── nnUNet/
│   ├── nnUNet_raw/                     # nnU-Net dataset format
│   ├── nnUNet_preprocessed/            # nnU-Net auto-preprocessed data
│   └── nnUNet_results/                 # nnU-Net trained models
├── scripts/
│   ├── preprocessing/
│   │   ├── preprocess_lung.py          # Raw data preprocessing (MSD, NIfTI, DICOM)
│   │   ├── combine_datasets.py         # Multi-dataset download & conversion
│   │   └── download_tcia.py            # Resumable TCIA downloader
│   ├── training/
│   │   ├── train_full_pipeline.py      # Full pipeline (custom + nnU-Net + stacking)
│   │   ├── train_base_model.py         # Train individual custom base models
│   │   ├── train_stacking.py           # Train stacking classifier
│   │   ├── setup_nnunet.py             # Convert data to nnU-Net format
│   │   ├── restacking_with_nnunet.py   # Re-run stacking with nnU-Net predictions
│   │   └── train_tumor_only.py         # Tumor-only training (no COVID)
│   ├── evaluation/
│   │   └── evaluate.py                 # Full evaluation pipeline
│   └── inference/
│       └── run_inference.py            # Inference on new CT volumes
├── src/segmentation/
│   ├── unet.py                         # LightweightUNet3D architecture
│   ├── enhanced_unet.py                # Deep supervised U-Net variants
│   ├── dataset.py                      # LungCTDataset loader
│   ├── advanced_losses.py              # SmallLesionOptimizedLoss + others
│   ├── augmentation.py                 # MONAI augmentation pipeline
│   ├── stacking.py                     # Stacking classifier & ensemble logic
│   ├── inference.py                    # Inference utilities
│   ├── tta.py                          # Test-time augmentation
│   ├── postprocessing.py               # Morphological post-processing
│   └── weighted_sampling.py            # Weighted sampling for imbalanced data
├── results/                            # Evaluation outputs
├── run_pipeline.py                     # Simple pipeline (MSD-only, legacy)
├── requirements.txt
└── pyproject.toml
```

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/ckirby04/LungSeg.git
cd LungSeg
pip install -r requirements.txt
pip install nnunetv2
```

Requires Python 3.9+ and CUDA-capable GPU(s). Key dependencies:

| Package | Purpose |
|---------|---------|
| PyTorch | Deep learning framework |
| nnU-Net v2 | State-of-the-art medical segmentation |
| MONAI | Medical imaging transforms & augmentation |
| nibabel / SimpleITK | NIfTI and DICOM I/O |
| tcia-utils | TCIA dataset downloads |

### 2. Download & Prepare Data

```bash
# Download all public datasets (NSCLC-Radiomics, RIDER, COVID-19, MSD)
python scripts/preprocessing/combine_datasets.py

# Or download TCIA datasets separately (resumable)
python scripts/preprocessing/download_tcia.py --dataset all
python scripts/preprocessing/combine_datasets.py
```

### 3. Run the Full Training Pipeline

```bash
# Full BrainMetScan-style pipeline:
# 4 custom U-Nets (1000 ep) → nnU-Net 3D + 2D (1000 ep) → stacking → eval
python scripts/training/train_full_pipeline.py
```

Skip phases if models are already trained:
```bash
python scripts/training/train_full_pipeline.py --skip-custom     # nnU-Net + stacking only
python scripts/training/train_full_pipeline.py --skip-nnunet      # custom + stacking only
python scripts/training/train_full_pipeline.py --skip-base        # stacking + eval only
```

### 4. Re-run Stacking with nnU-Net

After all base models are trained, fuse all 6 models into the stacking classifier:

```bash
python scripts/training/restacking_with_nnunet.py
```

### 5. Inference

```bash
python scripts/inference/run_inference.py \
    --input path/to/ct_scan.nii.gz \
    --output path/to/prediction.nii.gz
```

## Training Details

### Custom Base Models (4x LightweightUNet3D)

| Model | Patch Size | Best Dice | Best Epoch | Sensitivity |
|-------|-----------|-----------|------------|-------------|
| 8-patch | 8^3 | 0.863 | 637 | 0.897 |
| 12-patch | 12^3 | 0.877 | 587 | 0.930 |
| 24-patch | 24^3 | 0.882 | 627 | 0.921 |
| 36-patch | 36^3 | 0.880 | 605 | 0.930 |

All trained for 1000 epochs with:
- **SmallLesionOptimizedLoss** — Tversky (60%) + Focal (25%) + Sensitivity (15%)
- **MONAI augmentation** — flips, rotations, affine, intensity perturbations
- **Cosine annealing** with warm restarts
- **10 patches/volume**, learning rate 0.0005

### nnU-Net Models

Auto-configured by nnU-Net v2 experiment planner:

| Config | Patch Size | Batch | Architecture |
|--------|-----------|-------|-------------|
| 3D fullres | 128^3 | 2 | 6-stage PlainConvUNet (32-320 features) |
| 2D | 256x256 | 49 | 7-stage PlainConvUNet (32-512 features) |

Both trained for 1000 epochs with nnU-Net default training (Dice+CE loss, SGD, poly LR).

### GPU Requirements

Trained on 2x NVIDIA RTX 3070 Ti (8GB each):

| Phase | GPU 0 | GPU 1 | Time |
|-------|-------|-------|------|
| Custom batch 1 | 8-patch | 24-patch | ~8h |
| Custom batch 2 | 12-patch | 36-patch | ~8h |
| nnU-Net 3D | 3D fullres | — | ~24h |
| nnU-Net 2D | — | 2D | ~24h |
| Stacking | Meta-learner | — | ~4h |

### Data Format

After preprocessing, each case is stored as:

```
data/preprocessed/combined/
  nsclc_LUNG1-001/
    ct.nii.gz       # Float32, ~1mm isotropic, 256x256x256
    seg.nii.gz      # Binary mask (uint8), 0 = background, 1 = tumor
```

CT values windowed to [-1000, 400] HU and normalized to [0, 1] during training.

## License

This project is for research and educational purposes.

## Acknowledgments

- [Medical Segmentation Decathlon](http://medicaldecathlon.com/) for Task06_Lung
- [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/) for NSCLC-Radiomics, RIDER, and Interobserver datasets
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for auto-configured segmentation
- [MONAI](https://monai.io/) for medical imaging transforms
- [PyTorch](https://pytorch.org/) deep learning framework
