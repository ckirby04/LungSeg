# LungSeg

Automated lung tumor segmentation from CT scans using a stacking ensemble of 3D U-Nets. The system trains four base models at different patch scales (8, 12, 24, 36) and fuses their predictions with a lightweight CNN meta-learner, achieving robust segmentation across lesion sizes.

## Architecture

```
                        CT Volume (1-channel)
                              |
            +---------+-------+-------+---------+
            |         |               |         |
        8-patch   12-patch       24-patch   36-patch
        U-Net      U-Net          U-Net      U-Net
            |         |               |         |
            +---------+-------+-------+---------+
                              |
                    Stacking Features
                  (4 preds + variance + range)
                              |
                    Stacking Classifier
                     (3D CNN, ~25K params)
                              |
                    Final Segmentation Mask
```

**Base Models** — `LightweightUNet3D` with attention gates and residual connections (single-channel input, ~85K params each).

**Stacking Classifier** — 3D CNN meta-learner with residual blocks that takes 6-channel input (4 base predictions + inter-model variance + prediction range) and outputs the final fused segmentation.

## Project Structure

```
LungSeg/
├── configs/
│   └── models.yaml                # Model registry & ensemble configuration
├── data/
│   └── README.md                  # Data format documentation
├── model/
│   └── base_models/               # Trained model checkpoints
├── scripts/
│   ├── evaluation/
│   │   └── evaluate.py            # Full evaluation pipeline
│   ├── inference/
│   │   └── run_inference.py       # End-to-end inference on new CT volumes
│   ├── preprocessing/
│   │   └── preprocess_lung.py     # Data preprocessing (MSD, NIfTI, DICOM)
│   └── training/
│       ├── train_base_model.py    # Train individual base models
│       ├── train_stacking.py      # Train stacking classifier
│       └── overnight.py           # Full multi-phase training pipeline
├── src/segmentation/
│   ├── unet.py                    # LightweightUNet3D architecture
│   ├── enhanced_unet.py           # DeepSupervisedUNet3D, HybridUNet3D variants
│   ├── dataset.py                 # LungCTDataset loader
│   ├── advanced_losses.py         # SmallLesionOptimizedLoss + others
│   ├── augmentation.py            # MONAI augmentation pipeline
│   ├── stacking.py                # Stacking classifier & ensemble logic
│   ├── inference.py               # Inference utilities
│   ├── tta.py                     # Test-time augmentation
│   ├── postprocessing.py          # Morphological post-processing
│   └── weighted_sampling.py       # Weighted sampling for imbalanced data
├── results/                       # Evaluation outputs
├── run_pipeline.py                # Unified preprocess → train → eval script
├── requirements.txt
└── pyproject.toml
```

## Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/ckirby04/LungSeg.git
cd LungSeg
pip install -r requirements.txt
```

Requires Python 3.9+ and a CUDA-capable GPU. Key dependencies:

| Package | Purpose |
|---------|---------|
| PyTorch | Deep learning framework |
| MONAI | Medical imaging transforms & augmentation |
| nibabel | NIfTI file I/O |
| SimpleITK | Image resampling & preprocessing |
| scipy | Scientific computing & morphology |

### 2. Get Data

Download the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) Task06_Lung dataset (~63 cases, 9.2 GB):

```bash
# Place the tar file in data/raw/
mkdir -p data/raw
# Download Task06_Lung.tar into data/raw/
```

Other supported datasets:
- **NSCLC-Radiomics** — larger dataset with CT + contours
- **LCTSC** — Lung CT Segmentation Challenge

### 3. Run the Full Pipeline

```bash
# Full pipeline: extract → preprocess → train → evaluate
python run_pipeline.py

# Quick test run (20 epochs) to validate setup
python run_pipeline.py --quick

# Custom configuration
python run_pipeline.py --epochs 100 --patches-per-volume 8
```

## Usage

### Pipeline Steps

The `run_pipeline.py` script supports running individual steps:

```bash
python run_pipeline.py --step preprocess       # Only preprocess raw data
python run_pipeline.py --step train-base       # Only train base models
python run_pipeline.py --step train-stacking   # Only train stacking classifier
python run_pipeline.py --step train            # Train base + stacking
python run_pipeline.py --step eval             # Only evaluate
```

### Preprocessing

Converts raw CT data to standardized format (isotropic 1mm spacing, 256^3 volumes):

```bash
# MSD format
python scripts/preprocessing/preprocess_lung.py \
    --input-dir data/raw/Task06_Lung \
    --output-dir data/preprocessed/train \
    --format msd

# NIfTI pairs (ct.nii.gz + seg.nii.gz per case)
python scripts/preprocessing/preprocess_lung.py \
    --input-dir data/raw/my_dataset \
    --output-dir data/preprocessed/train \
    --format nifti

# DICOM series
python scripts/preprocessing/preprocess_lung.py \
    --input-dir data/raw/dicom_cases \
    --output-dir data/preprocessed/train \
    --format dicom
```

### Training Base Models

Each base model is a `LightweightUNet3D` trained at a specific patch scale:

```bash
python scripts/training/train_base_model.py --patch-size 8 --gpu 0 --epochs 250
python scripts/training/train_base_model.py --patch-size 12 --gpu 0 --epochs 250
python scripts/training/train_base_model.py --patch-size 24 --gpu 1 --epochs 250
python scripts/training/train_base_model.py --patch-size 36 --gpu 1 --epochs 250
```

Training features:
- **SmallLesionOptimizedLoss** — weighted combination of Tversky (60%), Focal (25%), and Sensitivity (15%) losses, tuned for detecting small tumors
- **MONAI augmentation** — random flips, rotations, affine transforms, intensity perturbations (30% probability)
- **Hybrid weighted sampling** — oversamples cases with small lesions and difficult cases
- **Cosine annealing** — learning rate schedule with warm restarts

### Training the Stacking Classifier

After base models are trained, the stacking classifier learns to fuse their predictions:

```bash
python scripts/training/train_stacking.py \
    --data-dir data/preprocessed/train \
    --epochs 150 \
    --stacking-patch 32
```

This script:
1. Generates sliding-window predictions from all 4 base models
2. Caches predictions to disk (resumable)
3. Builds 6-channel stacking features (4 predictions + variance + range)
4. Trains a lightweight CNN meta-learner
5. Tunes per-method thresholds
6. Runs failure analysis on worst cases

### Inference

Run inference on new CT volumes:

```bash
python scripts/inference/run_inference.py \
    --input path/to/ct_scan.nii.gz \
    --output path/to/prediction.nii.gz
```

### Evaluation

Evaluate the full ensemble on the validation set:

```bash
python scripts/evaluation/evaluate.py --threshold 0.9
```

Metrics reported:
- **Voxel-level Dice** — overall segmentation overlap
- **Lesion-level F1 / Recall / Precision** — detection performance
- **Size-stratified Dice** — breakdown by lesion size (tiny, small, medium, large)

## Training Details

### Multi-Scale Patch Strategy

| Patch Size | Receptive Field | Strength |
|-----------|----------------|----------|
| 8^3 | Fine-grained | Small lesion detail |
| 12^3 | Local context | Small-to-medium lesions |
| 24^3 | Regional context | Medium lesions |
| 36^3 | Wide context | Large lesions, global structure |

### GPU Requirements

The overnight pipeline trains models in parallel across 2 GPUs:

| Phase | GPU 0 | GPU 1 | Time |
|-------|-------|-------|------|
| Batch 1 | 8-patch | 24-patch | ~3-4h |
| Batch 2 | 12-patch | 36-patch | ~3-4h |
| Stacking | Classifier training | — | ~3-4h |
| Evaluation | Full eval | — | ~1h |

Single-GPU training is also supported — models will train sequentially.

### Data Format

After preprocessing, each case is stored as:

```
data/preprocessed/train/
  case_001/
    ct.nii.gz       # Float32, resampled to 1mm isotropic, 256x256x256
    seg.nii.gz      # Binary mask (uint8), 0 = background, 1 = tumor
  case_002/
    ...
```

CT values are windowed to [-1000, 400] HU and normalized to [0, 1] during training.

## Configuration

Model and inference settings are defined in `configs/models.yaml`:

```yaml
ensemble:
  fusion_mode: "stacking"
  default_threshold: 0.5

stacking:
  path: "model/stacking_classifier.pth"
  in_channels: 6          # 4 predictions + variance + range
  patch_size: 32
  overlap: 0.5
  threshold: 0.9

inference:
  window_size: [96, 96, 96]
  overlap: 0.75
  use_tta: false
  postprocessing:
    min_size: 15           # Remove components smaller than 15 voxels
```

## License

This project is for research and educational purposes.

## Acknowledgments

- [Medical Segmentation Decathlon](http://medicaldecathlon.com/) for the Task06_Lung dataset
- [MONAI](https://monai.io/) for medical imaging transforms
- [PyTorch](https://pytorch.org/) deep learning framework
