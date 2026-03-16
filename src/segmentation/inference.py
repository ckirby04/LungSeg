"""
Inference script for lung CT segmentation
Handles full 3D volumes with sliding window approach
"""

import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import os

from unet import LightweightUNet3D
from dataset import LungCTDataset
from tta import MinimalTTA, TestTimeAugmentation


@torch.no_grad()
def sliding_window_inference(model, image, window_size=(96, 96, 96), overlap=0.5,
                              device='cuda', use_tta=False, tta_mode='minimal'):
    """Perform inference on full 3D volume using sliding window"""
    model.eval()
    tta_predictor = None
    if use_tta:
        if tta_mode == 'minimal':
            tta_predictor = MinimalTTA(model, device)
        else:
            tta_predictor = TestTimeAugmentation(model, device)

    C, H, W, D = image.shape
    wh, ww, wd = window_size
    sh, sw, sd = int(wh * (1 - overlap)), int(ww * (1 - overlap)), int(wd * (1 - overlap))
    output = torch.zeros((1, H, W, D), device=device)
    count = torch.zeros((1, H, W, D), device=device)

    h_starts = list(range(0, H - wh + 1, sh)) + [H - wh]
    w_starts = list(range(0, W - ww + 1, sw)) + [W - ww]
    d_starts = list(range(0, D - wd + 1, sd)) + [D - wd]

    total_windows = len(h_starts) * len(w_starts) * len(d_starts)
    pbar = tqdm(total=total_windows, desc="Sliding window inference")

    for h_start in h_starts:
        for w_start in w_starts:
            for d_start in d_starts:
                window = image[:, h_start:h_start+wh, w_start:w_start+ww, d_start:d_start+wd]
                window = window.unsqueeze(0)
                if use_tta:
                    pred, _ = tta_predictor.predict(window, threshold=0.5)
                else:
                    window = window.to(device)
                    pred = model(window)
                    pred = torch.sigmoid(pred)
                output[:, h_start:h_start+wh, w_start:w_start+ww, d_start:d_start+wd] += pred[0].cpu()
                count[:, h_start:h_start+wh, w_start:w_start+ww, d_start:d_start+wd] += 1
                pbar.update(1)
    pbar.close()
    return output / count


def predict_case(model, case_dir, window_size=(96, 96, 96), overlap=0.5,
                 device='cuda', use_tta=False, tta_mode='minimal',
                 hu_min=-1000.0, hu_max=400.0):
    """Predict segmentation for a single case"""
    case_dir = Path(case_dir)
    img_path = case_dir / "ct.nii.gz"
    if not img_path.exists():
        raise FileNotFoundError(f"Missing ct.nii.gz for case {case_dir.name}")

    nii = nib.load(str(img_path))
    img = nii.get_fdata().astype(np.float32)
    affine = nii.affine

    # HU windowing
    img = np.clip(img, hu_min, hu_max)
    img = (img - hu_min) / (hu_max - hu_min)

    # Shape: (1, H, W, D)
    image = np.expand_dims(img, axis=0)
    image = torch.from_numpy(image).float()

    with torch.no_grad():
        pred = sliding_window_inference(model, image, window_size, overlap, device, use_tta, tta_mode)

    pred = (pred[0] > 0.5).float().cpu().numpy()
    return pred, affine


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if 'args' in checkpoint:
        model_args = checkpoint['args']
        base_channels = model_args.get('base_channels', 16)
        depth = model_args.get('depth', 3)
        dropout = model_args.get('dropout', 0.1)
    else:
        base_channels = args.base_channels
        depth = args.depth
        dropout = 0.0

    model = LightweightUNet3D(
        in_channels=1, out_channels=1,
        base_channels=base_channels, depth=depth, dropout_p=dropout
    ).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    input_dir = Path(args.input_dir)
    cases = sorted([d for d in input_dir.iterdir() if d.is_dir() and (d / 'ct.nii.gz').exists()])
    print(f"\nProcessing {len(cases)} cases...")

    for case_dir in cases:
        case_id = case_dir.name
        print(f"\nProcessing {case_id}...")
        try:
            pred, affine = predict_case(
                model, case_dir, window_size=tuple(args.window_size),
                overlap=args.overlap, device=device,
                use_tta=args.use_tta, tta_mode=args.tta_mode
            )
            output_path = Path(args.output_dir) / f"{case_id}_pred.nii.gz"
            nii_out = nib.Nifti1Image(pred.astype(np.float32), affine)
            nib.save(nii_out, str(output_path))
            print(f"  Saved to {output_path}")
            print(f"  Detected lesion voxels: {np.sum(pred > 0)}")
        except Exception as e:
            print(f"  Error processing {case_id}: {e}")

    print(f"\nInference completed! Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on lung CT cases")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='../../results/predictions')
    parser.add_argument('--window_size', type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument('--overlap', type=float, default=0.75)
    parser.add_argument('--base_channels', type=int, default=16)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--use_tta', action='store_true')
    parser.add_argument('--tta_mode', type=str, default='minimal', choices=['minimal', 'full'])
    args = parser.parse_args()
    main(args)
