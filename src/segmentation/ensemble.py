"""
SmartEnsemble module for multi-model segmentation.
Inference-only: loads multiple models and fuses their predictions.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .postprocessing import extract_lesion_details, full_postprocessing_pipeline
from .unet import LightweightUNet3D
from .enhanced_unet import DeepSupervisedUNet3D
from .tta import MinimalTTA, TestTimeAugmentation


class SmartEnsemble(nn.Module):
    def __init__(self, model_configs: list, device: str = "cuda", fusion_mode: str = "union"):
        super().__init__()
        self.device = device
        self.fusion_mode = fusion_mode
        self.models = nn.ModuleList()
        self.patch_sizes = []
        self.thresholds = []
        self.names = []

        for cfg in model_configs:
            model_path = cfg.get("full_path", cfg.get("path", ""))
            if not Path(model_path).exists():
                print(f"  Warning: {cfg['name']} not found at {model_path}, skipping")
                continue
            print(f"  Loading {cfg['name']} (patch {cfg['patch_size']}, threshold {cfg['threshold']})...")
            model = self._create_model(cfg)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            self.models.append(model)
            self.patch_sizes.append(cfg["patch_size"])
            self.thresholds.append(cfg["threshold"])
            self.names.append(cfg["name"])
        print(f"  Loaded {len(self.models)} models")
        if fusion_mode == "weighted" and len(self.models) > 0:
            self.fusion_weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))

    @staticmethod
    def _create_model(cfg: dict) -> nn.Module:
        arch = cfg.get("architecture", "lightweight")
        if arch == "deep_supervised":
            return DeepSupervisedUNet3D(
                in_channels=1, out_channels=1,
                base_channels=cfg.get("base_channels", 16),
                depth=cfg.get("depth", 4),
                dropout_p=0.0, deep_supervision=False,
            )
        else:
            return LightweightUNet3D(
                in_channels=1, out_channels=1,
                base_channels=cfg.get("base_channels", 16),
                depth=cfg.get("depth", 3),
                dropout_p=0.0,
                use_attention=cfg.get("use_attention", True),
                use_residual=cfg.get("use_residual", True),
            )

    @classmethod
    def from_config(cls, config_path: str, device: str = "cuda"):
        config_path = Path(config_path)
        project_root = config_path.parent.parent
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_configs = []
        for m in config.get("models", []):
            full_path = project_root / m["path"]
            model_configs.append({**m, "full_path": str(full_path)})
        fusion_mode = config.get("ensemble", {}).get("fusion_mode", "union")
        return cls(model_configs, device=device, fusion_mode=fusion_mode)

    @classmethod
    def from_registry(cls, registry, device: str = "cuda"):
        config = registry.get_ensemble_config()
        fusion_mode = config.get("ensemble", {}).get("fusion_mode", "union")
        return cls(config["models"], device=device, fusion_mode=fusion_mode)

    def forward(self, x, target_size=None):
        if target_size is None:
            target_size = x.shape[2]
        predictions = []
        for model, patch_size in zip(self.models, self.patch_sizes):
            if patch_size != x.shape[2]:
                x_resized = F.interpolate(x, size=(patch_size,) * 3, mode="trilinear", align_corners=False)
            else:
                x_resized = x
            with torch.no_grad():
                pred = model(x_resized)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                pred_prob = torch.sigmoid(pred)
            if patch_size != target_size:
                pred_prob = F.interpolate(pred_prob, size=(target_size,) * 3, mode="trilinear", align_corners=False)
            predictions.append(pred_prob)
        return self._fuse(predictions)

    def _fuse(self, predictions):
        if not predictions:
            raise ValueError("No predictions to fuse")
        stacked = torch.stack(predictions, dim=0)
        if self.fusion_mode == "union":
            return stacked.max(dim=0)[0]
        elif self.fusion_mode == "weighted":
            weights = F.softmax(self.fusion_weights, dim=0)
            return sum(w * p for w, p in zip(weights, predictions))
        elif self.fusion_mode == "hybrid":
            union = stacked.max(dim=0)[0]
            average = stacked.mean(dim=0)
            confident = (union > 0.3).float()
            return confident * union + (1 - confident) * average
        else:
            return stacked.mean(dim=0)

    @torch.no_grad()
    def predict_volume(self, image, window_size=(96, 96, 96), overlap=0.75,
                       use_tta=False, tta_mode="minimal", threshold=0.5,
                       postprocess=True, postprocess_params=None, voxel_spacing=(1.0, 1.0, 1.0)):
        C, H, W, D = image.shape
        wh, ww, wd = window_size
        tta_predictor = None
        if use_tta and len(self.models) > 0:
            if tta_mode == "minimal":
                tta_predictor = MinimalTTA(self, self.device)
            else:
                tta_predictor = TestTimeAugmentation(self, self.device)
        output = torch.zeros((1, H, W, D), device="cpu")
        count = torch.zeros((1, H, W, D), device="cpu")
        sh, sw, sd = int(wh * (1 - overlap)), int(ww * (1 - overlap)), int(wd * (1 - overlap))
        h_starts = list(range(0, max(1, H - wh + 1), sh))
        if H > wh and h_starts[-1] + wh < H: h_starts.append(H - wh)
        w_starts = list(range(0, max(1, W - ww + 1), sw))
        if W > ww and w_starts[-1] + ww < W: w_starts.append(W - ww)
        d_starts = list(range(0, max(1, D - wd + 1), sd))
        if D > wd and d_starts[-1] + wd < D: d_starts.append(D - wd)
        for h_start in h_starts:
            for w_start in w_starts:
                for d_start in d_starts:
                    window = image[:, h_start:h_start+wh, w_start:w_start+ww, d_start:d_start+wd]
                    window = window.unsqueeze(0)
                    if use_tta and tta_predictor is not None:
                        pred, _ = tta_predictor.predict(window, threshold=threshold)
                    else:
                        window = window.to(self.device)
                        pred = self.forward(window, target_size=wh)
                    output[:, h_start:h_start+wh, w_start:w_start+ww, d_start:d_start+wd] += pred[0].cpu()
                    count[:, h_start:h_start+wh, w_start:w_start+ww, d_start:d_start+wd] += 1
        probability_map = (output / (count + 1e-8)).numpy()
        pp = postprocess_params or {}
        if postprocess:
            binary_mask = full_postprocessing_pipeline(
                probability_map[0], threshold=pp.get("threshold", threshold),
                min_size=pp.get("min_size", 15), opening_size=pp.get("opening_size", 1),
                closing_size=pp.get("closing_size", 1),
            )
        else:
            binary_mask = (probability_map[0] > threshold).astype(np.float32)
        lesion_details = extract_lesion_details(binary_mask, probability_map[0], voxel_spacing=voxel_spacing)
        return {
            "probability_map": probability_map, "binary_mask": binary_mask,
            "lesion_count": len(lesion_details), "lesion_details": lesion_details,
        }
