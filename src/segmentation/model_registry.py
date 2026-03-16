"""
Model Registry for managing trained segmentation model checkpoints.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml


class ModelRegistry:
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = Path(project_root)
        self.model_dir = self.project_root / "model"
        self.config_path = self.project_root / "configs" / "models.yaml"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def register_model(self, source_path, name, patch_size, architecture="lightweight",
                       threshold=0.5, base_channels=16, depth=3,
                       use_attention=True, use_residual=True) -> Path:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source model not found: {source}")
        self.validate_checkpoint(str(source), architecture)
        dest = self.model_dir / f"{name}_best.pth"
        shutil.copy2(source, dest)
        config = self._load_config()
        model_entry = {
            "name": name, "path": f"model/{dest.name}",
            "architecture": architecture, "patch_size": patch_size,
            "threshold": threshold, "base_channels": base_channels,
            "depth": depth, "use_attention": use_attention,
            "use_residual": use_residual,
        }
        existing = [m for m in config.get("models", []) if m["name"] != name]
        existing.append(model_entry)
        config["models"] = existing
        self._save_config(config)
        print(f"Registered model '{name}' -> {dest}")
        return dest

    def validate_checkpoint(self, path, architecture="lightweight"):
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise ValueError(f"Cannot load checkpoint: {e}")
        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint missing 'model_state_dict' key")
        state_dict = checkpoint["model_state_dict"]
        has_inc = any(k.startswith("inc.") for k in state_dict.keys())
        has_down = any(k.startswith("down_blocks.") for k in state_dict.keys())
        has_up = any(k.startswith("up_blocks.") for k in state_dict.keys())
        has_outc = any(k.startswith("outc.") for k in state_dict.keys())
        if not all([has_inc, has_down, has_up, has_outc]):
            raise ValueError("Checkpoint state_dict missing expected UNet layers")
        if architecture == "deep_supervised":
            if not any(k.startswith("bottleneck.") for k in state_dict.keys()):
                raise ValueError("Deep supervised checkpoint missing bottleneck layer")
        return True

    def list_models(self):
        config = self._load_config()
        result = []
        for m in config.get("models", []):
            model_path = self.project_root / m["path"]
            result.append({**m, "exists": model_path.exists()})
        return result

    def get_ensemble_config(self):
        config = self._load_config()
        available = []
        for m in config.get("models", []):
            model_path = self.project_root / m["path"]
            if model_path.exists():
                available.append({**m, "full_path": str(model_path)})
        return {
            "models": available,
            "ensemble": config.get("ensemble", {"fusion_mode": "union"}),
            "inference": config.get("inference", {}),
        }

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        return {"ensemble": {"fusion_mode": "union"}, "models": [], "inference": {}}

    def _save_config(self, config):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
