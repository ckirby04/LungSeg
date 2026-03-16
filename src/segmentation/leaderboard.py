"""
Leaderboard for tracking model performance across different configurations.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class Leaderboard:
    def __init__(self, leaderboard_path=None):
        if leaderboard_path is None:
            project_dir = Path(__file__).parent.parent.parent
            self.path = project_dir / "model" / "leaderboard.json"
        else:
            self.path = Path(leaderboard_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, 'r') as f:
                return json.load(f)
        return {"models": {}, "ensemble_estimate": {}, "last_updated": None}

    def _save(self):
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def update(self, model_name, patch_size, epoch, train_loss, val_loss, val_dice,
               tiny_dice=None, small_dice=None, medium_dice=None, large_dice=None,
               sensitivity=None, specificity=None, model_path=None, is_best=False):
        key = f"{model_name}_{patch_size}patch"
        current = self.data["models"].get(key, {
            "model_name": model_name, "patch_size": patch_size,
            "best_val_dice": 0, "best_tiny_dice": 0, "best_epoch": 0, "history": []
        })
        entry = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                 "val_dice": val_dice, "timestamp": datetime.now().isoformat()}
        if tiny_dice is not None: entry["tiny_dice"] = tiny_dice
        if small_dice is not None: entry["small_dice"] = small_dice
        if medium_dice is not None: entry["medium_dice"] = medium_dice
        if large_dice is not None: entry["large_dice"] = large_dice
        if sensitivity is not None: entry["sensitivity"] = sensitivity
        if specificity is not None: entry["specificity"] = specificity
        if val_dice > current.get("best_val_dice", 0):
            current["best_val_dice"] = val_dice
            current["best_epoch"] = epoch
            current["best_val_loss"] = val_loss
            if model_path: current["best_model_path"] = model_path
        if tiny_dice is not None and tiny_dice > current.get("best_tiny_dice", 0):
            current["best_tiny_dice"] = tiny_dice
        if small_dice is not None and small_dice > current.get("best_small_dice", 0):
            current["best_small_dice"] = small_dice
        if medium_dice is not None and medium_dice > current.get("best_medium_dice", 0):
            current["best_medium_dice"] = medium_dice
        if large_dice is not None and large_dice > current.get("best_large_dice", 0):
            current["best_large_dice"] = large_dice
        current["history"].append(entry)
        current["history"] = current["history"][-10:]
        current["latest"] = entry
        self.data["models"][key] = current
        self._update_ensemble_estimate()
        self._save()

    def _update_ensemble_estimate(self):
        models = self.data["models"]
        if not models: return
        by_patch = {}
        for key, model in models.items():
            ps = model["patch_size"]
            if ps not in by_patch or model["best_val_dice"] > by_patch[ps]["best_val_dice"]:
                by_patch[ps] = model
        ensemble = {
            "models_used": list(by_patch.keys()),
            "strategy": "size-weighted routing",
        }
        if by_patch:
            dices = [m["best_val_dice"] for m in by_patch.values()]
            ensemble["estimated_overall_dice"] = max(dices) * 1.05
            smallest_patch = min(by_patch.keys())
            ensemble["estimated_tiny_dice"] = by_patch[smallest_patch].get("best_tiny_dice", 0)
            largest_patch = max(by_patch.keys())
            ensemble["estimated_large_dice"] = by_patch[largest_patch].get("best_large_dice", 0)
        self.data["ensemble_estimate"] = ensemble

    def get_summary(self):
        lines = ["=" * 70, "MODEL LEADERBOARD", "=" * 70]
        if not self.data["models"]:
            lines.append("No models trained yet.")
            return "\n".join(lines)
        sorted_models = sorted(self.data["models"].items(), key=lambda x: x[1]["patch_size"])
        lines.append(f"{'Model':<25} {'Patch':>6} {'Best Dice':>10} {'Tiny Dice':>10} {'Epoch':>6}")
        lines.append("-" * 70)
        for key, model in sorted_models:
            name = model["model_name"][:20]
            patch = f"{model['patch_size']}^3"
            dice = f"{model['best_val_dice']*100:.1f}%"
            tiny = f"{model.get('best_tiny_dice', 0)*100:.1f}%"
            epoch = model.get("best_epoch", 0)
            lines.append(f"{name:<25} {patch:>6} {dice:>10} {tiny:>10} {epoch:>6}")
        ens = self.data.get("ensemble_estimate", {})
        if ens:
            lines.extend(["", "-" * 70, "ENSEMBLE ESTIMATE", "-" * 70])
            lines.append(f"Models: {ens.get('models_used', [])}")
            if "estimated_overall_dice" in ens:
                lines.append(f"Estimated Overall Dice: {ens['estimated_overall_dice']*100:.1f}%")
        lines.extend(["", f"Last updated: {self.data.get('last_updated', 'Never')}", "=" * 70])
        return "\n".join(lines)

    def print_summary(self):
        print(self.get_summary())

    def get_best_model(self, metric="val_dice"):
        if not self.data["models"]: return None
        best, best_score = None, -1
        for key, model in self.data["models"].items():
            score = model.get(f"best_{metric}", model.get("best_val_dice", 0))
            if score > best_score:
                best_score = score
                best = model
        return best


_leaderboard = None

def get_leaderboard():
    global _leaderboard
    if _leaderboard is None:
        _leaderboard = Leaderboard()
    return _leaderboard

def update_leaderboard(**kwargs):
    get_leaderboard().update(**kwargs)

def print_leaderboard():
    get_leaderboard().print_summary()
