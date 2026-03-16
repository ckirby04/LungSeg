"""
Postprocessing module for segmentation.
Provides morphological operations, component filtering, and lesion extraction.
"""

from typing import Dict, List

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_opening


def remove_small_components(pred_binary: np.ndarray, min_size: int = 10) -> np.ndarray:
    if pred_binary.ndim == 5:
        result = np.zeros_like(pred_binary)
        for b in range(pred_binary.shape[0]):
            result[b, 0] = _remove_small_3d(pred_binary[b, 0], min_size)
        return result
    return _remove_small_3d(pred_binary, min_size)


def _remove_small_3d(volume: np.ndarray, min_size: int) -> np.ndarray:
    labeled, n_components = ndimage.label(volume)
    result = np.zeros_like(volume)
    for i in range(1, n_components + 1):
        component = labeled == i
        if component.sum() >= min_size:
            result[component] = 1
    return result


def morphological_opening(pred_binary: np.ndarray, structure_size: int = 2) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, 1)
    if structure_size > 1:
        structure = ndimage.iterate_structure(structure, structure_size)
    if pred_binary.ndim == 5:
        result = np.zeros_like(pred_binary)
        for b in range(pred_binary.shape[0]):
            result[b, 0] = binary_opening(pred_binary[b, 0], structure=structure).astype(np.float32)
        return result
    return binary_opening(pred_binary, structure=structure).astype(np.float32)


def morphological_closing(pred_binary: np.ndarray, structure_size: int = 2) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, 1)
    if structure_size > 1:
        structure = ndimage.iterate_structure(structure, structure_size)
    if pred_binary.ndim == 5:
        result = np.zeros_like(pred_binary)
        for b in range(pred_binary.shape[0]):
            result[b, 0] = binary_closing(pred_binary[b, 0], structure=structure).astype(np.float32)
        return result
    return binary_closing(pred_binary, structure=structure).astype(np.float32)


def full_postprocessing_pipeline(
    pred_probs: np.ndarray,
    threshold: float = 0.5,
    min_size: int = 15,
    opening_size: int = 1,
    closing_size: int = 1,
) -> np.ndarray:
    pred_binary = (pred_probs > threshold).astype(np.float32)
    if opening_size > 0:
        pred_binary = morphological_opening(pred_binary, structure_size=opening_size)
    if min_size > 0:
        pred_binary = remove_small_components(pred_binary, min_size=min_size)
    if closing_size > 0:
        pred_binary = morphological_closing(pred_binary, structure_size=closing_size)
    return pred_binary


def extract_lesion_details(
    binary_mask: np.ndarray,
    probability_map: np.ndarray = None,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
) -> List[Dict]:
    labeled, n_lesions = ndimage.label(binary_mask)
    voxel_volume_mm3 = float(np.prod(voxel_spacing))
    lesions = []
    for i in range(1, n_lesions + 1):
        component = labeled == i
        volume_voxels = int(component.sum())
        coords = np.argwhere(component)
        centroid = coords.mean(axis=0).tolist()
        mins = coords.min(axis=0).tolist()
        maxs = coords.max(axis=0).tolist()
        extent_mm = [(maxs[d] - mins[d] + 1) * voxel_spacing[d] for d in range(3)]
        max_diameter_mm = max(extent_mm)
        confidence = 1.0
        if probability_map is not None:
            confidence = float(probability_map[component].mean())
        lesions.append({
            "id": i,
            "volume_voxels": volume_voxels,
            "volume_mm3": round(volume_voxels * voxel_volume_mm3, 2),
            "centroid": [round(c, 1) for c in centroid],
            "confidence": round(confidence, 4),
            "max_diameter_mm": round(max_diameter_mm, 2),
            "bounding_box": {"min": mins, "max": maxs},
        })
    return lesions
