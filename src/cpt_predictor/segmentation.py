from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .models import SegmentationResult, StudyData


def _require_scipy():
    from scipy import ndimage as ndi

    return ndi


def _require_skimage():
    from skimage import measure, morphology

    return measure, morphology


def _segmentation_config(config: Optional[Dict[str, Any]] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = {
        "segmentation": {
            "bone_threshold_hu": 180,
            "remove_small_objects_voxels": 2500,
            "bridge_gap_slices": 4,
            "morphology_radius_voxels": 2,
            "monai_model_path": None,
        }
    }
    if config:
        for key, value in config.items():
            if key == "segmentation" and isinstance(value, dict):
                base["segmentation"].update(value)
            else:
                base[key] = value
    if overrides:
        base["segmentation"].update(overrides)
    return base


def classical_tibia_segmentation(
    hu_volume: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> np.ndarray:
    ndi = _require_scipy()
    measure, morphology = _require_skimage()

    seg_cfg = _segmentation_config(config, overrides)["segmentation"]
    threshold = float(seg_cfg.get("bone_threshold_hu", 180))
    min_size = int(seg_cfg.get("remove_small_objects_voxels", 2500))
    bridge_gap = max(1, int(seg_cfg.get("bridge_gap_slices", 4)))
    radius = max(1, int(seg_cfg.get("morphology_radius_voxels", 2)))

    mask = hu_volume >= threshold
    structure = np.ones((bridge_gap, 3, 3), dtype=bool)
    mask = ndi.binary_closing(mask, structure=structure)
    mask = ndi.binary_opening(mask, structure=np.ones((1, 3, 3), dtype=bool))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)

    labeled = measure.label(mask, connectivity=2)
    if labeled.max() == 0:
        return mask.astype(bool)

    center_y = hu_volume.shape[1] / 2.0
    center_x = hu_volume.shape[2] / 2.0
    best_label = None
    best_score = -1.0

    for region in measure.regionprops(labeled):
        z0, y0, x0, z1, y1, x1 = region.bbox
        z_extent = z1 - z0
        cx = float(region.centroid[2])
        cy = float(region.centroid[1])
        center_penalty = abs(cx - center_x) + abs(cy - center_y)
        score = float(region.area) + (25.0 * z_extent) - (4.0 * center_penalty)
        if score > best_score:
            best_score = score
            best_label = region.label

    selected = labeled == best_label
    selected = ndi.binary_closing(selected, structure=np.ones((bridge_gap, 3, 3), dtype=bool))
    selected = ndi.binary_fill_holes(selected)
    selected = morphology.remove_small_objects(selected.astype(bool), min_size=min_size)
    selected = ndi.binary_dilation(selected, iterations=radius)
    selected = ndi.binary_erosion(selected, iterations=max(1, radius - 1))
    return selected.astype(bool)


def monai_tibia_segmentation(study: StudyData, model_path: str) -> np.ndarray:
    import torch

    input_volume = study.normalized_volume
    if input_volume is None:
        clipped = np.clip(study.hu_volume, -1000.0, 2500.0)
        input_volume = (clipped + 1000.0) / 3500.0

    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    tensor = torch.from_numpy(input_volume.astype(np.float32))[None, None, ...]
    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.sigmoid(logits)

    return (probs.cpu().numpy()[0, 0] > 0.5).astype(bool)


class TibiaSegmenter:
    def __init__(self, config: Optional[Dict[str, Any]] = None, **overrides: Any):
        self.config = _segmentation_config(config, overrides)

    def segment(
        self,
        study: Any,
        output_dir: Optional[Path] = None,
        force_classical: bool = False,
    ) -> Any:
        is_raw_volume = isinstance(study, np.ndarray)
        model_path = self.config["segmentation"].get("monai_model_path")
        method = "classical"

        if is_raw_volume:
            hu_volume = np.asarray(study, dtype=np.float32)
            study_data = StudyData(
                volume=hu_volume.copy(),
                hu_volume=hu_volume,
                spacing_zyx=(1.0, 1.0, 1.0),
                origin_xyz=(0.0, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            )
        else:
            study_data = study

        if model_path and Path(model_path).exists() and not force_classical:
            try:
                mask = monai_tibia_segmentation(study_data, model_path)
                method = "monai"
            except Exception:
                mask = classical_tibia_segmentation(study_data.hu_volume, self.config)
                method = "classical_fallback"
        else:
            mask = classical_tibia_segmentation(study_data.hu_volume, self.config)

        slice_area = mask.sum(axis=(1, 2)).astype(float)
        nonzero = np.where(slice_area > 0)[0]
        defect_index = int(nonzero[np.argmin(slice_area[nonzero])]) if len(nonzero) else 0

        stats = {
            "voxel_count": int(mask.sum()),
            "slice_count": int(mask.shape[0]),
            "pseudarthrosis_slice_index": defect_index,
            "method": method,
        }
        if is_raw_volume:
            return mask.astype(bool)
        output_dir = output_dir or Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / "tibia_mask.npy"
        np.save(mask_path, mask.astype(np.uint8))
        return SegmentationResult(mask=mask, method=method, stats=stats, mask_path=mask_path)
