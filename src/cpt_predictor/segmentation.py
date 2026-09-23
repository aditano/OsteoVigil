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


def locate_pseudarthrosis_slice(
    mask: np.ndarray,
    hu_volume: Optional[np.ndarray] = None,
    interior_fraction: float = 0.12,
) -> int:
    """Pick a mid-shaft defect slice instead of a tapered bone end."""
    slice_area = np.asarray(mask, dtype=bool).sum(axis=(1, 2)).astype(float)
    nonzero = np.where(slice_area > 1e-3)[0]
    if nonzero.size == 0:
        return 0

    z0 = int(nonzero[0])
    z1 = int(nonzero[-1]) + 1
    length = z1 - z0
    margin = max(2, int(interior_fraction * length)) if length >= 16 else 0
    interior = np.arange(z0 + margin, max(z0 + margin + 1, z1 - margin))
    if interior.size == 0:
        interior = nonzero

    area_vals = np.maximum(slice_area[interior], 1.0)
    area_score = area_vals / max(float(np.median(area_vals)), 1.0)

    if hu_volume is None:
        return int(interior[int(np.argmin(area_vals))])

    hu_means = np.zeros(interior.shape[0], dtype=float)
    for index, z_index in enumerate(interior):
        voxels = np.asarray(hu_volume[z_index][mask[z_index]], dtype=float)
        hu_means[index] = float(voxels.mean()) if voxels.size else 0.0
    hu_median = float(np.median(hu_means)) if hu_means.size else 1.0
    hu_score = hu_means / max(hu_median, 1.0)
    combined = 0.45 * area_score + 0.55 * hu_score
    return int(interior[int(np.argmin(combined))])


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
    # Close only along the slice axis so CPT gaps are bridged without welding
    # the tibia to the fibula or tarsal bones across the joint space.
    mask = ndi.binary_closing(mask, structure=np.ones((bridge_gap, 1, 1), dtype=bool))
    mask = ndi.binary_opening(mask, structure=np.ones((1, 3, 3), dtype=bool))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)

    labeled = measure.label(mask, connectivity=2)
    if labeled.max() == 0:
        return mask.astype(bool)

    center_y = hu_volume.shape[1] / 2.0
    center_x = hu_volume.shape[2] / 2.0
    scored: list[tuple[float, Any]] = []
    for region in measure.regionprops(labeled):
        z0, y0, x0, z1, y1, x1 = region.bbox
        z_extent = z1 - z0
        cx = float(region.centroid[2])
        cy = float(region.centroid[1])
        center_penalty = abs(cx - center_x) + abs(cy - center_y)
        score = float(region.area) + (25.0 * z_extent) - (4.0 * center_penalty)
        scored.append((score, region))

    scored.sort(key=lambda item: item[0], reverse=True)
    primary = scored[0][1]
    primary_cy = float(primary.centroid[1])
    primary_cx = float(primary.centroid[2])
    primary_z0, primary_z1 = int(primary.bbox[0]), int(primary.bbox[3])
    primary_length = max(1, primary_z1 - primary_z0)
    primary_radius = max(np.sqrt(float(primary.area) / (primary_length * np.pi)), 6.0)
    keep_labels = {primary.label}

    # CPT defects can split one tibia into axially separated fragments. Keep those
    # co-axial pieces, but skip parallel bones such as the fibula that overlap in z
    # and compact tarsal bones that sit beyond the plafond.
    min_fragment_length = max(8, int(0.12 * primary_length))
    for _score, region in scored[1:]:
        offset = np.hypot(float(region.centroid[2]) - primary_cx, float(region.centroid[1]) - primary_cy)
        if offset > 1.5 * primary_radius:
            continue
        z0, z1 = int(region.bbox[0]), int(region.bbox[3])
        overlap = min(primary_z1, z1) - max(primary_z0, z0)
        fragment_length = max(1, z1 - z0)
        if overlap > 0.25 * min(primary_length, fragment_length):
            continue
        if fragment_length < min_fragment_length:
            continue
        keep_labels.add(region.label)

    selected = np.isin(labeled, list(keep_labels))
    z_kernel = max(bridge_gap * 2 + 1, 9)
    selected = ndi.binary_closing(selected, structure=np.ones((z_kernel, 1, 1), dtype=bool))
    selected = ndi.binary_fill_holes(selected)
    selected = morphology.remove_small_objects(selected.astype(bool), min_size=min_size)
    selected = ndi.binary_dilation(selected, iterations=radius)
    selected = ndi.binary_erosion(selected, iterations=radius)
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

        defect_index = locate_pseudarthrosis_slice(mask, study_data.hu_volume)

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
