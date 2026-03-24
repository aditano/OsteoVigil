"""Preprocessing helpers for CT normalization and resampling."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Tuple, Union

import numpy as np
from scipy import ndimage as ndi

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover - optional dependency at import time
    sitk = None

from .io.dicom_loader import CTVolume
from .models import StudyData


class MultipleLegSelectionRequiredError(ValueError):
    def __init__(self, scan_scope: str):
        self.scan_scope = scan_scope
        super().__init__(
            "A bilateral or full-body CT was detected. Please choose the left or right leg and run again."
        )


def clip_hu(volume: CTVolume, lower_hu: float = -1000.0, upper_hu: float = 2500.0) -> CTVolume:
    clipped = np.clip(volume.array.astype(np.float32), lower_hu, upper_hu)
    return replace(volume, array=clipped)


def normalize_hu(volume: CTVolume, lower_hu: float = -1000.0, upper_hu: float = 2500.0) -> np.ndarray:
    clipped = np.clip(volume.array.astype(np.float32), lower_hu, upper_hu)
    return (clipped - lower_hu) / (upper_hu - lower_hu)


def _config_get(config: Any, dotted_key: str, default: Any = None) -> Any:
    current = config
    for part in dotted_key.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
    return default if current is None else current


def _slice_components(binary_slice: np.ndarray, min_area: int) -> list[dict[str, float]]:
    labels, num_labels = ndi.label(binary_slice)
    if num_labels == 0:
        return []

    components: list[dict[str, float]] = []
    objects = ndi.find_objects(labels)
    for label_id, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        region = labels[bounds] == label_id
        area = int(region.sum())
        if area < min_area:
            continue
        ys, xs = np.where(region)
        y0 = int(bounds[0].start)
        x0 = int(bounds[1].start)
        components.append(
            {
                "area": float(area),
                "y0": float(y0),
                "y1": float(bounds[0].stop),
                "x0": float(x0),
                "x1": float(bounds[1].stop),
                "centroid_y": float(y0 + ys.mean()),
                "centroid_x": float(x0 + xs.mean()),
            }
        )
    return sorted(components, key=lambda item: item["centroid_x"])


def _split_components_by_largest_gap(
    components: list[dict[str, float]],
    min_interleg_gap_px: int,
) -> tuple[list[dict[str, float]], list[dict[str, float]]] | None:
    if len(components) < 2:
        return None

    largest_gap = 0.0
    split_index: int | None = None
    for index in range(len(components) - 1):
        gap = max(0.0, float(components[index + 1]["x0"] - components[index]["x1"]))
        if gap > largest_gap:
            largest_gap = gap
            split_index = index

    if split_index is None or largest_gap < float(min_interleg_gap_px):
        return None

    return components[: split_index + 1], components[split_index + 1 :]


def _component_group_bounds(components: list[dict[str, float]]) -> tuple[float, float, float, float]:
    return (
        float(min(component["x0"] for component in components)),
        float(max(component["x1"] for component in components)),
        float(min(component["y0"] for component in components)),
        float(max(component["y1"] for component in components)),
    )


def _longest_true_run(mask: np.ndarray, min_length: int) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    run_start: int | None = None
    for index, value in enumerate(mask.astype(bool).tolist() + [False]):
        if value and run_start is None:
            run_start = index
        elif not value and run_start is not None:
            candidate = (run_start, index)
            if candidate[1] - candidate[0] >= min_length:
                if best is None or (candidate[1] - candidate[0]) > (best[1] - best[0]):
                    best = candidate
            run_start = None
    return best


def _crop_study_to_bounds(
    study: StudyData,
    *,
    z_bounds: tuple[int, int],
    y_bounds: tuple[int, int],
    x_bounds: tuple[int, int],
    localization_metadata: dict[str, Any],
) -> StudyData:
    z0, z1 = z_bounds
    y0, y1 = y_bounds
    x0, x1 = x_bounds

    spacing_z, spacing_y, spacing_x = study.spacing_zyx
    origin_x, origin_y, origin_z = study.origin_xyz
    metadata = dict(study.metadata)
    metadata["leg_localization"] = localization_metadata

    normalized = study.normalized_volume[z0:z1, y0:y1, x0:x1].copy() if study.normalized_volume is not None else None
    return StudyData(
        volume=study.volume[z0:z1, y0:y1, x0:x1].copy(),
        hu_volume=study.hu_volume[z0:z1, y0:y1, x0:x1].copy(),
        spacing_zyx=study.spacing_zyx,
        origin_xyz=(
            float(origin_x + x0 * spacing_x),
            float(origin_y + y0 * spacing_y),
            float(origin_z + z0 * spacing_z),
        ),
        direction=study.direction,
        metadata=metadata,
        normalized_volume=normalized,
        source_dir=study.source_dir,
    )


def _select_leg_crop_bounds(study: StudyData, config: Any) -> StudyData:
    target_leg = str(_config_get(config, "input.target_leg_side", "auto")).strip().lower()
    if target_leg not in {"auto", "left", "right"}:
        target_leg = "auto"

    enabled = bool(_config_get(config, "input.enable_leg_localization", True))
    if not enabled:
        return study

    threshold_hu = int(_config_get(config, "preprocessing.leg_localization_bone_threshold_hu", 220))
    min_component_area = int(_config_get(config, "preprocessing.leg_localization_min_component_area_px", 80))
    min_bilateral_run = int(_config_get(config, "preprocessing.leg_localization_min_bilateral_run_slices", 12))
    min_tibfib_run = int(_config_get(config, "preprocessing.leg_localization_min_tibfib_run_slices", 18))
    max_tibfib_components = int(_config_get(config, "preprocessing.leg_localization_max_tibfib_components", 4))
    crop_margin_mm = float(_config_get(config, "preprocessing.leg_localization_margin_mm", 20.0))
    min_interleg_gap_mm = float(_config_get(config, "preprocessing.leg_localization_min_interleg_gap_mm", 35.0))
    full_body_extent_mm = float(_config_get(config, "preprocessing.full_body_extent_threshold_mm", 700.0))
    min_interleg_gap_px = max(1, int(np.ceil(min_interleg_gap_mm / study.spacing_zyx[2])))

    bone_mask = study.hu_volume > threshold_hu
    if not bone_mask.any():
        metadata = dict(study.metadata)
        metadata["leg_localization"] = {"detected": False, "reason": "no_bone_found"}
        return replace(study, metadata=metadata)

    slice_components = [_slice_components(bone_mask[z_index], min_component_area) for z_index in range(bone_mask.shape[0])]
    slice_groups = [_split_components_by_largest_gap(components, min_interleg_gap_px) for components in slice_components]
    bilateral_flags = np.array(
        [groups is not None for groups in slice_groups],
        dtype=bool,
    )
    bilateral_run = _longest_true_run(bilateral_flags, min_bilateral_run)
    extent_mm = float(study.hu_volume.shape[0] * study.spacing_zyx[0])
    scan_scope = "single_leg"
    if bilateral_run is not None and extent_mm >= full_body_extent_mm:
        scan_scope = "full_body_or_multileg"
    elif bilateral_run is not None:
        scan_scope = "bilateral_lower_extremity"

    if bilateral_run is None:
        metadata = dict(study.metadata)
        metadata["leg_localization"] = {
            "detected": False,
            "scan_scope": scan_scope,
            "target_leg_side": "single_leg",
            "cropped": False,
        }
        return replace(study, metadata=metadata)

    if target_leg == "auto":
        raise MultipleLegSelectionRequiredError(scan_scope)

    x_margin = max(4, int(np.ceil(crop_margin_mm / study.spacing_zyx[2])))
    y_margin = max(4, int(np.ceil(crop_margin_mm / study.spacing_zyx[1])))
    z_margin = max(2, int(np.ceil((0.5 * crop_margin_mm) / study.spacing_zyx[0])))

    selected_group_index = 0 if target_leg == "left" else 1
    selected_groups = [
        groups[selected_group_index] if groups is not None else []
        for groups in slice_groups
    ]
    selected_group_counts = np.array([len(group) for group in selected_groups], dtype=int)
    tibfib_flags = bilateral_flags & (selected_group_counts >= 2) & (selected_group_counts <= max_tibfib_components)
    tibfib_run = _longest_true_run(tibfib_flags, min_tibfib_run)
    if tibfib_run is None:
        fallback_flags = bilateral_flags & (selected_group_counts >= 1)
        tibfib_run = _longest_true_run(fallback_flags, min_tibfib_run)
    if tibfib_run is None:
        raise ValueError("Could not find a stable tibia/fibula region in the selected leg.")

    z0 = max(0, tibfib_run[0] - z_margin)
    z1 = min(study.hu_volume.shape[0], tibfib_run[1] + z_margin)

    tibfib_groups = [group for group in selected_groups[tibfib_run[0] : tibfib_run[1]] if group]
    if not tibfib_groups:
        raise ValueError("Could not isolate the requested leg from the bilateral CT.")

    group_bounds = [_component_group_bounds(group) for group in tibfib_groups]
    x0_values = np.array([bounds[0] for bounds in group_bounds], dtype=float)
    x1_values = np.array([bounds[1] for bounds in group_bounds], dtype=float)
    if len(group_bounds) >= 6:
        coarse_x0 = max(0, int(np.floor(np.percentile(x0_values, 10.0))) - x_margin)
        coarse_x1 = min(study.hu_volume.shape[2], int(np.ceil(np.percentile(x1_values, 90.0))) + x_margin)
    else:
        coarse_x0 = max(0, int(np.floor(np.min(x0_values))) - x_margin)
        coarse_x1 = min(study.hu_volume.shape[2], int(np.ceil(np.max(x1_values))) + x_margin)

    cropped_mask = bone_mask[z0:z1, :, coarse_x0:coarse_x1]
    coords = np.argwhere(cropped_mask)
    if coords.size == 0:
        raise ValueError("The selected leg crop did not contain any osseous voxels.")

    y0 = max(0, int(coords[:, 1].min()) - y_margin)
    y1 = min(study.hu_volume.shape[1], int(coords[:, 1].max()) + 1 + y_margin)
    x_local_min = int(coords[:, 2].min())
    x_local_max = int(coords[:, 2].max()) + 1
    x0 = max(0, coarse_x0 + x_local_min - x_margin)
    x1 = min(study.hu_volume.shape[2], coarse_x0 + x_local_max + x_margin)
    if x1 <= x0:
        x0 = max(0, x0 - x_margin)
        x1 = min(study.hu_volume.shape[2], coarse_x0 + x_local_max + 2 * x_margin)

    localization_metadata = {
        "detected": True,
        "scan_scope": scan_scope,
        "target_leg_side": target_leg,
        "cropped": True,
        "crop_bounds_zyx": {
            "z": [int(z0), int(z1)],
            "y": [int(y0), int(y1)],
            "x": [int(x0), int(x1)],
        },
        "bilateral_run_slices": [int(bilateral_run[0]), int(bilateral_run[1])],
        "tibfib_run_slices": [int(tibfib_run[0]), int(tibfib_run[1])],
    }
    return _crop_study_to_bounds(
        study,
        z_bounds=(z0, z1),
        y_bounds=(y0, y1),
        x_bounds=(x0, x1),
        localization_metadata=localization_metadata,
    )


def resample_isotropic(volume: CTVolume, target_spacing_mm: Union[float, Tuple[float, float, float]] = 1.0) -> CTVolume:
    if isinstance(target_spacing_mm, (int, float)):
        target_spacing = (float(target_spacing_mm),) * 3
    else:
        target_spacing = tuple(float(v) for v in target_spacing_mm)

    if sitk is not None:
        image = sitk.GetImageFromArray(volume.array.astype(np.float32))
        image.SetSpacing((float(volume.spacing_mm[2]), float(volume.spacing_mm[1]), float(volume.spacing_mm[0])))
        image.SetOrigin((float(volume.origin_mm[2]), float(volume.origin_mm[1]), float(volume.origin_mm[0])))
        if len(volume.direction) == 9:
            image.SetDirection(tuple(float(v) for v in volume.direction))

        original_spacing = np.array(image.GetSpacing(), dtype=float)
        original_size = np.array(image.GetSize(), dtype=float)
        target_spacing_xyz = np.array((target_spacing[2], target_spacing[1], target_spacing[0]), dtype=float)
        target_size = np.maximum(np.round(original_size * (original_spacing / target_spacing_xyz)).astype(int), 1)

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(tuple(target_spacing_xyz))
        resampler.SetSize([int(v) for v in target_size.tolist()])
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetDefaultPixelValue(-1024.0)
        resampled = resampler.Execute(image)
        array = sitk.GetArrayFromImage(resampled).astype(np.float32)
    else:
        from scipy.ndimage import zoom

        zoom_factors = tuple(float(old / new) for old, new in zip(volume.spacing_mm, target_spacing))
        array = zoom(volume.array.astype(np.float32), zoom=zoom_factors, order=1)

    return CTVolume(
        array=array,
        spacing_mm=target_spacing,
        origin_mm=volume.origin_mm,
        direction=volume.direction,
        source=f"{volume.source}+resampled",
    )


def preprocess_study(study: StudyData, config: Any) -> StudyData:
    clip_range = _config_get(config, "preprocessing.clip_range_hu", [-1000, 2500])
    target_spacing = _config_get(config, "preprocessing.target_spacing_mm", (1.0, 1.0, 1.0))

    clipped_array = np.clip(study.hu_volume.astype(np.float32), float(clip_range[0]), float(clip_range[1]))
    localized_input = StudyData(
        volume=clipped_array.astype(np.float32),
        hu_volume=clipped_array.astype(np.float32),
        spacing_zyx=study.spacing_zyx,
        origin_xyz=study.origin_xyz,
        direction=study.direction,
        metadata=dict(study.metadata),
        normalized_volume=study.normalized_volume,
        source_dir=study.source_dir,
    )
    localized_study = _select_leg_crop_bounds(localized_input, config)

    ct_volume = CTVolume(
        array=localized_study.hu_volume.astype(np.float32),
        spacing_mm=localized_study.spacing_zyx,
        origin_mm=(localized_study.origin_xyz[2], localized_study.origin_xyz[1], localized_study.origin_xyz[0]),
        direction=localized_study.direction,
        source=localized_study.metadata.get("loader", "StudyData"),
    )
    clipped = clip_hu(ct_volume, lower_hu=float(clip_range[0]), upper_hu=float(clip_range[1]))
    resampled = resample_isotropic(clipped, target_spacing_mm=target_spacing)
    normalized = normalize_hu(resampled, lower_hu=float(clip_range[0]), upper_hu=float(clip_range[1]))

    metadata = dict(localized_study.metadata)
    metadata["preprocessing"] = {
        "clip_range_hu": list(clip_range),
        "target_spacing_mm": list(target_spacing),
    }
    return StudyData(
        volume=resampled.array.astype(np.float32),
        hu_volume=resampled.array.astype(np.float32),
        spacing_zyx=resampled.spacing_zyx,
        origin_xyz=resampled.origin_xyz,
        direction=tuple(float(v) for v in resampled.direction),
        metadata=metadata,
        normalized_volume=normalized.astype(np.float32),
        source_dir=localized_study.source_dir,
    )
