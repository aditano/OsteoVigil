"""Cortical-index strength from one or two plain radiographs.

A single view assumes a circular cross-section. AP and lateral together use an
elliptical annulus. Both are compared with a digitally reconstructed radiograph
of the normal tib/fib segment of the same length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .comparison import project_views
from .errors import StrengthAnalysisError
from .reference_leg import build_reference_volume


CORTICAL_YIELD_MPA = 120.0


@dataclass
class ShaftMeasure:
    relative: np.ndarray
    area_mm2: np.ndarray
    outer_mm: np.ndarray
    inner_mm: np.ndarray
    length_mm: float
    spacing_mm: tuple[float, float]
    axis: str


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndi.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def bone_mask(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    finite = np.isfinite(values)
    if int(finite.sum()) < 16:
        raise StrengthAnalysisError("The radiograph has too few pixels to measure.", modality="radiograph")
    low, high = np.percentile(values[finite], [15, 99])
    if high <= low + 1.0e-6:
        raise StrengthAnalysisError("The radiograph has no contrast between bone and background.", modality="radiograph")
    threshold = low + 0.30 * (high - low)
    mask = _largest_component(finite & (values >= threshold))
    if int(mask.sum()) < 16:
        raise StrengthAnalysisError("Cortical bone could not be outlined on the radiograph.", modality="radiograph")
    return mask


def _runs(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks, indices.size - 1]
    return [indices[start : stop + 1] for start, stop in zip(starts, stops)]


def _section_widths(profile: np.ndarray, pixel_mm: float) -> tuple[float, float]:
    outer = float(profile.size) * float(pixel_mm)
    if profile.size < 4:
        return outer, 0.0
    smoothed = np.convolve(profile, [0.25, 0.5, 0.25], mode="same")
    mid = smoothed.size // 2
    left = int(np.argmax(smoothed[: max(mid, 1)]))
    right = mid + int(np.argmax(smoothed[mid:]))
    if right <= left + 1:
        return outer, 0.0
    valley = float(np.min(smoothed[left : right + 1]))
    peak = float(min(smoothed[left], smoothed[right]))
    if valley > 0.93 * max(peak, 1.0e-6):
        return outer, 0.0
    level = 0.5 * (peak + valley)
    inner_left = left
    for index in range(left, right):
        if smoothed[index] <= level:
            inner_left = index
            break
    inner_right = right
    for index in range(right, left, -1):
        if smoothed[index] <= level:
            inner_right = index
            break
    inner = max(0.0, float(inner_right - inner_left) * float(pixel_mm))
    return outer, min(inner, outer * 0.95)


def measure_shaft(image: np.ndarray, spacing_rc_mm: tuple[float, float]) -> ShaftMeasure:
    values = np.asarray(image, dtype=float)
    mask = bone_mask(values)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_span = int(np.count_nonzero(rows))
    col_span = int(np.count_nonzero(cols))
    along_rows = row_span >= col_span
    row_spacing, col_spacing = (float(spacing_rc_mm[0]), float(spacing_rc_mm[1]))
    if along_rows:
        indices = np.flatnonzero(rows)
        pixel_mm = col_spacing
        step_mm = row_spacing
        profiles = [(int(index), values[index], np.flatnonzero(mask[index])) for index in indices]
    else:
        indices = np.flatnonzero(cols)
        pixel_mm = row_spacing
        step_mm = col_spacing
        profiles = [(int(index), values[:, index], np.flatnonzero(mask[:, index])) for index in indices]

    outers: list[float] = []
    inners: list[float] = []
    areas: list[float] = []
    for _index, line, bone_idx in profiles:
        runs = _runs(bone_idx)
        if not runs:
            continue
        outer_sum = 0.0
        inner_sum = 0.0
        area_sum = 0.0
        for run in runs:
            if run.size < 2:
                continue
            outer, inner = _section_widths(line[run], pixel_mm)
            outer_sum += outer
            inner_sum += inner
            area_sum += 0.25 * np.pi * max(outer * outer - inner * inner, 0.0)
        if outer_sum <= 0:
            continue
        outers.append(outer_sum)
        inners.append(inner_sum)
        areas.append(area_sum)

    if len(areas) < 4:
        raise StrengthAnalysisError("The radiograph shaft is too short to measure a cortical index.", modality="radiograph")

    outer_arr = np.asarray(outers, dtype=float)
    end = max(1, outer_arr.size // 10)
    if float(np.mean(outer_arr[-end:])) > 1.15 * float(np.mean(outer_arr[:end])):
        outer_arr = outer_arr[::-1]
        inners = inners[::-1]
        areas = areas[::-1]
        inner_arr = np.asarray(inners, dtype=float)
        area_arr = np.asarray(areas, dtype=float)
    else:
        inner_arr = np.asarray(inners, dtype=float)
        area_arr = np.asarray(areas, dtype=float)
    relative = (np.arange(area_arr.size) + 0.5) / float(area_arr.size)
    return ShaftMeasure(
        relative=relative,
        area_mm2=area_arr,
        outer_mm=outer_arr,
        inner_mm=inner_arr,
        length_mm=float(area_arr.size) * step_mm,
        spacing_mm=(step_mm, pixel_mm),
        axis="row" if along_rows else "col",
    )


def _midshaft_failure(areas: np.ndarray) -> float:
    start = int(0.15 * areas.size)
    stop = max(start + 1, int(0.85 * areas.size))
    weakest = float(np.min(areas[start:stop]))
    return CORTICAL_YIELD_MPA * weakest


def elliptical_areas(primary: ShaftMeasure, secondary: ShaftMeasure) -> np.ndarray:
    outer_b = np.interp(primary.relative, secondary.relative, secondary.outer_mm)
    inner_b = np.interp(primary.relative, secondary.relative, secondary.inner_mm)
    outer = np.maximum(primary.outer_mm * outer_b, 0.0)
    inner = np.minimum(primary.inner_mm * inner_b, outer)
    return 0.25 * np.pi * np.maximum(outer - inner, 0.0)


def project_reference(length_mm: float, spacing_mm: float, view: str) -> tuple[np.ndarray, tuple[float, float]]:
    volume, spacing = build_reference_volume(length_mm, spacing_mm)
    density = np.clip(volume, 0.0, None)
    if view == "lateral":
        image = density.sum(axis=2) * spacing[2]
        return image.astype(np.float32), (spacing[0], spacing[1])
    image = density.sum(axis=1) * spacing[1]
    return image.astype(np.float32), (spacing[0], spacing[2])


# Display rasters are one float32 channel.
# [0, 1] is the windowed radiograph outside bone.
# Bone is offset by 2: floor(value - 2) / 10 is cortical weakness from stronger to weaker,
# and the fraction is the windowed radiograph at that pixel.
BONE_CODE_OFFSET = 2.0
DISPLAY_MAX_EDGE = 640


def _paint(image: np.ndarray, measure: ShaftMeasure, weakness_1d: np.ndarray) -> np.ndarray:
    mask = bone_mask(image)
    painted = np.zeros(np.asarray(image).shape, dtype=np.float32)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if measure.axis == "row":
        selected = np.flatnonzero(rows)
        if weakness_1d.size != selected.size and weakness_1d.size == measure.area_mm2.size:
            chosen = selected[: weakness_1d.size] if selected.size >= weakness_1d.size else selected
        else:
            chosen = selected
        count = min(chosen.size, weakness_1d.size)
        for slot, row in enumerate(chosen[:count]):
            painted[row, mask[row]] = np.float32(weakness_1d[slot])
    else:
        selected = np.flatnonzero(cols)
        count = min(selected.size, weakness_1d.size)
        for slot, col in enumerate(selected[:count]):
            painted[mask[:, col], col] = np.float32(weakness_1d[slot])
    return painted


def _window_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(values)
    shown = np.zeros(values.shape, dtype=np.float32)
    if not np.any(finite):
        return shown
    low, high = np.percentile(values[finite], [1.0, 99.0])
    span = float(high - low)
    if span <= 1.0e-6:
        shown[finite] = 0.5
        return shown
    shown[finite] = np.clip((values[finite] - np.float32(low)) / np.float32(span), 0.0, 1.0)
    return shown


def _display_stride(image: np.ndarray, max_edge: int = DISPLAY_MAX_EDGE) -> np.ndarray:
    height, width = image.shape
    longest = max(int(height), int(width))
    if longest <= max_edge:
        return np.ascontiguousarray(image, dtype=np.float32)
    step = int(np.ceil(longest / float(max_edge)))
    return np.ascontiguousarray(image[::step, ::step], dtype=np.float32)


def display_raster(image: np.ndarray, measure: ShaftMeasure, weakness_1d: np.ndarray) -> np.ndarray:
    """Windowed radiograph. Cortical pixels also carry the weakness code above 2."""
    gray = np.minimum(_window_image(image), np.float32(0.99))
    overlay = _paint(image, measure, weakness_1d)
    mask = bone_mask(image)
    shown = gray.copy()
    weakness_t = np.clip((overlay + 1.0) * 0.5, 0.0, 1.0)
    steps = np.rint(weakness_t * 10.0)
    shown[mask] = (BONE_CODE_OFFSET + steps[mask] + gray[mask]).astype(np.float32)
    return _display_stride(shown)


def analyze_radiographs(
    views: dict[str, tuple[np.ndarray, tuple[float, float]]],
) -> dict[str, object]:
    if not views:
        raise StrengthAnalysisError("No radiograph pixels were loaded.", modality="radiograph")
    measured = {name: measure_shaft(image, spacing) for name, (image, spacing) in views.items()}
    primary_name = "ap" if "ap" in measured else next(iter(measured))
    primary = measured[primary_name]
    circular = len(measured) == 1
    if circular:
        areas = primary.area_mm2
        assumption = "One radiographic view was used, so the cross-section is assumed circular."
    else:
        other_name = "lateral" if primary_name == "ap" else "ap"
        areas = elliptical_areas(primary, measured[other_name])
        assumption = "AP and lateral views were combined as an elliptical cortical annulus."

    length = float(np.median([item.length_mm for item in measured.values()]))
    spacing = float(np.median([item.spacing_mm[0] for item in measured.values()]))
    reference_areas = {}
    for name in measured:
        ref_image, ref_spacing = project_reference(max(length, spacing * 8), max(spacing, 0.5), name)
        reference_areas[name] = measure_shaft(ref_image, ref_spacing)
    if circular:
        ref_failure = _midshaft_failure(reference_areas[primary_name].area_mm2)
        ref_profile = reference_areas[primary_name]
        patient_profile_area = areas
    else:
        ref_primary = reference_areas[primary_name]
        ref_other_name = "lateral" if primary_name == "ap" else "ap"
        ref_areas = elliptical_areas(ref_primary, reference_areas[ref_other_name])
        ref_failure = _midshaft_failure(ref_areas)
        ref_profile = ref_primary
        ref_profile = ShaftMeasure(
            relative=ref_primary.relative,
            area_mm2=ref_areas,
            outer_mm=ref_primary.outer_mm,
            inner_mm=ref_primary.inner_mm,
            length_mm=ref_primary.length_mm,
            spacing_mm=ref_primary.spacing_mm,
            axis=ref_primary.axis,
        )
        patient_profile_area = areas

    failure = _midshaft_failure(patient_profile_area)
    ref_at_patient = np.interp(primary.relative, ref_profile.relative, ref_profile.area_mm2)
    ref_at_patient = np.maximum(ref_at_patient, 1.0e-3)
    weakness_1d = 1.0 - (patient_profile_area / ref_at_patient)

    rasters = {}
    for name, (image, _spacing) in views.items():
        profile = weakness_1d
        if name != primary_name:
            profile = np.interp(measured[name].relative, primary.relative, weakness_1d)
        rasters[name] = display_raster(image, measured[name], profile)

    hotspot = None
    if weakness_1d.size:
        hotspot = float((int(np.argmax(weakness_1d)) + 0.5) / weakness_1d.size * primary.length_mm)
    return {
        "failure_load_n": float(failure),
        "reference_failure_load_n": float(ref_failure),
        "bone_length_mm": length,
        "assumption": assumption,
        "rasters": rasters,
        "hotspot_z_mm": hotspot,
        "tibia_voxels": int(sum(int(np.asarray(image).size) for image, _spacing in views.values())),
        "fibula_voxels": 0,
    }


def raster_volume(rasters: dict[str, np.ndarray]) -> np.ndarray | None:
    """Pack a single acquired view so the same AP/lateral projector can display it."""
    if "ap" in rasters and "lateral" not in rasters:
        image = rasters["ap"]
        return image[:, None, :]
    if "lateral" in rasters and "ap" not in rasters:
        image = rasters["lateral"]
        return image[:, :, None]
    return None


def projection_pair_from_rasters(rasters: dict[str, np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None]:
    volume = raster_volume(rasters)
    if volume is None:
        return rasters.get("ap"), rasters.get("lateral")
    ap, lateral = project_views(volume)
    if "ap" not in rasters:
        ap = None
    if "lateral" not in rasters:
        lateral = None
    return ap, lateral
