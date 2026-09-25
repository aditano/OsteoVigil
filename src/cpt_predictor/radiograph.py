"""Cortical-index strength from one or two plain radiographs.

A single view assumes a circular cross-section. AP and lateral together use an
elliptical annulus. Both are compared with a digitally reconstructed radiograph
of the normal tib/fib segment of the same length.

The shaft outline follows cortical crests, not the soft-tissue silhouette.
Hardware brighter than cortex is removed before that outline is traced. An
unresolved medullary canal is given a thin cortex instead of the area of a
solid rod. A stronger-than-normal claim still has to clear a separate check:
resolved canal, calibrated spacing, and an adult tibial outer diameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .comparison import project_views
from .errors import StrengthAnalysisError
from .reference_leg import build_reference_volume


CORTICAL_YIELD_MPA = 120.0
ADULT_TIBIA_OUTER_MIN_MM = 20.0
ADULT_TIBIA_OUTER_MAX_MM = 35.0
STRONGER_CLAIM_LIMIT_PERCENT = 15.0
CONSERVATIVE_CORTEX_MM = 2.0
CANAL_RESOLVED_FRACTION = 0.70
UNRELIABLE_STRONGER_NOTE = (
    "Measurement unreliable. Cannot claim stronger than normal from this radiograph."
)

# Display rasters are one float32 channel.
# [0, 1] is the windowed radiograph outside bone.
# Bone is offset by 2: floor(value - 2) / 10 is cortical weakness from stronger to weaker,
# and the fraction is the windowed radiograph at that pixel.
BONE_CODE_OFFSET = 2.0
DISPLAY_MAX_EDGE = 640


@dataclass
class ShaftMeasure:
    relative: np.ndarray
    area_mm2: np.ndarray
    outer_mm: np.ndarray
    inner_mm: np.ndarray
    length_mm: float
    spacing_mm: tuple[float, float]
    axis: str
    canal_resolved: np.ndarray
    line_index: np.ndarray
    mask: np.ndarray


def blocks_stronger_claim(
    percent_stronger: float,
    *,
    canal_resolved: bool,
    outer_diameters_mm: list[float],
    spacing_assumed: bool,
    reference_resolved: bool = True,
) -> bool:
    """Return True when a stronger-than-normal radiograph claim must be withheld.

    Claims at or below the limit stay available. Larger claims need a resolved
    canal on the patient and on the reference projection, spacing that came
    from a DICOM tag, and a tibial outer diameter in the adult midshaft range.
    """
    if float(percent_stronger) <= STRONGER_CLAIM_LIMIT_PERCENT:
        return False
    if spacing_assumed or not canal_resolved or not reference_resolved or not outer_diameters_mm:
        return True
    return any(
        float(diameter) < ADULT_TIBIA_OUTER_MIN_MM or float(diameter) > ADULT_TIBIA_OUTER_MAX_MM
        for diameter in outer_diameters_mm
    )


def _finite_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    finite = np.isfinite(values)
    if int(finite.sum()) < 16:
        raise StrengthAnalysisError("The radiograph has too few pixels to measure.", modality="radiograph")
    if not np.all(finite):
        values = values.copy()
        values[~finite] = float(np.median(values[finite]))
    return values


def _suppress_metal(image: np.ndarray) -> np.ndarray:
    """Drop a small saturated tail so labels and hardware are not the shaft edge."""
    finite = image[np.isfinite(image)]
    low, high = np.percentile(finite, [98.0, 99.9])
    capped = image.copy()
    if high <= max(float(low) * 1.4, float(low) + 30.0):
        return capped
    limit = float(low) * 1.15 if low > 0 else float(low) + 30.0
    metal = capped > limit
    if float(np.mean(metal)) >= 0.02 or not np.any(~metal):
        return capped
    capped[metal] = float(np.median(capped[~metal]))
    return capped


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, count = ndi.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def _shaft_along_rows(image: np.ndarray) -> bool:
    """Long axis of the bright shaft, not of the brightest speckle inside it."""
    low, high = np.percentile(image, [20.0, 99.0])
    if high <= low + 1.0e-6:
        return image.shape[0] >= image.shape[1]
    level = float(low + 0.45 * (high - low))
    bright = image >= level
    if int(bright.sum()) < 16:
        return image.shape[0] >= image.shape[1]
    bone = _largest_component(bright)
    rows = int(np.count_nonzero(np.any(bone, axis=1)))
    cols = int(np.count_nonzero(np.any(bone, axis=0)))
    return rows >= cols


def _conservative_inner(outer_mm: float) -> float:
    thickness = min(CONSERVATIVE_CORTEX_MM, max(0.6, 0.08 * float(outer_mm)))
    return max(0.0, float(outer_mm) - 2.0 * thickness)


def _runs_above(smooth: np.ndarray, level: float) -> list[tuple[int, int]]:
    indices = np.flatnonzero(smooth >= level)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks, indices.size - 1]
    return [(int(indices[start]), int(indices[stop])) for start, stop in zip(starts, stops)]


def _canal_inner(smooth: np.ndarray, left: int, right: int, outer_mm: float, pixel_mm: float) -> tuple[float, bool]:
    if right <= left + 1:
        return _conservative_inner(outer_mm), False
    segment = smooth[left : right + 1]
    half = max(1, segment.size // 2)
    left_peak = int(np.argmax(segment[:half]))
    right_peak = half + int(np.argmax(segment[half:]))
    if right_peak <= left_peak + 1:
        return _conservative_inner(outer_mm), False
    valley = float(np.min(segment[left_peak : right_peak + 1]))
    lower = float(min(segment[left_peak], segment[right_peak]))
    contrast = (lower - valley) / max(lower, 1.0e-6)
    if contrast < 0.08:
        return _conservative_inner(outer_mm), False
    level = 0.5 * (lower + valley)
    inner_left = left_peak
    for index in range(left_peak, right_peak):
        if float(segment[index]) <= level:
            inner_left = index
            break
    inner_right = right_peak
    for index in range(right_peak, left_peak, -1):
        if float(segment[index]) <= level:
            inner_right = index
            break
    inner = max(0.0, float(inner_right - inner_left) * pixel_mm)
    if inner <= 1.0 or inner >= outer_mm * 0.96:
        return _conservative_inner(outer_mm), False
    return min(inner, outer_mm * 0.95), True


def _section_from_line(profile: np.ndarray, pixel_mm: float) -> tuple[float, float, bool, int, int] | None:
    """Return outer mm, inner mm, canal resolved, and the column span on this line.

    Cortical pixels are the bright runs. A single wide run is one shaft whose
    canal may sit inside it. Two runs are the two walls of one bone. The
    brightest plausible shaft is kept, so a second bone or a metal edge does
    not set the outer diameter.
    """
    raw = np.asarray(profile, dtype=float)
    if raw.size < 6:
        return None
    sigma = float(np.clip(0.8 / max(pixel_mm, 1.0e-3), 0.8, 10.0))
    smooth = ndi.gaussian_filter1d(raw, sigma)
    peak = float(np.max(smooth))
    if peak <= 1.0e-6:
        return None
    min_px = max(1, int(round(0.6 / pixel_mm)))
    runs = [run for run in _runs_above(smooth, 0.70 * peak) if run[1] - run[0] + 1 >= min_px]
    if not runs:
        return None
    candidates: list[tuple[float, float, int, int]] = []
    for index, (start, stop) in enumerate(runs):
        outer = float(stop - start + 1) * pixel_mm
        height = float(np.max(smooth[start : stop + 1]))
        if 8.0 <= outer <= 42.0:
            candidates.append((height, outer, start, stop))
        for next_start, next_stop in runs[index + 1 :]:
            outer = float(next_stop - start + 1) * pixel_mm
            # Adult tibial outer diameter stays under this. A tibia-fibula span does not.
            if outer < 8.0 or outer > 36.0:
                continue
            if next_start <= stop:
                continue
            valley = float(np.min(smooth[stop : next_start + 1]))
            right_height = float(np.max(smooth[next_start : next_stop + 1]))
            lower = min(height, right_height)
            # A medullary canal is a deep valley that is still above background.
            # A gap that falls to air is the space between two bones, not one canal.
            if valley <= max(0.18 * peak, 0.22 * lower):
                continue
            candidates.append((lower, outer, start, next_stop))
    if not candidates:
        start, stop = max(runs, key=lambda run: run[1] - run[0])
        left, right = _expand_bone(smooth, start, stop, peak, pixel_mm)
        outer = float(right - left + 1) * pixel_mm
        if outer < 4.0:
            return None
        return outer, _conservative_inner(outer), False, left, right
    _height, _outer, core_left, core_right = max(candidates, key=lambda item: (item[0], item[1]))
    left, right = _expand_bone(smooth, core_left, core_right, peak, pixel_mm)
    outer = float(right - left + 1) * pixel_mm
    if outer > 48.0:
        return outer, _conservative_inner(outer), False, left, right
    inner, resolved = _canal_inner(smooth, left, right, outer, pixel_mm)
    return outer, inner, resolved, left, right


def _expand_bone(smooth: np.ndarray, left: int, right: int, peak: float, pixel_mm: float) -> tuple[int, int]:
    """Follow the cortical falloff a short distance, stopping at soft tissue or air."""
    return (
        _expand_edge(smooth, left, -1, peak, pixel_mm),
        _expand_edge(smooth, right, 1, peak, pixel_mm),
    )


def _expand_edge(smooth: np.ndarray, start: int, direction: int, peak: float, pixel_mm: float) -> int:
    """Stop at the periosteal gradient instead of walking into soft tissue.

    The cortical core is already the bright run. The outer cortex is the first
    steep falloff beyond that run. A shallow tail (soft tissue, cast, or blur)
    is not added to the diameter.
    """
    index = int(start)
    limit = 0 if direction < 0 else int(smooth.size) - 1
    max_steps = max(1, int(round(3.0 / max(pixel_mm, 1.0e-3))))
    best_index = index
    best_drop = 0.0
    for _step in range(max_steps):
        if index == limit:
            break
        nxt = index + direction
        current = float(smooth[index])
        nxt_value = float(smooth[nxt])
        drop = current - nxt_value
        if drop < -0.04 * peak and (peak - current) > 0.12 * peak:
            break
        if drop > best_drop:
            best_drop = drop
            best_index = nxt
        if nxt_value < 0.18 * peak:
            return nxt
        if best_drop > 0.025 * peak and drop < 0.45 * best_drop:
            return best_index
        index = nxt
    return best_index


def _longest_contiguous(records: list[dict[str, float | int | bool]]) -> list[dict[str, float | int | bool]]:
    if not records:
        return []
    best_start = 0
    best_length = 1
    start = 0
    for index in range(1, len(records)):
        if int(records[index]["index"]) > int(records[index - 1]["index"]) + 1:
            length = index - start
            if length > best_length:
                best_start = start
                best_length = length
            start = index
    tail = len(records) - start
    if tail > best_length:
        best_start = start
        best_length = tail
    return records[best_start : best_start + best_length]


def _dominant_shaft(records: list[dict[str, float | int | bool]]) -> list[dict[str, float | int | bool]]:
    if not records:
        return []
    outers = np.asarray([float(record["outer"]) for record in records], dtype=float)
    median = float(np.median(outers))
    tolerance = max(6.0, 0.35 * median)
    kept = [record for record in records if abs(float(record["outer"]) - median) <= tolerance]
    if len(kept) < max(4, int(0.4 * len(records))):
        kept = records
    return _longest_contiguous(kept)


def _section_area(outer_mm: float, inner_mm: float) -> float:
    return float(0.25 * np.pi * max(outer_mm * outer_mm - inner_mm * inner_mm, 0.0))


def measure_shaft(image: np.ndarray, spacing_rc_mm: tuple[float, float]) -> ShaftMeasure:
    values = _suppress_metal(_finite_image(image))
    along_rows = _shaft_along_rows(values)
    row_spacing, col_spacing = (float(spacing_rc_mm[0]), float(spacing_rc_mm[1]))
    if along_rows:
        pixel_mm = col_spacing
        step_mm = row_spacing
        bright = values >= float(np.percentile(values, 90))
        indices = np.flatnonzero(np.any(bright, axis=1))
        lines = ((int(index), values[index]) for index in indices)
    else:
        pixel_mm = row_spacing
        step_mm = col_spacing
        bright = values >= float(np.percentile(values, 90))
        indices = np.flatnonzero(np.any(bright, axis=0))
        lines = ((int(index), values[:, index]) for index in indices)

    records: list[dict[str, float | int | bool]] = []
    for index, line in lines:
        section = _section_from_line(line, pixel_mm)
        if section is None:
            continue
        outer, inner, resolved, left, right = section
        records.append(
            {
                "index": index,
                "outer": outer,
                "inner": inner,
                "resolved": resolved,
                "left": left,
                "right": right,
                "area": _section_area(outer, inner),
            }
        )
    records = _dominant_shaft(records)
    if len(records) < 4:
        raise StrengthAnalysisError("The radiograph shaft is too short to measure a cortical index.", modality="radiograph")

    outer_arr = np.asarray([float(record["outer"]) for record in records], dtype=float)
    inner_arr = np.asarray([float(record["inner"]) for record in records], dtype=float)
    area_arr = np.asarray([float(record["area"]) for record in records], dtype=float)
    resolved_arr = np.asarray([bool(record["resolved"]) for record in records], dtype=bool)
    line_index = np.asarray([int(record["index"]) for record in records], dtype=int)
    end = max(1, outer_arr.size // 10)
    if float(np.mean(outer_arr[-end:])) > 1.15 * float(np.mean(outer_arr[:end])):
        outer_arr = outer_arr[::-1]
        inner_arr = inner_arr[::-1]
        area_arr = area_arr[::-1]
        resolved_arr = resolved_arr[::-1]
        line_index = line_index[::-1]
        records = list(reversed(records))

    mask = np.zeros(values.shape, dtype=bool)
    for record in records:
        left = int(record["left"])
        right = int(record["right"]) + 1
        index = int(record["index"])
        if along_rows:
            mask[index, left:right] = True
        else:
            mask[left:right, index] = True
    relative = (np.arange(area_arr.size) + 0.5) / float(area_arr.size)
    return ShaftMeasure(
        relative=relative,
        area_mm2=area_arr,
        outer_mm=outer_arr,
        inner_mm=inner_arr,
        length_mm=float(area_arr.size) * step_mm,
        spacing_mm=(step_mm, pixel_mm),
        axis="row" if along_rows else "col",
        canal_resolved=resolved_arr,
        line_index=line_index,
        mask=mask,
    )


def _midshaft_window(count: int) -> slice:
    start = int(0.15 * count)
    stop = max(start + 1, int(0.85 * count))
    return slice(start, stop)


def _failure_from_profile(areas: np.ndarray, resolved: np.ndarray) -> tuple[float, bool]:
    window = _midshaft_window(int(areas.size))
    section = np.asarray(areas[window], dtype=float)
    flags = np.asarray(resolved[window], dtype=bool)
    if section.size == 0:
        section = np.asarray(areas, dtype=float)
        flags = np.asarray(resolved, dtype=bool)
    trusted = bool(flags.size and float(np.mean(flags)) >= CANAL_RESOLVED_FRACTION and int(flags.sum()) >= 3)
    chosen = section[flags] if trusted else section
    return CORTICAL_YIELD_MPA * float(np.min(chosen)), trusted


def _resample_flags(source: ShaftMeasure, relative: np.ndarray) -> np.ndarray:
    if source.canal_resolved.size == 0:
        return np.zeros(relative.shape, dtype=bool)
    positions = np.clip(np.searchsorted(source.relative, relative, side="left"), 0, source.relative.size - 1)
    previous = np.clip(positions - 1, 0, source.relative.size - 1)
    use_previous = np.abs(source.relative[previous] - relative) < np.abs(source.relative[positions] - relative)
    positions = np.where(use_previous, previous, positions)
    return source.canal_resolved[positions]


def elliptical_areas(primary: ShaftMeasure, secondary: ShaftMeasure) -> np.ndarray:
    outer_b = np.interp(primary.relative, secondary.relative, secondary.outer_mm)
    inner_b = np.interp(primary.relative, secondary.relative, secondary.inner_mm)
    outer = np.maximum(primary.outer_mm * outer_b, 0.0)
    inner = np.minimum(primary.inner_mm * inner_b, outer)
    return 0.25 * np.pi * np.maximum(outer - inner, 0.0)


def _combined_profile(primary: ShaftMeasure, secondary: ShaftMeasure | None) -> tuple[np.ndarray, np.ndarray]:
    if secondary is None:
        return primary.area_mm2, primary.canal_resolved
    return elliptical_areas(primary, secondary), primary.canal_resolved & _resample_flags(secondary, primary.relative)


def _mid_median(values: np.ndarray) -> float:
    return float(np.median(values[_midshaft_window(int(values.size))]))


def project_reference(length_mm: float, spacing_mm: float, view: str) -> tuple[np.ndarray, tuple[float, float]]:
    volume, spacing = build_reference_volume(length_mm, spacing_mm)
    density = np.clip(volume, 0.0, None)
    if view == "lateral":
        image = density.sum(axis=2) * spacing[2]
        return image.astype(np.float32), (spacing[0], spacing[1])
    image = density.sum(axis=1) * spacing[1]
    return image.astype(np.float32), (spacing[0], spacing[2])


def _paint(measure: ShaftMeasure, weakness_1d: np.ndarray) -> np.ndarray:
    painted = np.zeros(measure.mask.shape, dtype=np.float32)
    count = min(int(measure.line_index.size), int(weakness_1d.size))
    for slot in range(count):
        index = int(measure.line_index[slot])
        value = np.float32(weakness_1d[slot])
        if measure.axis == "row":
            painted[index, measure.mask[index]] = value
        else:
            painted[measure.mask[:, index], index] = value
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
    overlay = _paint(measure, weakness_1d)
    mask = measure.mask
    shown = gray.copy()
    weakness_t = np.clip((overlay + 1.0) * 0.5, 0.0, 1.0)
    steps = np.rint(weakness_t * 10.0)
    shown[mask] = (BONE_CODE_OFFSET + steps[mask] + gray[mask]).astype(np.float32)
    return _display_stride(shown)


def _view_audit(measure: ShaftMeasure, spacing_rc_mm: tuple[float, float], trusted: bool) -> dict[str, object]:
    return {
        "outer_diameter_mm": _mid_median(measure.outer_mm),
        "inner_diameter_mm": _mid_median(measure.inner_mm),
        "inner_estimated": not trusted,
        "canal_resolved": trusted,
        "spacing_mm": [float(spacing_rc_mm[0]), float(spacing_rc_mm[1])],
    }


def analyze_radiographs(
    views: dict[str, tuple[np.ndarray, tuple[float, float]]],
) -> dict[str, object]:
    if not views:
        raise StrengthAnalysisError("No radiograph pixels were loaded.", modality="radiograph")
    measured = {name: measure_shaft(image, spacing) for name, (image, spacing) in views.items()}
    primary_name = "ap" if "ap" in measured else next(iter(measured))
    primary = measured[primary_name]
    circular = len(measured) == 1
    other_name = None if circular else ("lateral" if primary_name == "ap" else "ap")
    secondary = None if other_name is None else measured[other_name]
    if circular:
        assumption = "One radiographic view was used, so the cross-section is assumed circular."
    else:
        assumption = "AP and lateral views were combined as an elliptical cortical annulus."
    patient_areas, patient_flags = _combined_profile(primary, secondary)
    patient_failure_n, canal_resolved = _failure_from_profile(patient_areas, patient_flags)

    length = float(np.median([item.length_mm for item in measured.values()]))
    spacing = float(np.median([item.spacing_mm[0] for item in measured.values()]))
    reference_measured: dict[str, ShaftMeasure] = {}
    for name in measured:
        ref_image, ref_spacing = project_reference(max(length, spacing * 8), max(spacing, 0.5), name)
        reference_measured[name] = measure_shaft(ref_image, ref_spacing)
    ref_primary = reference_measured[primary_name]
    ref_secondary = None if other_name is None else reference_measured[other_name]
    ref_areas, ref_flags = _combined_profile(ref_primary, ref_secondary)
    reference_failure_n, reference_resolved = _failure_from_profile(ref_areas, ref_flags)

    ref_at_patient = np.interp(primary.relative, ref_primary.relative, ref_areas)
    ref_at_patient = np.maximum(ref_at_patient, 1.0e-3)
    weakness_1d = 1.0 - (patient_areas / ref_at_patient)

    rasters = {}
    for name, (image, _spacing) in views.items():
        profile = weakness_1d
        if name != primary_name:
            profile = np.interp(measured[name].relative, primary.relative, weakness_1d)
        rasters[name] = display_raster(image, measured[name], profile)

    hotspot = None
    if weakness_1d.size:
        hotspot = float((int(np.argmax(weakness_1d)) + 0.5) / weakness_1d.size * primary.length_mm)
    view_measures = {
        name: _view_audit(measure, views[name][1], bool(_failure_from_profile(measure.area_mm2, measure.canal_resolved)[1]))
        for name, measure in measured.items()
    }
    return {
        "failure_load_n": float(patient_failure_n),
        "reference_failure_load_n": float(reference_failure_n),
        "cortical_area_mm2": float(patient_failure_n / CORTICAL_YIELD_MPA),
        "reference_cortical_area_mm2": float(reference_failure_n / CORTICAL_YIELD_MPA),
        "canal_resolved": bool(canal_resolved),
        "reference_resolved": bool(reference_resolved),
        "outer_diameters_mm": [float(item["outer_diameter_mm"]) for item in view_measures.values()],
        "view_measures": view_measures,
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
