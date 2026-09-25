"""Run tib/fib vertical-load analysis for CT, MRI, or radiographs."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import block_reduce

from .comparison import (
    RESEARCH_BANNER,
    field_of_view_note,
    hotspot_height_mm,
    project_views,
    strength_comparison,
    walking_assessment,
    weakness_volume,
)
from .errors import StrengthAnalysisError
from .materials import apparent_density_g_cm3, yield_strength_from_density, youngs_modulus_from_density
from .mri_geometry import UNIFORM_CORTEX_MODULUS_MPA, UNIFORM_CORTEX_YIELD_MPA, segment_mri_cortex
from .radiograph import (
    MISSING_SPACING_NOTE,
    UNRELIABLE_STRONGER_NOTE,
    UNRESOLVED_CANAL_NOTE,
    analyze_radiographs,
    blocks_stronger_claim,
    projection_pair_from_rasters,
)
from .reference_leg import (
    CORTICAL_HU,
    FIBULA_CORTEX_MM,
    FIBULA_OFFSET_MM,
    FIBULA_RADIUS_MM,
    TIBIA_CORTEX_MM,
    TIBIA_RADIUS_MM,
    build_reference_volume,
    synthetic_tibfib_volume,
)
from .segmentation import TibFibMasks, segment_tibia_and_fibula
from .strength_io import load_dicom_path
from .voxel_fea import VoxelSolveResult, body_weight_newtons, solve_voxel_elasticity


@dataclass
class StrengthReport:
    modality: str
    method: str
    body_mass_kg: float
    body_weight_n: float
    failure_load_n: float
    failure_load_bodyweights: float
    reference_failure_load_n: float
    percent_vs_normal: float
    signed_percent_weaker: float
    comparison: str
    walking_verdict: str
    walking_text: str
    field_note: str
    assumptions: list[str]
    hotspot_z_mm: float | None
    bone_length_mm: float
    tibia_voxels: int
    fibula_voxels: int
    solver: str
    research_banner: str = RESEARCH_BANNER
    weakness: np.ndarray | None = None
    weakness_spacing_zyx: tuple[float, float, float] | None = None
    views_acquired: dict[str, bool] = field(default_factory=lambda: {"ap": True, "lateral": True})
    rasters: dict[str, np.ndarray] = field(default_factory=dict)
    measurement_reliable: bool = True
    measurement_note: str = ""
    radiograph_measures: dict[str, Any] | None = None
    body_mass_source: str = "entered"
    unreliable_reason: str = ""

    def public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "modality": self.modality,
            "method": self.method,
            "body_mass_kg": self.body_mass_kg,
            "body_weight_n": self.body_weight_n,
            "failure_load_n": self.failure_load_n,
            "failure_load_bodyweights": self.failure_load_bodyweights,
            "reference_failure_load_n": self.reference_failure_load_n,
            "percent_vs_normal": self.percent_vs_normal,
            "signed_percent_weaker": self.signed_percent_weaker,
            "comparison": self.comparison,
            "walking_verdict": self.walking_verdict,
            "walking_text": self.walking_text,
            "field_note": self.field_note,
            "assumptions": self.assumptions,
            "hotspot_z_mm": self.hotspot_z_mm,
            "bone_length_mm": self.bone_length_mm,
            "tibia_voxels": self.tibia_voxels,
            "fibula_voxels": self.fibula_voxels,
            "solver": self.solver,
            "research_banner": self.research_banner,
            "measurement_reliable": bool(self.measurement_reliable),
            "measurement_note": self.measurement_note,
            "body_mass_source": self.body_mass_source,
            "unreliable_reason": self.unreliable_reason,
            "views": {
                "ap": {"acquired": bool(self.views_acquired.get("ap", False))},
                "lateral": {"acquired": bool(self.views_acquired.get("lateral", False))},
            },
        }
        if self.weakness is not None:
            encoded, shape = _encode_array(self.weakness)
            payload["weakness"] = {
                "shape": shape,
                "spacing_mm": list(self.weakness_spacing_zyx or (1.0, 1.0, 1.0)),
                "encoding": "float32-little",
                "b64": encoded,
            }
        if self.radiograph_measures is not None:
            payload["radiograph_measures"] = self.radiograph_measures
        if self.rasters:
            payload["rasters"] = {}
            for name, image in self.rasters.items():
                encoded, shape = _encode_array(image)
                payload["rasters"][name] = {"shape": shape, "encoding": "float32-little", "b64": encoded}
        return payload


def _encode_array(values: np.ndarray) -> tuple[str, list[int]]:
    array = np.ascontiguousarray(values, dtype=np.float32)
    return base64.b64encode(array.tobytes()).decode("ascii"), [int(v) for v in array.shape]


def _bounds(mask: np.ndarray, margin: int = 0) -> tuple[slice, slice, slice]:
    slices = []
    for axis in range(3):
        other = tuple(index for index in range(3) if index != axis)
        present = np.flatnonzero(np.any(mask, axis=other))
        if present.size == 0:
            raise StrengthAnalysisError("The bone mask is empty.")
        start = max(0, int(present[0]) - margin)
        stop = min(int(mask.shape[axis]), int(present[-1]) + 1 + margin)
        slices.append(slice(start, stop))
    return slices[0], slices[1], slices[2]


def _resample(volume: np.ndarray, spacing_zyx: tuple[float, float, float], target_mm: float) -> tuple[np.ndarray, tuple[float, float, float]]:
    zoom = tuple(float(spacing) / float(target_mm) for spacing in spacing_zyx)
    if all(0.92 <= factor <= 1.08 for factor in zoom):
        return np.asarray(volume, dtype=np.float32), tuple(float(value) for value in spacing_zyx)
    resampled = ndi.zoom(np.asarray(volume, dtype=np.float32), zoom, order=1)
    target = float(target_mm)
    return resampled.astype(np.float32), (target, target, target)


def _downsample_max(
    volume: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    target_mm: float,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Downsample with a block maximum so cortical peaks are not averaged away."""
    block = tuple(max(1, int(round(float(target_mm) / float(spacing)))) for spacing in spacing_zyx)
    if all(size == 1 for size in block):
        return np.asarray(volume, dtype=np.float32), tuple(float(value) for value in spacing_zyx)
    shape = volume.shape
    trimmed = volume[
        : shape[0] - (shape[0] % block[0]),
        : shape[1] - (shape[1] % block[1]),
        : shape[2] - (shape[2] % block[2]),
    ]
    reduced = block_reduce(trimmed, block, np.max)
    spacing = tuple(float(spacing_zyx[axis]) * block[axis] for axis in range(3))
    return np.asarray(reduced, dtype=np.float32), spacing


def _isolate_primary_limb(volume: np.ndarray, spacing_zyx: tuple[float, float, float]) -> np.ndarray:
    """Keep one tib/fib pair when a bilateral field contains two distant limbs."""
    bone = np.asarray(volume) >= 250.0
    labeled, count = ndi.label(bone)
    if count <= 1:
        return volume
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    centroids: list[tuple[int, float, float, float]] = []
    for label in order:
        if sizes[label] < 20:
            break
        zz, yy, xx = np.nonzero(labeled == label)
        centroids.append((int(label), float(yy.mean()), float(xx.mean()), float(zz.mean())))
    if len(centroids) <= 1:
        return volume
    primary = centroids[0]
    limit_y = 45.0 / max(float(spacing_zyx[1]), 1.0e-3)
    limit_x = 45.0 / max(float(spacing_zyx[2]), 1.0e-3)
    nearby = [primary[0]]
    for label, y_mean, x_mean, _z_mean in centroids[1:]:
        if abs(y_mean - primary[1]) <= limit_y and abs(x_mean - primary[2]) <= limit_x:
            nearby.append(label)
    if len(nearby) == len(centroids):
        return volume
    keep = np.isin(labeled, nearby)
    if int(keep.sum()) < 30:
        return volume
    slc = _bounds(keep, margin=2)
    return np.asarray(volume[slc], dtype=np.float32)


def _straighten_shaft(volume: np.ndarray) -> np.ndarray:
    """Remove a linear positioning tilt so axial load follows the shaft."""
    bone = np.asarray(volume) >= 180.0
    rows: list[tuple[int, float, float]] = []
    for index in range(volume.shape[0]):
        yy, xx = np.nonzero(bone[index])
        if yy.size < 15:
            continue
        rows.append((index, float(yy.mean()), float(xx.mean())))
    if len(rows) < 6:
        return volume
    z_index = np.asarray([row[0] for row in rows], dtype=float)
    y_mean = np.asarray([row[1] for row in rows], dtype=float)
    x_mean = np.asarray([row[2] for row in rows], dtype=float)
    slope_y, intercept_y = np.polyfit(z_index, y_mean, 1)
    slope_x, intercept_x = np.polyfit(z_index, x_mean, 1)
    drift = max(abs(float(slope_y)) * volume.shape[0], abs(float(slope_x)) * volume.shape[0])
    if drift < 1.5:
        return volume
    z_mid = 0.5 * (volume.shape[0] - 1)
    target_y = float(slope_y) * z_mid + float(intercept_y)
    target_x = float(slope_x) * z_mid + float(intercept_x)
    air = float(np.percentile(volume, 5))
    straightened = np.empty_like(volume)
    for index in range(volume.shape[0]):
        shift_y = target_y - (float(slope_y) * index + float(intercept_y))
        shift_x = target_x - (float(slope_x) * index + float(intercept_x))
        straightened[index] = ndi.shift(volume[index], shift=(shift_y, shift_x), order=1, cval=air)
    return straightened.astype(np.float32)


def _bone_length_mm(mask: np.ndarray, spacing_z: float) -> float:
    present = np.flatnonzero(np.any(mask, axis=(1, 2)))
    if present.size == 0:
        return 0.0
    return float(present[-1] - present[0] + 1) * float(spacing_z)


def _crop_pair(volume: np.ndarray, masks: TibFibMasks) -> tuple[np.ndarray, TibFibMasks]:
    slc = _bounds(masks.combined, margin=0)
    return np.asarray(volume[slc], dtype=np.float32), TibFibMasks(masks.tibia[slc], masks.fibula[slc])


def _prepare_ct_grid(
    hu: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> tuple[np.ndarray, TibFibMasks, tuple[float, float, float]]:
    volume = np.asarray(hu, dtype=np.float32)
    rough = volume >= 180.0
    if not np.any(rough):
        raise StrengthAnalysisError("No bone-density voxels were found in the CT.", modality="ct")
    slc = _bounds(rough, margin=2)
    volume = volume[slc]
    spacing = tuple(float(value) for value in spacing_zyx)
    volume = _isolate_primary_limb(volume, spacing)
    volume = _straighten_shaft(volume)
    if min(spacing) < 1.7:
        volume, spacing = _downsample_max(volume, spacing, 2.0)
    elif volume.size > 180_000:
        volume, spacing = _resample(volume, spacing, 2.5)
    masks = segment_tibia_and_fibula(volume, spacing)
    voxels = int(masks.combined.sum())
    if voxels > 9000:
        scale = (voxels / 7000.0) ** (1.0 / 3.0)
        volume, spacing = _resample(volume, spacing, spacing[0] * scale)
        masks = segment_tibia_and_fibula(volume, spacing)
        voxels = int(masks.combined.sum())
    if voxels < 30:
        raise StrengthAnalysisError("Tibia and fibula could not be segmented from the CT.", modality="ct")
    if voxels > 14000:
        volume, spacing = _resample(volume, spacing, spacing[0] * 1.35)
        masks = segment_tibia_and_fibula(volume, spacing)
        if int(masks.combined.sum()) < 30:
            raise StrengthAnalysisError("Tibia and fibula could not be segmented from the CT.", modality="ct")
    cropped, cropped_masks = _crop_pair(volume, masks)
    return cropped, cropped_masks, spacing


def _solve_masked(
    modulus: np.ndarray,
    strength: np.ndarray,
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    force_n: float,
) -> VoxelSolveResult:
    try:
        return solve_voxel_elasticity(modulus, strength, mask, spacing_zyx, force_n)
    except (RuntimeError, ValueError) as exc:
        raise StrengthAnalysisError(str(exc)) from exc


def _ct_materials(hu: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    density = apparent_density_g_cm3(hu)
    modulus = np.where(mask, youngs_modulus_from_density(density), 0.0)
    strength = np.where(mask, yield_strength_from_density(density), 0.0)
    return modulus, strength


def _load_mask(masks: TibFibMasks) -> np.ndarray:
    """Score the tibia. A thin fibula voxel chain should not set the limb failure load."""
    if int(masks.tibia.sum()) >= 30:
        return masks.tibia
    return masks.combined


def _shaft_calibration(
    hu: np.ndarray,
    masks: TibFibMasks,
    spacing_zyx: tuple[float, float, float],
) -> tuple[float, float, bool]:
    """Return radius scale, cortical HU, and whether the reference should be size-matched.

    A phantom that already has the reference cross-section keeps that exact model.
    A clinical shaft uses its own periosteal size and cortical HU so scanner
    calibration does not become a fake strength deficit.
    """
    tibia = masks.tibia if int(masks.tibia.sum()) >= 30 else masks.combined
    areas = []
    for index in range(tibia.shape[0]):
        count = int(tibia[index].sum())
        if count >= 4:
            areas.append(count * float(spacing_zyx[1]) * float(spacing_zyx[2]))
    if len(areas) < 3:
        return 1.0, CORTICAL_HU, False
    radius = float(np.sqrt(float(np.median(areas)) / np.pi))
    scale = radius / TIBIA_RADIUS_MM
    cortical = hu[tibia & (hu >= 700.0)]
    cortical_hu = float(np.percentile(cortical, 60)) if cortical.size >= 8 else CORTICAL_HU
    matched = abs(scale - 1.0) >= 0.08 or abs(cortical_hu - CORTICAL_HU) >= 120.0
    if not matched:
        return 1.0, CORTICAL_HU, False
    return float(np.clip(scale, 0.60, 1.40)), float(np.clip(cortical_hu, 700.0, 1700.0)), True


def _reference_ct(
    length_mm: float,
    spacing_zyx: tuple[float, float, float],
    force_n: float,
    radius_scale: float = 1.0,
    cortical_hu: float = CORTICAL_HU,
    size_matched: bool = False,
) -> VoxelSolveResult:
    spacing_zyx = tuple(float(value) for value in spacing_zyx)
    isotropic = max(spacing_zyx) - min(spacing_zyx) < 0.05
    if not size_matched and isotropic:
        reference_hu, spacing = build_reference_volume(length_mm, spacing_zyx[0])
        masks = segment_tibia_and_fibula(reference_hu, spacing)
        if int(masks.combined.sum()) < 30:
            raise StrengthAnalysisError("The normal tib/fib reference could not be segmented.", modality="ct")
        hu, masks = _crop_pair(reference_hu, masks)
        modulus, strength = _ct_materials(hu, masks.combined)
        return _solve_masked(modulus, strength, masks.combined, spacing, force_n)
    fine = 0.8
    scale = float(radius_scale)
    reference_hu, _fine_spacing = synthetic_tibfib_volume(
        length_mm,
        fine,
        cortical_hu=float(cortical_hu),
        trabecular_hu=max(180.0, float(cortical_hu) * 0.25),
        tibia_radius_mm=TIBIA_RADIUS_MM * scale,
        tibia_cortex_mm=max(1.2, TIBIA_CORTEX_MM * scale),
        fibula_radius_mm=FIBULA_RADIUS_MM * scale,
        fibula_cortex_mm=max(0.8, FIBULA_CORTEX_MM * scale),
        fibula_offset_mm=FIBULA_OFFSET_MM * scale,
    )
    zoom = tuple(fine / spacing for spacing in spacing_zyx)
    matched = ndi.zoom(reference_hu, zoom, order=0).astype(np.float32)
    masks = segment_tibia_and_fibula(matched, spacing_zyx)
    if int(masks.combined.sum()) < 30:
        raise StrengthAnalysisError("The normal tib/fib reference could not be segmented.", modality="ct")
    hu, masks = _crop_pair(matched, masks)
    modulus, strength = _ct_materials(hu, masks.combined)
    return _solve_masked(modulus, strength, _load_mask(masks), spacing_zyx, force_n)


def _finish(
    *,
    modality: str,
    method: str,
    solver: str,
    mass_kg: float,
    patient: VoxelSolveResult,
    reference: VoxelSolveResult,
    bone_length_mm: float,
    tibia_voxels: int,
    fibula_voxels: int,
    assumptions: list[str],
    views: dict[str, bool] | None = None,
) -> StrengthReport:
    compared = strength_comparison(patient.failure_load_n, reference.failure_load_n)
    walked = walking_assessment(patient.failure_load_n, mass_kg)
    weakness = weakness_volume(patient, reference)
    return StrengthReport(
        modality=modality,
        method=method,
        body_mass_kg=float(mass_kg),
        body_weight_n=float(walked["body_weight_n"]),
        failure_load_n=float(patient.failure_load_n),
        failure_load_bodyweights=float(walked["failure_load_bodyweights"]),
        reference_failure_load_n=float(compared["reference_failure_load_n"]),
        percent_vs_normal=float(compared["percent_vs_normal"]),
        signed_percent_weaker=float(compared["signed_percent_weaker"]),
        comparison=str(compared["comparison"]),
        walking_verdict=str(walked["walking_verdict"]),
        walking_text=str(walked["walking_text"]),
        field_note=field_of_view_note(bone_length_mm),
        assumptions=assumptions,
        hotspot_z_mm=hotspot_height_mm(weakness, patient.spacing_zyx[0]),
        bone_length_mm=float(bone_length_mm),
        tibia_voxels=int(tibia_voxels),
        fibula_voxels=int(fibula_voxels),
        solver=solver,
        weakness=weakness,
        weakness_spacing_zyx=patient.spacing_zyx,
        views_acquired=views or {"ap": True, "lateral": True},
    )


def analyze_ct_volume(
    hu: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    body_mass_kg: float,
) -> StrengthReport:
    volume, masks, spacing = _prepare_ct_grid(hu, spacing_zyx)
    weight = body_weight_from_mass(body_mass_kg)
    modulus, strength = _ct_materials(volume, masks.combined)
    radius_scale, cortical_hu, size_matched = _shaft_calibration(volume, masks, spacing)
    solved_mask = _load_mask(masks) if size_matched else masks.combined
    patient = _solve_masked(modulus, strength, solved_mask, spacing, weight)
    length = _bone_length_mm(solved_mask, spacing[0])
    reference = _reference_ct(
        length,
        spacing,
        weight,
        radius_scale=radius_scale,
        cortical_hu=cortical_hu,
        size_matched=size_matched,
    )
    assumptions = [
        "Uncalibrated clinical CT: absolute newtons are not phantom-calibrated. The percent comparison uses the same material law on both legs.",
        "Linear voxel finite-element model. Failure is the axial load at which 2% of interior cortical voxels exceed yield.",
        "Vertical load only. The distal end is fixed and the proximal end carries body weight.",
    ]
    if size_matched:
        assumptions.append(
            "The normal leg is scaled to this bone's length and midshaft size, and it uses this scan's cortical HU."
        )
    if masks.fibula_voxels == 0:
        assumptions.append("The fibula was not a separate connected component, so the model uses the bone that could be separated.")
    elif size_matched and int(masks.tibia.sum()) >= 30:
        assumptions.append("Failure load is the tibial cortex. The fibula is kept in the segmentation but does not set the percent.")
    return _finish(
        modality="ct",
        method="voxel_hexahedral_linear_elastic",
        solver=patient.solver,
        mass_kg=body_mass_kg,
        patient=patient,
        reference=reference,
        bone_length_mm=length,
        tibia_voxels=masks.tibia_voxels,
        fibula_voxels=masks.fibula_voxels,
        assumptions=assumptions,
    )


def body_weight_from_mass(mass_kg: float) -> float:
    try:
        return body_weight_newtons(mass_kg)
    except ValueError as exc:
        raise StrengthAnalysisError(str(exc)) from exc


def _uniform_cortex(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    modulus = np.where(mask, UNIFORM_CORTEX_MODULUS_MPA, 0.0)
    strength = np.where(mask, UNIFORM_CORTEX_YIELD_MPA, 0.0)
    return modulus, strength


def analyze_mri_volume(
    volume: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    body_mass_kg: float,
) -> StrengthReport:
    data = np.asarray(volume, dtype=np.float32)
    spacing = tuple(float(value) for value in spacing_zyx)
    if data.size > 180_000 or min(spacing) < 1.5:
        data, spacing = _resample(data, spacing, 2.5)
    try:
        cortex = segment_mri_cortex(data)
    except StrengthAnalysisError:
        raise
    except RuntimeError as exc:
        raise StrengthAnalysisError(str(exc), modality="mri") from exc
    slc = _bounds(cortex, margin=0)
    data = data[slc]
    cortex = cortex[slc]
    weight = body_weight_from_mass(body_mass_kg)
    modulus, strength = _uniform_cortex(cortex)
    patient = _solve_masked(modulus, strength, cortex, spacing, weight)
    length = _bone_length_mm(cortex, spacing[0])
    reference_hu, ref_spacing = build_reference_volume(length, spacing[0])
    reference_cortex = reference_hu >= 1000.0
    ref_slc = _bounds(reference_cortex, margin=0)
    reference_cortex = reference_cortex[ref_slc]
    ref_mod, ref_yield = _uniform_cortex(reference_cortex)
    reference = _solve_masked(ref_mod, ref_yield, reference_cortex, ref_spacing, weight)
    assumptions = [
        "MRI cannot measure bone density. This percent is geometric stiffness of the cortical shell versus a normal leg, using a uniform cortical modulus of 17 GPa.",
        "If the dark cortical ring cannot be segmented, the analysis stops instead of inventing a percent.",
    ]
    solved = _finish(
        modality="mri",
        method="mri_uniform_cortex_geometry",
        solver=patient.solver,
        mass_kg=body_mass_kg,
        patient=patient,
        reference=reference,
        bone_length_mm=length,
        tibia_voxels=int(cortex.sum()),
        fibula_voxels=0,
        assumptions=assumptions,
    )
    solved.modality = "mri"
    return solved


def _spacing_report(
    views: dict[str, tuple[np.ndarray, tuple[float, float]]],
    spacing_meta: dict[str, dict[str, Any]] | None,
) -> tuple[bool, str, str]:
    if not spacing_meta:
        bits = [
            f"{name} {float(spacing[0]):.3f} × {float(spacing[1]):.3f} mm"
            for name, (_image, spacing) in views.items()
        ]
        detail = ", ".join(bits)
        return False, "provided", f"Pixel spacing was supplied with the radiograph ({detail})."
    assumed = any(bool(item.get("assumed")) for item in spacing_meta.values())
    sources: list[str] = []
    bits = []
    for name, item in spacing_meta.items():
        source = str(item.get("source", "unknown"))
        if source not in sources:
            sources.append(source)
        bits.append(f"{name} {float(item['row_mm']):.3f} × {float(item['col_mm']):.3f} mm ({source})")
    source_label = ", ".join(sources) if sources else "unknown"
    detail = ", ".join(bits)
    if assumed:
        sentence = (
            "Pixel spacing was not in the DICOM header, so 1.0 mm was assumed "
            f"({detail}). Millimetre diameters are not calibrated."
        )
    else:
        sentence = f"Pixel spacing came from {source_label} ({detail})."
    return assumed, source_label, sentence


def analyze_radiograph_views(
    views: dict[str, tuple[np.ndarray, tuple[float, float]]],
    body_mass_kg: float,
    view_assumed: dict[str, bool] | None = None,
    spacing_meta: dict[str, dict[str, Any]] | None = None,
) -> StrengthReport:
    measured = analyze_radiographs(views)
    compared = strength_comparison(float(measured["failure_load_n"]), float(measured["reference_failure_load_n"]))
    walked = walking_assessment(float(measured["failure_load_n"]), body_mass_kg)
    rasters = {name: np.asarray(image) for name, image in dict(measured["rasters"]).items()}
    assumptions = [
        str(measured["assumption"]),
        "Radiograph strength uses a 120 MPa cortical yield and the weakest midshaft section. It is not a voxel finite-element model.",
        "The normal leg is a digitally reconstructed radiograph of the reference tib/fib segment.",
    ]
    spacing_assumed, spacing_source, spacing_sentence = _spacing_report(views, spacing_meta)
    assumptions.append(spacing_sentence)
    canal_resolved = bool(measured["canal_resolved"])
    if not canal_resolved:
        assumptions.append(
            "The medullary canal was not resolved, so cortical thickness was limited to 2 mm instead of treating the shaft as a solid rod."
        )
    assumed = view_assumed or {}
    if any(assumed.values()):
        assumptions.append("View position was not in the DICOM header, so the projection was treated as AP.")
    percent_stronger = float(compared["percent_vs_normal"]) if compared["comparison"] == "stronger" else 0.0
    blocked = blocks_stronger_claim(
        percent_stronger,
        canal_resolved=canal_resolved,
        outer_diameters_mm=[float(value) for value in measured["outer_diameters_mm"]],
        spacing_assumed=spacing_assumed,
        reference_resolved=bool(measured["reference_resolved"]),
    )
    comparison = str(compared["comparison"])
    measurement_reliable = True
    measurement_note = ""
    unreliable_reason = ""
    walking_verdict = str(walked["walking_verdict"])
    walking_text = str(walked["walking_text"])
    if spacing_assumed:
        comparison = "unreliable"
        measurement_reliable = False
        unreliable_reason = "spacing"
        measurement_note = MISSING_SPACING_NOTE
        walking_verdict = "unreliable"
        walking_text = "Walking tolerance is not estimated. Pixel spacing was not in the DICOM header."
        assumptions.append(measurement_note)
    elif blocked:
        comparison = "unreliable"
        measurement_reliable = False
        unreliable_reason = "stronger"
        measurement_note = UNRELIABLE_STRONGER_NOTE
        walking_verdict = "unreliable"
        walking_text = (
            "Walking tolerance is not estimated. This radiograph cannot claim the limb is stronger than a normal shaft."
        )
        assumptions.append(measurement_note)
    elif not canal_resolved:
        comparison = "unreliable"
        measurement_reliable = False
        unreliable_reason = "canal"
        measurement_note = UNRESOLVED_CANAL_NOTE
        walking_verdict = "unreliable"
        walking_text = (
            "Walking tolerance is not estimated. The medullary canal was not resolved, so this study is not a clearance to walk."
        )
        assumptions.append(measurement_note)
    view_rows: dict[str, Any] = {}
    for name, item in dict(measured["view_measures"]).items():
        view_rows[name] = {
            "outer_diameter_mm": float(item["outer_diameter_mm"]),
            "inner_diameter_mm": float(item["inner_diameter_mm"]),
            "inner_estimated": bool(item["inner_estimated"]),
            "canal_resolved": bool(item["canal_resolved"]),
            "spacing_mm": [float(value) for value in item["spacing_mm"]],
        }
    radiograph_measures = {
        "cortical_area_mm2": float(measured["cortical_area_mm2"]),
        "reference_cortical_area_mm2": float(measured["reference_cortical_area_mm2"]),
        "canal_resolved": canal_resolved,
        "reference_resolved": bool(measured["reference_resolved"]),
        "spacing_source": spacing_source,
        "spacing_assumed": spacing_assumed,
        "patient_failure_load_n": float(measured["failure_load_n"]),
        "reference_failure_load_n": float(measured["reference_failure_load_n"]),
        "views": view_rows,
    }
    return StrengthReport(
        modality="radiograph",
        method="radiograph_cortical_index",
        body_mass_kg=float(body_mass_kg),
        body_weight_n=float(walked["body_weight_n"]),
        failure_load_n=float(measured["failure_load_n"]),
        failure_load_bodyweights=float(walked["failure_load_bodyweights"]),
        reference_failure_load_n=float(compared["reference_failure_load_n"]),
        percent_vs_normal=float(compared["percent_vs_normal"]),
        signed_percent_weaker=float(compared["signed_percent_weaker"]),
        comparison=comparison,
        walking_verdict=walking_verdict,
        walking_text=walking_text,
        field_note=field_of_view_note(float(measured["bone_length_mm"])),
        assumptions=assumptions,
        hotspot_z_mm=None if measured["hotspot_z_mm"] is None else float(measured["hotspot_z_mm"]),
        bone_length_mm=float(measured["bone_length_mm"]),
        tibia_voxels=int(measured["tibia_voxels"]),
        fibula_voxels=int(measured["fibula_voxels"]),
        solver="radiograph_cortical_index",
        weakness=None,
        weakness_spacing_zyx=None,
        views_acquired={"ap": "ap" in rasters, "lateral": "lateral" in rasters},
        rasters=rasters,
        measurement_reliable=measurement_reliable,
        measurement_note=measurement_note,
        radiograph_measures=radiograph_measures,
        unreliable_reason=unreliable_reason,
    )


def analyze_loaded_study(study: dict[str, Any], body_mass_kg: float | None, prefer_dicom_weight: bool = True) -> StrengthReport:
    header_weight = study.get("patient_weight_kg")
    used_header = bool(prefer_dicom_weight and header_weight)
    if used_header:
        mass = float(header_weight)
    elif body_mass_kg is None:
        raise StrengthAnalysisError("Enter a body mass. This DICOM header has no patient weight.")
    else:
        mass = float(body_mass_kg)
    if mass < 20 or mass > 300:
        raise StrengthAnalysisError("Body mass must be between 20 and 300 kg.")
    kind = study["kind"]
    if kind == "ct":
        report = analyze_ct_volume(study["volume"], study["spacing_zyx"], mass)
    elif kind == "mri":
        report = analyze_mri_volume(study["volume"], study["spacing_zyx"], mass)
    elif kind == "radiograph":
        report = analyze_radiograph_views(
            study["views"],
            mass,
            study.get("view_assumed"),
            study.get("radiograph_spacing"),
        )
    else:
        raise StrengthAnalysisError(f"Unsupported scan type: {kind}")
    if used_header:
        report.body_mass_source = "dicom"
    else:
        report.body_mass_source = "entered"
        report.assumptions.append(
            "Body mass was not taken from the DICOM header. The value entered for this run is not a measured patient weight."
        )
    return report


def analyze_dicom_path(
    path: Path,
    body_mass_kg: float | None = None,
    prefer_dicom_weight: bool = True,
) -> StrengthReport:
    study = load_dicom_path(Path(path))
    return analyze_loaded_study(study, body_mass_kg, prefer_dicom_weight=prefer_dicom_weight)


def projections_for_report(report: StrengthReport) -> tuple[np.ndarray | None, np.ndarray | None]:
    if report.modality == "radiograph":
        return projection_pair_from_rasters(report.rasters)
    if report.weakness is None:
        return None, None
    return project_views(report.weakness)
