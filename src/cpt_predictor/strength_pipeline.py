"""Run tib/fib vertical-load analysis for CT, MRI, or radiographs."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

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
from .radiograph import analyze_radiographs, projection_pair_from_rasters, raster_volume
from .reference_leg import build_reference_volume
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
    if volume.size > 180_000 or min(spacing) < 1.5:
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


def _reference_ct(
    length_mm: float,
    spacing_mm: float,
    force_n: float,
) -> VoxelSolveResult:
    reference_hu, spacing = build_reference_volume(length_mm, spacing_mm)
    masks = segment_tibia_and_fibula(reference_hu, spacing)
    if int(masks.combined.sum()) < 30:
        raise StrengthAnalysisError("The normal tib/fib reference could not be segmented.", modality="ct")
    hu, masks = _crop_pair(reference_hu, masks)
    modulus, strength = _ct_materials(hu, masks.combined)
    return _solve_masked(modulus, strength, masks.combined, spacing, force_n)


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
    patient = _solve_masked(modulus, strength, masks.combined, spacing, weight)
    length = _bone_length_mm(masks.combined, spacing[0])
    reference = _reference_ct(length, spacing[0], weight)
    assumptions = [
        "Uncalibrated clinical CT: absolute newtons are not phantom-calibrated. The percent comparison uses the same material law on both legs.",
        "Linear voxel finite-element model. Failure is the axial load at which 2% of interior cortical voxels exceed yield.",
        "Vertical load only. The distal end is fixed and the proximal end carries body weight.",
    ]
    if masks.fibula_voxels == 0:
        assumptions.append("The fibula was not a separate connected component, so the model uses the bone that could be separated.")
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


def analyze_radiograph_views(
    views: dict[str, tuple[np.ndarray, tuple[float, float]]],
    body_mass_kg: float,
    view_assumed: dict[str, bool] | None = None,
) -> StrengthReport:
    measured = analyze_radiographs(views)
    compared = strength_comparison(float(measured["failure_load_n"]), float(measured["reference_failure_load_n"]))
    walked = walking_assessment(float(measured["failure_load_n"]), body_mass_kg)
    rasters = {name: np.asarray(image) for name, image in dict(measured["rasters"]).items()}
    volume = raster_volume(rasters)
    assumptions = [
        str(measured["assumption"]),
        "Radiograph strength uses a 120 MPa cortical yield and the weakest midshaft section. It is not a voxel finite-element model.",
        "The normal leg is a digitally reconstructed radiograph of the reference tib/fib segment.",
    ]
    assumed = view_assumed or {}
    if any(assumed.values()):
        assumptions.append("View position was not in the DICOM header, so the projection was treated as AP.")
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
        comparison=str(compared["comparison"]),
        walking_verdict=str(walked["walking_verdict"]),
        walking_text=str(walked["walking_text"]),
        field_note=field_of_view_note(float(measured["bone_length_mm"])),
        assumptions=assumptions,
        hotspot_z_mm=None if measured["hotspot_z_mm"] is None else float(measured["hotspot_z_mm"]),
        bone_length_mm=float(measured["bone_length_mm"]),
        tibia_voxels=int(measured["tibia_voxels"]),
        fibula_voxels=int(measured["fibula_voxels"]),
        solver="radiograph_cortical_index",
        weakness=None if volume is None else np.asarray(volume, dtype=np.float32),
        weakness_spacing_zyx=(1.0, 1.0, 1.0),
        views_acquired={"ap": "ap" in rasters, "lateral": "lateral" in rasters},
        rasters=rasters,
    )


def analyze_loaded_study(study: dict[str, Any], body_mass_kg: float | None, prefer_dicom_weight: bool = True) -> StrengthReport:
    header_weight = study.get("patient_weight_kg")
    if prefer_dicom_weight and header_weight:
        mass = float(header_weight)
    elif body_mass_kg is None:
        raise StrengthAnalysisError("Enter a body mass. This DICOM header has no patient weight.")
    else:
        mass = float(body_mass_kg)
    if mass < 20 or mass > 300:
        raise StrengthAnalysisError("Body mass must be between 20 and 300 kg.")
    kind = study["kind"]
    if kind == "ct":
        return analyze_ct_volume(study["volume"], study["spacing_zyx"], mass)
    if kind == "mri":
        return analyze_mri_volume(study["volume"], study["spacing_zyx"], mass)
    if kind == "radiograph":
        return analyze_radiograph_views(study["views"], mass, study.get("view_assumed"))
    raise StrengthAnalysisError(f"Unsupported scan type: {kind}")


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
