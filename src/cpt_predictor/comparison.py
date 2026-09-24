"""Compare a solved limb with the normal leg and score daily vertical loads.

Walking thresholds follow in vivo knee contact forces in Bergmann et al. 2014
(OrthoLoad): level walking peaks near 2.8 body weights, and stairs or other
high daily loads reach about 4 body weights.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .voxel_fea import VoxelSolveResult, body_weight_newtons


LEVEL_WALKING_BODYWEIGHTS = 2.8
DAILY_LIFE_BODYWEIGHTS = 4.0
# Clamp artifacts sit just inside the solver's 8% end exclusion. The map ignores
# a slightly wider band so a matched leg does not highlight that numerical spike.
MAP_END_EXCLUSION = 0.12
RESEARCH_BANNER = (
    "Research biomechanical estimate. This is not a diagnosis, not a medical device, "
    "and not a clearance to walk."
)
WalkingVerdict = Literal["not_expected", "limited", "tolerated"]
ComparisonWord = Literal["weaker", "stronger", "matched"]


def walking_assessment(failure_load_n: float, mass_kg: float) -> dict[str, float | str]:
    weight = body_weight_newtons(mass_kg)
    multiple = float(failure_load_n) / weight
    if multiple < LEVEL_WALKING_BODYWEIGHTS:
        verdict: WalkingVerdict = "not_expected"
        text = "Not expected to tolerate level walking unassisted."
    elif multiple < DAILY_LIFE_BODYWEIGHTS:
        verdict = "limited"
        text = "Level walking may be tolerated. Stairs and unassisted daily life exceed the estimated margin."
    else:
        verdict = "tolerated"
        text = "Estimated to tolerate unassisted walking and ordinary daily loads."
    return {
        "failure_load_bodyweights": multiple,
        "body_weight_n": weight,
        "walking_verdict": verdict,
        "walking_text": text,
    }


def strength_comparison(patient_failure_n: float, reference_failure_n: float) -> dict[str, float | str]:
    if reference_failure_n <= 0:
        raise ValueError("The normal-leg failure load must be positive.")
    signed_percent = 100.0 * (1.0 - (float(patient_failure_n) / float(reference_failure_n)))
    if abs(signed_percent) < 1.0:
        word: ComparisonWord = "matched"
        displayed = 0.0
    elif signed_percent > 0:
        word = "weaker"
        displayed = signed_percent
    else:
        word = "stronger"
        displayed = -signed_percent
    return {
        "percent_vs_normal": float(displayed),
        "signed_percent_weaker": float(signed_percent),
        "comparison": word,
        "reference_failure_load_n": float(reference_failure_n),
    }


def _utilization_profile(result: VoxelSolveResult, n_bins: int = 48) -> tuple[np.ndarray, np.ndarray]:
    use = result.cortical if int(result.cortical.sum()) else np.ones(result.utilization.shape, dtype=bool)
    z_rel = (result.element_k[use].astype(float) + 0.5) / float(result.grid_shape[0])
    values = result.utilization[use]
    bins = int(max(8, min(n_bins, max(8, values.size // 4))))
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(bins, np.nan, dtype=float)
    bin_ids = np.clip(np.digitize(z_rel, edges) - 1, 0, bins - 1)
    for index in range(bins):
        chosen = values[bin_ids == index]
        if chosen.size:
            means[index] = float(np.mean(chosen))
    valid = np.isfinite(means)
    if int(valid.sum()) == 0:
        fill = float(np.mean(values)) if values.size else 0.0
        return centers, np.full(bins, fill, dtype=float)
    means = np.interp(centers, centers[valid], means[valid])
    return centers, means


def weakness_volume(patient: VoxelSolveResult, reference: VoxelSolveResult) -> np.ndarray:
    """Positive values are more highly utilized than the normal leg at the same relative height.

    Cortical voxels are always mapped. Low-modulus bone is mapped only when its
    utilization is well above the normal cortex at that height, which is how a
    pseudarthrosis-style defect shows up without painting the whole trabecular core.
    """
    centers, means = _utilization_profile(reference)
    z_rel = (patient.element_k.astype(float) + 0.5) / float(patient.grid_shape[0])
    reference_util = np.interp(z_rel, centers, means)
    delta = patient.utilization - reference_util
    elevated = patient.utilization > np.maximum(2.0 * reference_util, reference_util + 0.01)
    interior = (z_rel >= MAP_END_EXCLUSION) & (z_rel <= 1.0 - MAP_END_EXCLUSION)
    include = interior & (patient.cortical | elevated)
    volume = np.zeros(patient.grid_shape, dtype=np.float32)
    if not np.any(include):
        return volume
    volume[patient.element_k[include], patient.element_j[include], patient.element_i[include]] = delta[include]
    return volume


def project_views(weakness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clinical AP collapses Y. Clinical lateral collapses X. Both keep shaft height Z."""
    volume = np.asarray(weakness, dtype=float)
    ap = np.max(volume, axis=1)
    lateral = np.max(volume, axis=2)
    return ap.astype(np.float32), lateral.astype(np.float32)


def hotspot_height_mm(weakness: np.ndarray, spacing_z_mm: float, minimum_weakness: float = 0.004) -> float | None:
    """Height of a localized weakness peak, in millimetres from the distal end.

    The solve is loaded at one body weight, so utilization is a small number.
    A notch or defect lifts one height well above the shaft median. Matched
    legs only differ by solver noise near 0.002, which stays below the floor.
    """
    volume = np.asarray(weakness, dtype=float)
    if volume.ndim != 3 or volume.size == 0:
        return None
    slice_score = np.max(volume, axis=(1, 2))
    peak = float(np.max(slice_score))
    positive = slice_score[slice_score > 0]
    if positive.size == 0 or peak <= 0:
        return None
    baseline = float(np.median(positive))
    if peak < float(minimum_weakness) or peak < 2.5 * baseline:
        return None
    index = int(np.argmax(slice_score))
    return (index + 0.5) * float(spacing_z_mm)


def field_of_view_note(bone_length_mm: float) -> str:
    if bone_length_mm < 250.0:
        return (
            "The scanned field covers a distal or partial tib/fib segment, not a full tibial shaft. "
            "The comparison uses a normal shaft segment of the same length."
        )
    return "The scanned field is long enough to compare as a tib/fib shaft segment against a normal shaft of the same length."
