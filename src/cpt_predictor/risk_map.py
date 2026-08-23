"""Highlight abnormal weak bone, not relative stress on a healthy shaft."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

# Failure utilization: 1.0 means von Mises equals yield.
VISIBLE_UTILIZATION_HEALTHY = 0.80
VISIBLE_UTILIZATION_DEFECT = 0.15
YIELD_UTILIZATION = 1.0
END_MARGIN = 0.12
CORTICAL_HU = 700.0
CORTICAL_MODULUS_MPA = 10000.0
HEALTHY_YIELD_VOLUME_FRACTION = 0.05
CANCELLOUS_HU_MAX = 900.0
DEFECT_ABS_HU = 280.0
DEFECT_RATIO = 0.55
DEFECT_DELTA_HU = 80.0
DEFECT_HALF_WIDTH_MM = 15.0


@dataclass
class DefectAssessment:
    has_defect: bool
    defect_z_mm: Optional[float]
    score: float
    reason: str
    weak_mean_hu: float
    median_cancellous_hu: float
    z_min: float
    z_max: float

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["has_structural_defect"] = bool(self.has_defect)
        return payload


def failure_utilization(von_mises: np.ndarray, yield_strength: np.ndarray) -> np.ndarray:
    stress = np.asarray(von_mises, dtype=float)
    strength = np.asarray(yield_strength, dtype=float)
    return stress / np.maximum(strength, 1e-6)


def failure_overlay_rgba(
    utilization: np.ndarray,
    *,
    min_visible: float = VISIBLE_UTILIZATION_DEFECT,
    vmax: float = YIELD_UTILIZATION,
) -> np.ndarray:
    """RGBA overlay: transparent below min_visible, opaque at yield."""
    from matplotlib import colormaps

    values = np.asarray(utilization, dtype=float)
    rgba = np.zeros(values.shape + (4,), dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return rgba
    scaled = np.clip(values / max(float(vmax), 1e-6), 0.0, 1.0)
    try:
        cmap = colormaps["inferno"]
    except Exception:
        from matplotlib import cm

        cmap = cm.get_cmap("inferno")
    colors = np.asarray(cmap(scaled), dtype=float)
    span = max(float(vmax) - float(min_visible), 1e-6)
    alpha = np.clip((values - float(min_visible)) / span, 0.0, 1.0)
    alpha = np.where(finite & (values >= float(min_visible)), np.clip(0.20 + 0.80 * alpha, 0.0, 1.0), 0.0)
    alpha = np.where(finite & (values >= float(vmax)), 1.0, alpha)
    colors[..., 3] = alpha
    colors[~finite] = 0.0
    return colors


def _cell_center_points(mesh: Any) -> Optional[np.ndarray]:
    n_cells = int(getattr(mesh, "n_cells", 0))
    if hasattr(mesh, "cell_centers"):
        try:
            centers = mesh.cell_centers()
            points = np.asarray(getattr(centers, "points", centers), dtype=float)
            if points.ndim == 2 and points.shape[0] == n_cells:
                return points
        except Exception:
            pass
    points = np.asarray(getattr(mesh, "points", []), dtype=float)
    if n_cells and points.ndim == 2 and points.shape[0] >= 1:
        centroid = np.mean(points, axis=0, keepdims=True)
        return np.repeat(centroid, n_cells, axis=0)
    return None


def _interior_mask(z: np.ndarray, margin: float = END_MARGIN) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        return np.zeros(0, dtype=bool)
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    span = z_max - z_min
    if span < 1e-3:
        return np.ones(z.shape, dtype=bool)
    zrel = (z - z_min) / span
    return (zrel >= float(margin)) & (zrel <= 1.0 - float(margin))


def assess_defect_from_samples(z: np.ndarray, hu: np.ndarray, n_bins: int = 36) -> DefectAssessment:
    """Flag a localized low-HU band (pseudarthrosis-like), not a uniformly osteopenic metaphysis."""
    z = np.asarray(z, dtype=float)
    hu = np.asarray(hu, dtype=float)
    empty = DefectAssessment(
        has_defect=False,
        defect_z_mm=None,
        score=0.0,
        reason="insufficient_samples",
        weak_mean_hu=float("nan"),
        median_cancellous_hu=float("nan"),
        z_min=float(np.min(z)) if z.size else 0.0,
        z_max=float(np.max(z)) if z.size else 0.0,
    )
    if z.size == 0 or hu.size != z.size:
        return empty

    interior = _interior_mask(z, margin=0.18)
    usable = interior & np.isfinite(hu)
    if not usable.any():
        empty.reason = "no_interior_hu"
        return empty

    z_sel = z[usable]
    hu_sel = hu[usable]
    z_min = float(np.min(z_sel))
    z_max = float(np.max(z_sel))
    n_bins = int(max(8, min(int(n_bins), max(8, z_sel.size // 6))))
    edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_ids = np.clip(np.digitize(z_sel, edges) - 1, 0, n_bins - 1)
    means = np.full(n_bins, np.nan, dtype=float)
    low_frac = np.full(n_bins, np.nan, dtype=float)
    for idx in range(n_bins):
        in_bin = bin_ids == idx
        count = int(in_bin.sum())
        if count < 2:
            continue
        values = hu_sel[in_bin]
        means[idx] = float(np.mean(values))
        low_frac[idx] = float(np.mean(values < 200.0))

    valid = np.isfinite(means)
    if int(valid.sum()) < 3:
        empty.reason = "too_few_length_bins"
        empty.z_min, empty.z_max = z_min, z_max
        return empty

    median_hu = float(np.nanmedian(means))
    weak_idx = int(np.nanargmin(means))
    weak_hu = float(means[weak_idx])
    other_means = np.delete(means[valid], np.argmin(means[valid]))
    neighbor_mean = float(np.mean(other_means)) if other_means.size else median_hu
    localized = (
        weak_hu < DEFECT_RATIO * median_hu
        and weak_hu < median_hu - DEFECT_DELTA_HU
        and weak_hu < DEFECT_ABS_HU
        and (neighbor_mean - weak_hu) > 60.0
    )
    fibrous = (
        np.isfinite(low_frac[weak_idx])
        and low_frac[weak_idx] >= 0.30
        and low_frac[weak_idx] > 1.8 * np.nanmedian(low_frac)
        and weak_hu < DEFECT_ABS_HU
        and weak_hu < median_hu - 50.0
    )
    has_defect = bool(localized or fibrous)
    defect_z = float(0.5 * (edges[weak_idx] + edges[weak_idx + 1])) if has_defect else None
    score = float(max(0.0, (median_hu - weak_hu) / max(median_hu, 1.0))) if has_defect else 0.0
    reason = "localized_low_hu_band" if has_defect else "no_localized_pseudarthrosis_band"
    return DefectAssessment(
        has_defect=has_defect,
        defect_z_mm=defect_z,
        score=score,
        reason=reason,
        weak_mean_hu=weak_hu,
        median_cancellous_hu=median_hu,
        z_min=z_min,
        z_max=z_max,
    )


def assess_mesh_defect(mesh: Any) -> DefectAssessment:
    centers = _cell_center_points(mesh)
    n_cells = int(getattr(mesh, "n_cells", 0))
    cell_data = getattr(mesh, "cell_data", {}) or {}
    hu = np.asarray(cell_data.get("HU", []), dtype=float)
    if centers is None:
        z = np.zeros(n_cells, dtype=float)
    else:
        z = centers[:, 2]
    if hu.size != n_cells:
        modulus = np.asarray(cell_data.get("youngs_modulus_mpa", []), dtype=float)
        if modulus.size == n_cells:
            hu = np.where(modulus >= CORTICAL_MODULUS_MPA, 1100.0, 350.0)
        else:
            return DefectAssessment(
                has_defect=False,
                defect_z_mm=None,
                score=0.0,
                reason="missing_hu",
                weak_mean_hu=float("nan"),
                median_cancellous_hu=float("nan"),
                z_min=float(np.min(z)) if z.size else 0.0,
                z_max=float(np.max(z)) if z.size else 0.0,
            )
    return assess_defect_from_samples(z, hu)


def clinical_analysis_mask(mesh: Any, assessment: Optional[DefectAssessment] = None) -> np.ndarray:
    """Cells that count toward clinical safety factor / years-to-failure."""
    n_cells = int(getattr(mesh, "n_cells", 0))
    cell_data = getattr(mesh, "cell_data", {}) or {}
    centers = _cell_center_points(mesh)
    if n_cells == 0:
        return np.zeros(0, dtype=bool)
    if centers is None:
        interior = np.ones(n_cells, dtype=bool)
        z = np.zeros(n_cells, dtype=float)
    else:
        z = centers[:, 2]
        interior = _interior_mask(z)
    hu = np.asarray(cell_data.get("HU", np.full(n_cells, np.nan)), dtype=float)
    modulus = np.asarray(cell_data.get("youngs_modulus_mpa", np.full(n_cells, np.nan)), dtype=float)
    cortex = np.zeros(n_cells, dtype=bool)
    if hu.size == n_cells:
        cortex |= np.isfinite(hu) & (hu >= CORTICAL_HU)
    if modulus.size == n_cells:
        cortex |= np.isfinite(modulus) & (modulus >= CORTICAL_MODULUS_MPA)
    mask = interior & cortex
    assessment = assessment or assess_mesh_defect(mesh)
    if assessment.has_defect and assessment.defect_z_mm is not None:
        near = np.abs(z - float(assessment.defect_z_mm)) <= DEFECT_HALF_WIDTH_MM
        mask = interior & (near | cortex)
    if not mask.any():
        safety = np.asarray(cell_data.get("safety_factor", np.full(n_cells, np.nan)), dtype=float)
        mask = interior & np.isfinite(safety) if safety.size == n_cells else interior
        if not mask.any() and safety.size == n_cells:
            mask = np.isfinite(safety)
        if not mask.any():
            mask = np.ones(n_cells, dtype=bool)
    return mask


def clinical_overlay_mask(
    mesh: Any,
    utilization: np.ndarray,
    assessment: DefectAssessment,
) -> np.ndarray:
    """Where heat may be drawn: defect tissue, or cortex that actually approaches yield."""
    n_cells = int(getattr(mesh, "n_cells", utilization.size))
    analysis = clinical_analysis_mask(mesh, assessment)
    util = np.asarray(utilization, dtype=float)
    if util.size != n_cells:
        util = np.resize(util, n_cells) if util.size else np.zeros(n_cells)
    if assessment.has_defect:
        return analysis & np.isfinite(util) & (util >= VISIBLE_UTILIZATION_DEFECT)
    near_yield = analysis & np.isfinite(util) & (util >= VISIBLE_UTILIZATION_HEALTHY)
    n_analysis = int(analysis.sum())
    if n_analysis == 0 or (int(near_yield.sum()) / n_analysis) < HEALTHY_YIELD_VOLUME_FRACTION:
        return np.zeros(n_cells, dtype=bool)
    return near_yield


def attach_clinical_risk_fields(mesh: Any) -> DefectAssessment:
    """Write clinical utilization / safety fields used by maps and risk summary."""
    cell_data = getattr(mesh, "cell_data", None)
    if cell_data is None:
        return DefectAssessment(
            has_defect=False,
            defect_z_mm=None,
            score=0.0,
            reason="no_cell_data",
            weak_mean_hu=float("nan"),
            median_cancellous_hu=float("nan"),
            z_min=0.0,
            z_max=0.0,
        )
    n_cells = int(getattr(mesh, "n_cells", 0))
    von_mises = np.asarray(cell_data.get("von_mises_mpa", np.zeros(n_cells)), dtype=float)
    strength = np.asarray(cell_data.get("yield_strength_mpa", np.full(n_cells, np.nan)), dtype=float)
    if von_mises.size != n_cells:
        von_mises = np.resize(von_mises, n_cells) if von_mises.size else np.zeros(n_cells)
    if strength.size != n_cells:
        strength = np.full(n_cells, np.nan, dtype=float)
    util = failure_utilization(von_mises, strength)
    assessment = assess_mesh_defect(mesh)
    analysis = clinical_analysis_mask(mesh, assessment)
    overlay = clinical_overlay_mask(mesh, util, assessment)
    safety = np.asarray(cell_data.get("safety_factor", np.full(n_cells, np.nan)), dtype=float)
    if safety.size != n_cells:
        safety = np.full(n_cells, np.nan, dtype=float)

    cell_data["failure_utilization"] = util
    cell_data["clinical_utilization"] = np.where(overlay, util, np.nan)
    cell_data["clinical_safety_factor"] = np.where(analysis, safety, np.nan)

    field_data = getattr(mesh, "field_data", None)
    if field_data is not None:
        field_data["has_structural_defect"] = np.asarray([int(assessment.has_defect)])
        if assessment.defect_z_mm is not None:
            field_data["defect_z_mm"] = np.asarray([float(assessment.defect_z_mm)])
    return assessment


def clinical_summary(mesh: Any, assessment: Optional[DefectAssessment] = None) -> Dict[str, Any]:
    assessment = assessment or assess_mesh_defect(mesh)
    cell_data = getattr(mesh, "cell_data", {}) or {}
    clinical_sf = np.asarray(cell_data.get("clinical_safety_factor", []), dtype=float)
    clinical_util = np.asarray(cell_data.get("failure_utilization", []), dtype=float)
    analysis = clinical_analysis_mask(mesh, assessment)
    yielded_fraction = 0.0
    if clinical_sf.size and analysis.size == clinical_sf.size and analysis.any():
        selected = clinical_sf[analysis]
        min_sf = float(np.nanmin(selected)) if assessment.has_defect else float(np.nanpercentile(selected, 1.0))
    elif clinical_sf.size and np.isfinite(clinical_sf).any():
        min_sf = float(np.nanmin(clinical_sf))
    else:
        min_sf = float("inf")
    if clinical_util.size and analysis.size == clinical_util.size and analysis.any():
        selected_util = np.nan_to_num(clinical_util[analysis], nan=0.0)
        max_util = float(np.max(selected_util))
        yielded_fraction = float(np.mean(selected_util >= 1.0))
    elif clinical_util.size and np.isfinite(clinical_util).any():
        max_util = float(np.nanmax(clinical_util))
    else:
        max_util = 0.0
    cycles = np.asarray(cell_data.get("fatigue_cycles", []), dtype=float)
    if cycles.size and analysis.size == cycles.size and analysis.any():
        selected_cycles = cycles[analysis]
        min_cycles = float(np.min(selected_cycles)) if assessment.has_defect else float(np.percentile(selected_cycles, 1.0))
    else:
        min_cycles = float("inf")
    if assessment.has_defect:
        caption = (
            "Color is closeness to failure at the detected weak zone (1 = yield). "
            "Gray bone is not approaching fracture under the modeled load."
        )
    else:
        caption = (
            "No pseudarthrosis-like weak zone was detected. Anatomy stays gray unless cortical bone approaches yield."
        )
    return {
        "has_structural_defect": bool(assessment.has_defect),
        "defect_reason": assessment.reason,
        "defect_z_mm": assessment.defect_z_mm,
        "defect_score": float(assessment.score),
        "min_clinical_safety_factor": min_sf,
        "max_clinical_utilization": max_util,
        "min_clinical_fatigue_cycles": min_cycles,
        "overlay_caption": caption,
        "n_clinical_cells": int(analysis.sum()) if analysis.size else 0,
        "yielded_volume_fraction": yielded_fraction,
    }
