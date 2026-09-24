from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .models import MaterialResult, MeshResult, StudyData


MATERIAL_BINS = [-500.0, 0.0, 250.0, 750.0, 1250.0, 2000.0]

# Morgan et al. 2003 proximal-tibia trabecular curve, E(MPa) = 6850 * rho_app^1.49.
# The cortical branch is fit so uncalibrated clinical HU of about 1200 and 1800
# land near 12 GPa and 18 GPa. Apparent density uses the existing ash line
# rho_ash = 0.000887*HU + 0.0633. The curves blend from 0.85 to 1.15 g/cm^3.
_CORTICAL_RHO_REF = 1.173
_CORTICAL_E_REF_MPA = 12000.0
_CORTICAL_EXPONENT = 1.049


def _smoothstep(values: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    span = edge1 - edge0
    blend = np.clip((values - edge0) / span, 0.0, 1.0)
    return blend * blend * (3.0 - 2.0 * blend)


def apparent_density_g_cm3(hu_values: Any, ash_to_apparent_scale: float = 1.04) -> np.ndarray:
    hu_array = np.asarray(hu_values, dtype=float)
    rho_ash = np.maximum(0.05, (0.000887 * hu_array) + 0.0633)
    return np.clip(rho_ash * float(ash_to_apparent_scale), 0.10, 2.40)


def youngs_modulus_from_density(rho_app: Any) -> np.ndarray:
    rho = np.asarray(rho_app, dtype=float)
    safe = np.maximum(rho, 1.0e-6)
    trabecular = 6850.0 * np.power(safe, 1.49)
    cortical = _CORTICAL_E_REF_MPA * np.power(safe / _CORTICAL_RHO_REF, _CORTICAL_EXPONENT)
    weight = _smoothstep(rho, 0.85, 1.15)
    modulus = ((1.0 - weight) * trabecular) + (weight * cortical)
    return np.clip(modulus, 100.0, 22000.0)


def yield_strength_from_density(rho_app: Any) -> np.ndarray:
    rho = np.maximum(np.asarray(rho_app, dtype=float), 1.0e-6)
    return np.clip(114.8 * np.power(rho, 1.72), 2.0, 220.0)


def hu_to_bone_properties(
    hu_values: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    hu_array = np.asarray(hu_values, dtype=float)
    cfg = config or {"materials": {"density_scale_from_ash_to_apparent": 1.04}}
    scale = float(cfg["materials"].get("density_scale_from_ash_to_apparent", 1.04))

    rho_app = apparent_density_g_cm3(hu_array, scale)
    youngs_modulus = youngs_modulus_from_density(rho_app)
    yield_strength = yield_strength_from_density(rho_app)
    if np.isscalar(hu_values) or hu_array.shape == ():
        return {
            "density_g_cm3": float(rho_app),
            "youngs_modulus_mpa": float(youngs_modulus),
            "yield_strength_mpa": float(yield_strength),
        }
    return rho_app, youngs_modulus, yield_strength


class BoneMaterialMapper:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _sample_hu_at_points(self, study: StudyData, points_xyz: np.ndarray) -> np.ndarray:
        z_idx = np.rint((points_xyz[:, 2] - study.origin_xyz[2]) / study.spacing_zyx[0]).astype(int)
        y_idx = np.rint((points_xyz[:, 1] - study.origin_xyz[1]) / study.spacing_zyx[1]).astype(int)
        x_idx = np.rint((points_xyz[:, 0] - study.origin_xyz[0]) / study.spacing_zyx[2]).astype(int)

        z_idx = np.clip(z_idx, 0, study.hu_volume.shape[0] - 1)
        y_idx = np.clip(y_idx, 0, study.hu_volume.shape[1] - 1)
        x_idx = np.clip(x_idx, 0, study.hu_volume.shape[2] - 1)
        return study.hu_volume[z_idx, y_idx, x_idx]

    def map_to_mesh(self, study: StudyData, mesh_result: MeshResult, output_dir: Path) -> MaterialResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh = mesh_result.mesh.copy(deep=True)
        centers = np.asarray(mesh.cell_centers().points)
        hu_values = self._sample_hu_at_points(study, centers)
        density, modulus, strength = hu_to_bone_properties(hu_values, self.config)

        bin_count = max(2, int(self.config["materials"].get("material_bins", 6)))
        quantiles = np.quantile(modulus, np.linspace(0.0, 1.0, bin_count + 1))
        quantiles = np.unique(quantiles)
        if len(quantiles) <= 2:
            material_bin = np.ones_like(modulus, dtype=int)
            unique_bins = [1]
        else:
            material_bin = np.digitize(modulus, quantiles[1:-1], right=True) + 1
            unique_bins = sorted({int(value) for value in material_bin.tolist()})

        mesh.cell_data["HU"] = hu_values
        mesh.cell_data["density_g_cm3"] = density
        mesh.cell_data["youngs_modulus_mpa"] = modulus
        mesh.cell_data["yield_strength_mpa"] = strength
        mesh.cell_data["material_bin"] = material_bin

        table: List[Dict[str, Any]] = []
        for bin_id in unique_bins:
            mask = material_bin == bin_id
            table.append(
                {
                    "bin": int(bin_id),
                    "elements": int(mask.sum()),
                    "mean_density_g_cm3": float(np.mean(density[mask])),
                    "mean_youngs_modulus_mpa": float(np.mean(modulus[mask])),
                    "mean_yield_strength_mpa": float(np.mean(strength[mask])),
                }
            )

        mesh_path = output_dir / "material_mesh.vtu"
        mesh.save(mesh_path)

        stats = {
            "cell_count": int(mesh.n_cells),
            "hu_min": float(np.min(hu_values)),
            "hu_max": float(np.max(hu_values)),
            "modulus_min_mpa": float(np.min(modulus)),
            "modulus_max_mpa": float(np.max(modulus)),
        }
        return MaterialResult(mesh=mesh, mesh_path=mesh_path, materials_table=table, stats=stats)
