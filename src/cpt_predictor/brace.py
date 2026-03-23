from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .models import BraceModel, MeshResult


def _import_pyvista():
    import pyvista as pv

    return pv


class BraceContactBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def prepare(
        self,
        mesh_result: MeshResult,
        output_dir: Path,
        brace_stl: Optional[str] = None,
    ) -> BraceModel:
        pv = _import_pyvista()
        output_dir.mkdir(parents=True, exist_ok=True)

        brace_cfg = self.config["brace"]
        if brace_stl and Path(brace_stl).exists():
            surface = pv.read(brace_stl).extract_surface().triangulate()
            path = output_dir / "brace_model.stl"
            surface.save(path)
            bounds = tuple(float(value) for value in surface.bounds)
            return BraceModel(
                enabled=True,
                surface=surface,
                surface_path=path,
                source="user_stl",
                support_bounds_xyz=bounds,
                metadata={"stress_reduction_factor": float(brace_cfg.get("stress_reduction_factor", 0.72))},
            )

        if not brace_cfg.get("create_proxy_if_missing", True):
            return BraceModel(enabled=False, surface=None, surface_path=None, source="disabled")

        bounds = mesh_result.surface.bounds
        x_min, x_max, y_min, y_max, z_min, z_max = [float(v) for v in bounds]
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        cz = 0.5 * (z_min + z_max)
        height = (z_max - z_min) * float(brace_cfg.get("support_length_fraction", 0.75))
        radius = 0.5 * max(x_max - x_min, y_max - y_min) + float(brace_cfg.get("radial_clearance_mm", 8.0))

        surface = pv.Cylinder(
            center=(cx, cy, cz),
            direction=(0.0, 0.0, 1.0),
            radius=radius,
            height=height,
            resolution=64,
            capping=False,
        ).triangulate()

        path = output_dir / "brace_proxy.stl"
        surface.save(path)
        support_bounds = (cx - radius, cx + radius, cy - radius, cy + radius, cz - height / 2.0, cz + height / 2.0)
        return BraceModel(
            enabled=True,
            surface=surface,
            surface_path=path,
            source="proxy",
            support_bounds_xyz=support_bounds,
            metadata={"stress_reduction_factor": float(brace_cfg.get("stress_reduction_factor", 0.72))},
        )

