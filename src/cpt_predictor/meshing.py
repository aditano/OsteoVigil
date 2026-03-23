from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .models import MeshResult, SegmentationResult, StudyData


def _import_pyvista():
    import pyvista as pv

    return pv


def _import_skimage_measure():
    from skimage import measure

    return measure


def tetra_connectivity(mesh: Any) -> np.ndarray:
    if hasattr(mesh, "cells_dict") and 10 in mesh.cells_dict:
        return np.asarray(mesh.cells_dict[10], dtype=np.int64)

    cells = np.asarray(mesh.cells)
    if cells.size == 0:
        return np.empty((0, 4), dtype=np.int64)

    reshaped = cells.reshape((-1, 5))
    return reshaped[:, 1:5].astype(np.int64)


class MeshBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def build(self, study: StudyData, segmentation: SegmentationResult, output_dir: Path) -> MeshResult:
        pv = _import_pyvista()
        measure = _import_skimage_measure()

        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_cfg = self.config["meshing"]

        verts_zyx, faces, _normals, _values = measure.marching_cubes(
            segmentation.mask.astype(np.float32),
            level=0.5,
            spacing=study.spacing_zyx,
        )

        points_xyz = np.column_stack(
            [
                verts_zyx[:, 2] + study.origin_xyz[0],
                verts_zyx[:, 1] + study.origin_xyz[1],
                verts_zyx[:, 0] + study.origin_xyz[2],
            ]
        )
        face_array = np.hstack(
            [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
        ).ravel()

        surface = pv.PolyData(points_xyz, face_array)
        surface = surface.clean()

        decimation = float(mesh_cfg.get("surface_decimation", 0.0))
        if decimation > 0:
            surface = surface.decimate_pro(decimation)

        smoothing_iterations = int(mesh_cfg.get("surface_smoothing_iterations", 30))
        if smoothing_iterations > 0:
            surface = surface.smooth(n_iter=smoothing_iterations)

        grid = None
        if mesh_cfg.get("use_tetgen_if_available", True):
            try:
                import tetgen

                tet = tetgen.TetGen(surface.triangulate())
                tet.tetrahedralize(
                    order=1,
                    mindihedral=float(mesh_cfg.get("tetgen_mindihedral", 18.0)),
                    minratio=float(mesh_cfg.get("tetgen_minratio", 1.4)),
                )
                grid = tet.grid
            except Exception:
                grid = None

        if grid is None:
            grid = surface.delaunay_3d(alpha=float(mesh_cfg.get("delaunay_alpha", 2.0)))

        if hasattr(grid, "extract_cells") and hasattr(grid, "celltypes"):
            tet_type = 10
            tet_ids = np.where(np.asarray(grid.celltypes) == tet_type)[0]
            if len(tet_ids):
                grid = grid.extract_cells(tet_ids)

        surface_path = output_dir / "tibia_surface.stl"
        mesh_path = output_dir / "tibia_mesh.vtu"
        surface.save(surface_path)
        grid.save(mesh_path)

        stats = {
            "surface_points": int(surface.n_points),
            "surface_faces": int(surface.n_cells),
            "tetra_cells": int(grid.n_cells),
            "mesh_points": int(grid.n_points),
        }

        return MeshResult(mesh=grid, surface=surface, mesh_path=mesh_path, surface_path=surface_path, stats=stats)

