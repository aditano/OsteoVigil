from __future__ import annotations

import numpy as np

from cpt_predictor.fem_solver import solve_linear_tet_fea


def _unit_cube_tets() -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    tets = np.asarray(
        [
            [0, 1, 2, 5],
            [0, 2, 3, 7],
            [0, 2, 5, 6],
            [0, 4, 5, 7],
            [0, 5, 6, 7],
            [0, 2, 6, 7],
        ],
        dtype=np.int64,
    )
    return points, tets


def test_linear_tet_fea_produces_finite_stress_under_axial_load() -> None:
    points, tets = _unit_cube_tets()
    youngs = np.full(tets.shape[0], 1000.0)
    result = solve_linear_tet_fea(
        points,
        tets,
        youngs,
        poisson_ratio=0.30,
        distal_node_ids=[1, 2, 3, 4],
        proximal_node_ids=[5, 6, 7, 8],
        peak_force_n=10.0,
        lateral_force_n=0.0,
    )
    assert result.von_mises.size == tets.shape[0]
    assert np.all(np.isfinite(result.von_mises))
    assert float(np.max(result.von_mises)) > 0.0
    assert float(np.max(np.abs(result.displacement[:, 2]))) > 0.0
    assert result.residual_norm < 1.0e-6
    # Compression of a 1 mm^2 column by 10 N should be on the order of 10 MPa.
    assert 1.0 < float(np.mean(result.von_mises)) < 40.0


def test_linear_tet_fea_stiffer_material_displaces_less() -> None:
    points, tets = _unit_cube_tets()
    soft = solve_linear_tet_fea(
        points, tets, np.full(tets.shape[0], 500.0), 0.3, [1, 2, 3, 4], [5, 6, 7, 8], 8.0, 0.0
    )
    stiff = solve_linear_tet_fea(
        points, tets, np.full(tets.shape[0], 4000.0), 0.3, [1, 2, 3, 4], [5, 6, 7, 8], 8.0, 0.0
    )
    assert float(stiff.stats["max_displacement_mm"]) < float(soft.stats["max_displacement_mm"])
