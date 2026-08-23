from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg, spsolve

from .meshing import tetra_connectivity


@dataclass
class LinearTetFeaResult:
    displacement: np.ndarray
    von_mises: np.ndarray
    principal_strain: np.ndarray
    residual_norm: float
    iterations: int
    solver: str
    stats: Dict[str, float] = field(default_factory=dict)


def _isotropic_d_matrix(nu: float) -> np.ndarray:
    lam_over_e = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_over_e = 1.0 / (2.0 * (1.0 + nu))
    d_matrix = np.zeros((6, 6), dtype=float)
    d_matrix[0, 0] = d_matrix[1, 1] = d_matrix[2, 2] = lam_over_e + 2.0 * mu_over_e
    d_matrix[0, 1] = d_matrix[0, 2] = d_matrix[1, 0] = d_matrix[1, 2] = d_matrix[2, 0] = d_matrix[2, 1] = lam_over_e
    d_matrix[3, 3] = d_matrix[4, 4] = d_matrix[5, 5] = mu_over_e
    return d_matrix


def _tet_gradients(points: np.ndarray, tets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    jacobian = np.empty((tets.shape[0], 3, 3), dtype=float)
    jacobian[:, :, 0] = p1 - p0
    jacobian[:, :, 1] = p2 - p0
    jacobian[:, :, 2] = p3 - p0
    volume = np.linalg.det(jacobian) / 6.0
    degenerate = np.abs(volume) < 1.0e-14
    if np.any(degenerate):
        jacobian[degenerate] = np.eye(3)[None, :, :] * 1.0e-3
        volume = np.where(degenerate, 1.0e-12, volume)
    inv_j = np.linalg.inv(jacobian)
    d_n_dxi = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    grad = np.einsum("ib,eba->eia", d_n_dxi, inv_j)
    return grad, np.abs(volume)


def _strain_displacement(grad: np.ndarray) -> np.ndarray:
    n_elem = grad.shape[0]
    b_matrix = np.zeros((n_elem, 6, 12), dtype=float)
    for node_index in range(4):
        gx = grad[:, node_index, 0]
        gy = grad[:, node_index, 1]
        gz = grad[:, node_index, 2]
        col = 3 * node_index
        b_matrix[:, 0, col] = gx
        b_matrix[:, 1, col + 1] = gy
        b_matrix[:, 2, col + 2] = gz
        b_matrix[:, 3, col] = gy
        b_matrix[:, 3, col + 1] = gx
        b_matrix[:, 4, col + 1] = gz
        b_matrix[:, 4, col + 2] = gy
        b_matrix[:, 5, col] = gz
        b_matrix[:, 5, col + 2] = gx
    return b_matrix


def _element_stiffness(b_matrix: np.ndarray, volume: np.ndarray, youngs: np.ndarray, d0: np.ndarray) -> np.ndarray:
    d_b = np.einsum("ij,ejk->eik", d0, b_matrix)
    stiffness = np.einsum("eji,ejk->eik", b_matrix, d_b)
    stiffness *= (volume * youngs)[:, None, None]
    return stiffness


def _assemble_stiffness(tets: np.ndarray, ke: np.ndarray, n_dof: int):
    n_elem = tets.shape[0]
    dof = np.empty((n_elem, 12), dtype=np.int64)
    dof[:, 0::3] = tets * 3
    dof[:, 1::3] = tets * 3 + 1
    dof[:, 2::3] = tets * 3 + 2
    rows = np.broadcast_to(dof[:, :, None], (n_elem, 12, 12)).reshape(-1)
    cols = np.broadcast_to(dof[:, None, :], (n_elem, 12, 12)).reshape(-1)
    return coo_matrix((ke.ravel(), (rows, cols)), shape=(n_dof, n_dof)).tocsr()


def _one_based_to_zero(ids: Optional[Sequence[int]], n_nodes: int) -> np.ndarray:
    if not ids:
        return np.empty((0,), dtype=np.int64)
    values = np.asarray(list(ids), dtype=np.int64)
    if values.size == 0:
        return values
    if int(values.min()) >= 1:
        values = values - 1
    return values[(values >= 0) & (values < n_nodes)]


def _von_mises(stress: np.ndarray) -> np.ndarray:
    sxx, syy, szz, txy, tyz, txz = (stress[:, i] for i in range(6))
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (txy**2 + tyz**2 + txz**2)
    )


def _principal_strain(strain: np.ndarray) -> np.ndarray:
    tensor = np.zeros((strain.shape[0], 3, 3), dtype=float)
    tensor[:, 0, 0] = strain[:, 0]
    tensor[:, 1, 1] = strain[:, 1]
    tensor[:, 2, 2] = strain[:, 2]
    tensor[:, 0, 1] = tensor[:, 1, 0] = 0.5 * strain[:, 3]
    tensor[:, 1, 2] = tensor[:, 2, 1] = 0.5 * strain[:, 4]
    tensor[:, 0, 2] = tensor[:, 2, 0] = 0.5 * strain[:, 5]
    return np.max(np.linalg.eigvalsh(tensor), axis=1)


def solve_linear_tet_fea(
    points: np.ndarray,
    tets: np.ndarray,
    youngs_modulus: np.ndarray,
    poisson_ratio: float,
    distal_node_ids: Sequence[int],
    proximal_node_ids: Sequence[int],
    peak_force_n: float,
    lateral_force_n: float,
    brace_node_ids: Optional[Sequence[int]] = None,
    rtol: float = 1.0e-4,
    maxiter: int = 2000,
) -> LinearTetFeaResult:
    points = np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    youngs = np.maximum(np.asarray(youngs_modulus, dtype=float), 1.0)
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("Expected a tetrahedral connectivity array of shape (n_elem, 4).")
    if youngs.shape[0] != tets.shape[0]:
        raise ValueError("Young's modulus must be defined per tetrahedral element.")

    n_nodes = int(points.shape[0])
    n_dof = n_nodes * 3
    nu = float(np.clip(poisson_ratio, 0.0, 0.49))
    d0 = _isotropic_d_matrix(nu)
    grad, volume = _tet_gradients(points, tets)
    b_matrix = _strain_displacement(grad)
    ke = _element_stiffness(b_matrix, volume, youngs, d0)
    stiffness = _assemble_stiffness(tets, ke, n_dof)
    diag = np.asarray(stiffness.diagonal(), dtype=float)
    stiffness = stiffness + diags(1.0e-10 * np.maximum(np.max(np.abs(diag)), 1.0) * np.ones(n_dof))

    distal = _one_based_to_zero(distal_node_ids, n_nodes)
    proximal = _one_based_to_zero(proximal_node_ids, n_nodes)
    brace = _one_based_to_zero(brace_node_ids or [], n_nodes)
    z_coords = points[:, 2]
    z_min = float(z_coords.min())
    z_max = float(z_coords.max())
    span = max(z_max - z_min, 1.0e-6)
    if distal.size == 0:
        distal = np.where(z_coords <= z_min + 0.08 * span)[0]
    if proximal.size == 0:
        proximal = np.where(z_coords >= z_max - 0.08 * span)[0]

    fixed = np.zeros(n_dof, dtype=bool)
    for node in distal:
        fixed[3 * int(node) : 3 * int(node) + 3] = True
    for node in brace:
        fixed[3 * int(node)] = True
        fixed[3 * int(node) + 1] = True
    if not np.any(~fixed):
        raise RuntimeError("Linear tetrahedral FEA has no free degrees of freedom.")

    loads = np.zeros(n_dof, dtype=float)
    n_prox = max(1, int(proximal.size))
    axial = float(peak_force_n) / n_prox
    lateral = float(lateral_force_n) / n_prox
    for node in proximal:
        loads[3 * int(node)] += lateral
        loads[3 * int(node) + 2] -= axial

    free = ~fixed
    k_ff = stiffness[free][:, free]
    f_free = loads[free]
    n_free = int(np.count_nonzero(free))
    iterations = 0
    solver_name = "cg"
    if n_free <= 30000:
        u_free = np.asarray(spsolve(k_ff, f_free), dtype=float)
        solver_name = "direct"
        residual = float(np.linalg.norm(k_ff @ u_free - f_free))
    else:
        jacobi = np.asarray(k_ff.diagonal(), dtype=float)
        jacobi = np.where(np.abs(jacobi) > 1.0e-12, jacobi, 1.0)
        preconditioner = diags(1.0 / jacobi)
        try:
            u_free, info = cg(k_ff, f_free, M=preconditioner, rtol=float(rtol), atol=0.0, maxiter=int(maxiter))
        except TypeError:
            try:
                u_free, info = cg(k_ff, f_free, M=preconditioner, tol=float(rtol), atol=0.0, maxiter=int(maxiter))
            except TypeError:
                u_free, info = cg(k_ff, f_free, M=preconditioner, tol=float(rtol), maxiter=int(maxiter))
        u_free = np.asarray(u_free, dtype=float)
        residual = float(np.linalg.norm(k_ff @ u_free - f_free))
        rhs_norm = max(float(np.linalg.norm(f_free)), 1.0e-12)
        if info != 0 and residual / rhs_norm > 0.05:
            raise RuntimeError(
                f"Linear tetrahedral FEA did not converge (cg info={info}, rel residual={residual / rhs_norm:.3e})."
            )
        iterations = int(info if info > 0 else 0)

    displacement = np.zeros(n_dof, dtype=float)
    displacement[free] = u_free
    disp_nodes = displacement.reshape((n_nodes, 3))

    elem_disp = np.empty((tets.shape[0], 12), dtype=float)
    elem_disp[:, 0::3] = disp_nodes[tets][:, :, 0]
    elem_disp[:, 1::3] = disp_nodes[tets][:, :, 1]
    elem_disp[:, 2::3] = disp_nodes[tets][:, :, 2]
    strain = np.einsum("eij,ej->ei", b_matrix, elem_disp)
    stress = youngs[:, None] * np.einsum("ij,ej->ei", d0, strain)
    von_mises = _von_mises(stress)
    rhs_norm = max(float(np.linalg.norm(f_free)), 1.0e-12)
    return LinearTetFeaResult(
        displacement=disp_nodes,
        von_mises=von_mises,
        principal_strain=_principal_strain(strain),
        residual_norm=residual / rhs_norm,
        iterations=iterations,
        solver=solver_name,
        stats={
            "n_nodes": float(n_nodes),
            "n_elements": float(tets.shape[0]),
            "n_free_dof": float(n_free),
            "max_displacement_mm": float(np.max(np.linalg.norm(disp_nodes, axis=1))),
            "max_von_mises_mpa": float(np.max(von_mises)) if von_mises.size else 0.0,
        },
    )


def solve_mesh_linear_tet_fea(mesh: Any, febio_setup: Any, config: Dict) -> LinearTetFeaResult:
    points = np.asarray(mesh.points, dtype=float)
    tets = tetra_connectivity(mesh)
    cell_data = getattr(mesh, "cell_data", None) or mesh.cell_data
    youngs = np.asarray(cell_data["youngs_modulus_mpa"], dtype=float)
    poisson = float(config.get("materials", {}).get("poisson_ratio", 0.30))
    node_sets = getattr(febio_setup, "node_sets", {}) or {}
    load_summary = getattr(febio_setup, "load_summary", {}) or {}
    sim_cfg = config.get("simulation", {})
    return solve_linear_tet_fea(
        points,
        tets,
        youngs,
        poisson,
        distal_node_ids=node_sets.get("distal_nodes", []),
        proximal_node_ids=node_sets.get("proximal_nodes", []),
        peak_force_n=float(load_summary.get("peak_force_n", 0.0)),
        lateral_force_n=float(load_summary.get("lateral_force_n", 0.0)),
        brace_node_ids=node_sets.get("brace_support_nodes", []),
        rtol=float(sim_cfg.get("internal_fea_rtol", 1.0e-4)),
        maxiter=int(sim_cfg.get("internal_fea_maxiter", 2000)),
    )
