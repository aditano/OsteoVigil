"""Voxel hexahedral linear-elastic solver for vertical tib/fib loads.

Each bone voxel is an 8-node brick. Distal nodes are fixed and a proximal
axial force is applied. Because the model is linear, the load at which a
critical fraction of cortical voxels yields is the applied force times
yield / stress. This is a quantitative CT finite-element estimate, not the
analytic surrogate used by the legacy CPT path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg, spsolve


GRAVITY_M_S2 = 9.80665
CORTICAL_MODULUS_MPA = 8000.0
END_EXCLUSION = 0.08
CRITICAL_VOLUME_FRACTION = 0.02


@dataclass
class VoxelSolveResult:
    failure_load_n: float
    applied_force_n: float
    stiffness_n_per_mm: float
    mean_proximal_displacement_mm: float
    utilization: np.ndarray
    von_mises_mpa: np.ndarray
    element_k: np.ndarray
    element_j: np.ndarray
    element_i: np.ndarray
    cortical: np.ndarray
    grid_shape: tuple[int, int, int]
    spacing_zyx: tuple[float, float, float]
    n_elements: int
    n_nodes: int
    solver: str
    residual_norm: float


def body_weight_newtons(mass_kg: float) -> float:
    if mass_kg <= 0:
        raise ValueError("Body mass must be positive.")
    return float(mass_kg) * GRAVITY_M_S2


def _isotropic_d_over_e(nu: float) -> np.ndarray:
    lam = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = 1.0 / (2.0 * (1.0 + nu))
    matrix = np.zeros((6, 6), dtype=float)
    matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = lam + 2.0 * mu
    matrix[0, 1] = matrix[0, 2] = matrix[1, 0] = matrix[1, 2] = matrix[2, 0] = matrix[2, 1] = lam
    matrix[3, 3] = matrix[4, 4] = matrix[5, 5] = mu
    return matrix


def _shape_deriv_natural(xi: float, eta: float, zeta: float) -> np.ndarray:
    signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    d_xi = 0.125 * signs[:, 0] * (1.0 + signs[:, 1] * eta) * (1.0 + signs[:, 2] * zeta)
    d_eta = 0.125 * signs[:, 1] * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 2] * zeta)
    d_zeta = 0.125 * signs[:, 2] * (1.0 + signs[:, 0] * xi) * (1.0 + signs[:, 1] * eta)
    return np.stack([d_xi, d_eta, d_zeta], axis=1)


def _b_matrix(deriv_xyz: np.ndarray) -> np.ndarray:
    matrix = np.zeros((6, 24), dtype=float)
    for node in range(8):
        dx, dy, dz = (float(deriv_xyz[node, axis]) for axis in range(3))
        col = 3 * node
        matrix[0, col] = dx
        matrix[1, col + 1] = dy
        matrix[2, col + 2] = dz
        matrix[3, col] = dy
        matrix[3, col + 1] = dx
        matrix[4, col + 1] = dz
        matrix[4, col + 2] = dy
        matrix[5, col] = dz
        matrix[5, col + 2] = dx
    return matrix


def voxel_element_operators(hx: float, hy: float, hz: float, poisson_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return K for E=1, the centroid B matrix, and D/E for a rectangular brick."""
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [hx, 0.0, 0.0],
            [hx, hy, 0.0],
            [0.0, hy, 0.0],
            [0.0, 0.0, hz],
            [hx, 0.0, hz],
            [hx, hy, hz],
            [0.0, hy, hz],
        ],
        dtype=float,
    )
    d_over_e = _isotropic_d_over_e(float(np.clip(poisson_ratio, 0.0, 0.49)))
    gauss = 1.0 / np.sqrt(3.0)
    stiffness = np.zeros((24, 24), dtype=float)
    for xi in (-gauss, gauss):
        for eta in (-gauss, gauss):
            for zeta in (-gauss, gauss):
                deriv_nat = _shape_deriv_natural(float(xi), float(eta), float(zeta))
                jacobian = corners.T @ deriv_nat
                det_j = float(np.linalg.det(jacobian))
                deriv_xyz = deriv_nat @ np.linalg.inv(jacobian)
                operator = _b_matrix(deriv_xyz)
                stiffness += (operator.T @ d_over_e @ operator) * det_j
    stiffness = 0.5 * (stiffness + stiffness.T)
    center_deriv = _shape_deriv_natural(0.0, 0.0, 0.0)
    center_j = corners.T @ center_deriv
    b_center = _b_matrix(center_deriv @ np.linalg.inv(center_j))
    return stiffness, b_center, d_over_e


def _von_mises(stress: np.ndarray) -> np.ndarray:
    sxx, syy, szz, txy, tyz, txz = (stress[:, index] for index in range(6))
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (txy**2 + tyz**2 + txz**2)
    )


def _critical_multiplier(utilization: np.ndarray, include: np.ndarray, fraction: float) -> float:
    values = utilization[include]
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise RuntimeError("The voxel model produced no stressed cortical bone.")
    n_crit = max(1, int(np.ceil(float(fraction) * values.size)))
    order_index = values.size - n_crit
    threshold = float(np.partition(values, order_index)[order_index])
    if threshold <= 1.0e-12:
        raise RuntimeError("Cortical utilization is zero, so a failure load cannot be scaled.")
    return 1.0 / threshold


def _solve_free(stiffness, loads: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, str, float]:
    free = ~fixed
    if not np.any(free):
        raise RuntimeError("Voxel FEA has no free degrees of freedom.")
    k_ff = stiffness[free][:, free]
    force = loads[free]
    n_free = int(np.count_nonzero(free))
    solver_name = "direct"
    if n_free <= 35000:
        displacement = np.asarray(spsolve(k_ff.tocsc(), force), dtype=float)
    else:
        solver_name = "cg"
        diagonal = np.asarray(k_ff.diagonal(), dtype=float)
        diagonal = np.where(np.abs(diagonal) > 1.0e-12, diagonal, 1.0)
        try:
            displacement, info = cg(k_ff, force, M=diags(1.0 / diagonal), rtol=1.0e-5, atol=0.0, maxiter=4000)
        except TypeError:
            displacement, info = cg(k_ff, force, M=diags(1.0 / diagonal), tol=1.0e-5, atol=0.0, maxiter=4000)
        displacement = np.asarray(displacement, dtype=float)
        if info != 0:
            residual = float(np.linalg.norm(k_ff @ displacement - force))
            rhs = max(float(np.linalg.norm(force)), 1.0e-12)
            if residual / rhs > 0.05:
                raise RuntimeError(f"Voxel FEA did not converge (cg info={info}).")
    residual_vec = k_ff @ displacement - force
    rhs_norm = max(float(np.linalg.norm(force)), 1.0e-12)
    full = np.zeros(loads.shape, dtype=float)
    full[free] = displacement
    return full, solver_name, float(np.linalg.norm(residual_vec) / rhs_norm)


def solve_voxel_elasticity(
    modulus_mpa: np.ndarray,
    yield_mpa: np.ndarray,
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    force_n: float,
    poisson_ratio: float = 0.30,
    critical_fraction: float = CRITICAL_VOLUME_FRACTION,
    cortical_modulus_mpa: float = CORTICAL_MODULUS_MPA,
) -> VoxelSolveResult:
    """Solve axial compression. Index 0 along z is the distal (fixed) end."""
    modulus = np.asarray(modulus_mpa, dtype=float)
    strength = np.asarray(yield_mpa, dtype=float)
    bone = np.asarray(mask, dtype=bool) & np.isfinite(modulus) & (modulus > 1.0)
    if bone.ndim != 3:
        raise ValueError("The bone mask must be a 3D volume.")
    if modulus.shape != bone.shape or strength.shape != bone.shape:
        raise ValueError("Modulus, yield, and mask must share one shape.")
    if int(bone.sum()) < 8:
        raise RuntimeError("Not enough bone voxels to build a finite-element model.")
    if force_n <= 0:
        raise ValueError("Axial force must be positive.")

    hz, hy, hx = (float(value) for value in spacing_zyx)
    k_unit, b_center, d_over_e = voxel_element_operators(hx, hy, hz, poisson_ratio)
    k_idx, j_idx, i_idx = np.nonzero(bone)
    n_elem = int(k_idx.size)
    ny1 = bone.shape[1] + 1
    nx1 = bone.shape[2] + 1
    stride = ny1 * nx1

    def _node(k: np.ndarray, j: np.ndarray, i: np.ndarray) -> np.ndarray:
        return (k * stride) + (j * nx1) + i

    corners = np.stack(
        [
            _node(k_idx, j_idx, i_idx),
            _node(k_idx, j_idx, i_idx + 1),
            _node(k_idx, j_idx + 1, i_idx + 1),
            _node(k_idx, j_idx + 1, i_idx),
            _node(k_idx + 1, j_idx, i_idx),
            _node(k_idx + 1, j_idx, i_idx + 1),
            _node(k_idx + 1, j_idx + 1, i_idx + 1),
            _node(k_idx + 1, j_idx + 1, i_idx),
        ],
        axis=1,
    )
    _unique, inverse = np.unique(corners.ravel(), return_inverse=True)
    elem_nodes = inverse.reshape(n_elem, 8)
    n_nodes = int(elem_nodes.max()) + 1
    node_k = _unique // stride
    moduli = modulus[k_idx, j_idx, i_idx]
    yields = np.maximum(strength[k_idx, j_idx, i_idx], 1.0e-3)

    dof = np.empty((n_elem, 24), dtype=np.int64)
    dof[:, 0::3] = elem_nodes * 3
    dof[:, 1::3] = elem_nodes * 3 + 1
    dof[:, 2::3] = elem_nodes * 3 + 2
    local = moduli[:, None, None] * k_unit[None, :, :]
    rows = np.broadcast_to(dof[:, :, None], (n_elem, 24, 24)).reshape(-1)
    cols = np.broadcast_to(dof[:, None, :], (n_elem, 24, 24)).reshape(-1)
    n_dof = n_nodes * 3
    stiffness = coo_matrix((local.reshape(-1), (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    diagonal = np.asarray(stiffness.diagonal(), dtype=float)
    stiffness = stiffness + diags(1.0e-10 * max(float(np.max(np.abs(diagonal))), 1.0) * np.ones(n_dof))

    k_min = int(node_k.min())
    k_max = int(node_k.max())
    if k_min == k_max:
        raise RuntimeError("Bone mask is only one voxel thick along the shaft.")
    distal = np.flatnonzero(node_k == k_min)
    proximal = np.flatnonzero(node_k == k_max)
    fixed = np.zeros(n_dof, dtype=bool)
    fixed[distal * 3] = True
    fixed[distal * 3 + 1] = True
    fixed[distal * 3 + 2] = True
    loads = np.zeros(n_dof, dtype=float)
    loads[proximal * 3 + 2] = -float(force_n) / float(proximal.size)

    displacement_flat, solver_name, residual = _solve_free(stiffness, loads, fixed)
    displacement = displacement_flat.reshape((n_nodes, 3))
    elem_disp = displacement[elem_nodes].reshape(n_elem, 24)
    strain = np.einsum("ij,ej->ei", b_center, elem_disp)
    stress = moduli[:, None] * np.einsum("ij,ej->ei", d_over_e, strain)
    von_mises = _von_mises(stress)
    utilization = von_mises / yields

    cortical = moduli >= float(cortical_modulus_mpa)
    if int(cortical.sum()) < max(8, int(0.05 * n_elem)):
        cortical = np.ones(n_elem, dtype=bool)
    z_rel = (k_idx.astype(float) + 0.5) / float(bone.shape[0])
    interior = (z_rel >= END_EXCLUSION) & (z_rel <= 1.0 - END_EXCLUSION)
    assessed = cortical & interior
    if int(assessed.sum()) < 8:
        assessed = cortical
    multiplier = _critical_multiplier(utilization, assessed, critical_fraction)
    mean_uz = float(np.mean(displacement[proximal, 2]))
    stiffness_value = float(force_n / abs(mean_uz)) if abs(mean_uz) > 1.0e-12 else float("inf")
    return VoxelSolveResult(
        failure_load_n=float(force_n) * multiplier,
        applied_force_n=float(force_n),
        stiffness_n_per_mm=stiffness_value,
        mean_proximal_displacement_mm=mean_uz,
        utilization=utilization,
        von_mises_mpa=von_mises,
        element_k=k_idx.astype(np.int32),
        element_j=j_idx.astype(np.int32),
        element_i=i_idx.astype(np.int32),
        cortical=assessed,
        grid_shape=tuple(int(value) for value in bone.shape),
        spacing_zyx=(hz, hy, hx),
        n_elements=n_elem,
        n_nodes=n_nodes,
        solver=f"voxel_hexahedral_{solver_name}",
        residual_norm=residual,
    )
