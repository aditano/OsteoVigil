"""Deterministic normal adult tibia and fibula shaft used as the comparison leg.

Cross-section follows published midshaft proportions: a tibial periosteal
radius near 13 mm with about 5 mm of cortex, and a narrower fibula offset
laterally. The generated length matches the scanned bone instead of squashing
a full 37 cm tibia into a short field of view.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


TIBIA_RADIUS_MM = 13.0
TIBIA_CORTEX_MM = 5.0
FIBULA_RADIUS_MM = 6.5
FIBULA_CORTEX_MM = 2.4
FIBULA_OFFSET_MM = 24.0
CORTICAL_HU = 1500.0
TRABECULAR_HU = 350.0
OUTSIDE_HU = -500.0


def synthetic_tibfib_volume(
    length_mm: float,
    spacing_mm: float = 2.0,
    cortical_hu: float = CORTICAL_HU,
    trabecular_hu: float = TRABECULAR_HU,
    tibia_radius_mm: float = TIBIA_RADIUS_MM,
    tibia_cortex_mm: float = TIBIA_CORTEX_MM,
    fibula_radius_mm: float = FIBULA_RADIUS_MM,
    fibula_cortex_mm: float = FIBULA_CORTEX_MM,
    fibula_offset_mm: float = FIBULA_OFFSET_MM,
    include_fibula: bool = True,
    notch_center_mm: Optional[float] = None,
    notch_depth_mm: float = 0.0,
    notch_half_width_mm: float = 3.0,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    spacing = float(spacing_mm)
    if spacing <= 0 or length_mm <= spacing * 4:
        raise ValueError("Reference length must cover several voxels.")
    margin = 6.0
    x_min = -float(tibia_radius_mm) - margin
    x_max = float(tibia_radius_mm) + margin
    if include_fibula:
        x_max = max(x_max, float(fibula_offset_mm) + float(fibula_radius_mm) + margin)
    y_min = -float(tibia_radius_mm) - margin
    y_max = float(tibia_radius_mm) + margin
    nx = max(8, int(np.ceil((x_max - x_min) / spacing)))
    ny = max(8, int(np.ceil((y_max - y_min) / spacing)))
    nz = max(8, int(np.round(float(length_mm) / spacing)))

    z_index, y_index, x_index = np.indices((nz, ny, nx))
    x_mm = x_min + (x_index + 0.5) * spacing
    y_mm = y_min + (y_index + 0.5) * spacing
    z_mm = (z_index + 0.5) * spacing
    tibia_radius = np.sqrt(x_mm**2 + y_mm**2)
    volume = np.full((nz, ny, nx), OUTSIDE_HU, dtype=np.float32)
    tibia_cortex = (tibia_radius <= tibia_radius_mm) & (tibia_radius >= tibia_radius_mm - tibia_cortex_mm)
    tibia_core = tibia_radius < (tibia_radius_mm - tibia_cortex_mm)
    volume[tibia_cortex] = np.float32(cortical_hu)
    volume[tibia_core] = np.float32(trabecular_hu)

    if include_fibula:
        fibula_radius = np.sqrt((x_mm - float(fibula_offset_mm)) ** 2 + y_mm**2)
        fibula_cortex = (fibula_radius <= fibula_radius_mm) & (fibula_radius >= fibula_radius_mm - fibula_cortex_mm)
        fibula_core = fibula_radius < (fibula_radius_mm - fibula_cortex_mm)
        volume[fibula_cortex] = np.float32(cortical_hu)
        volume[fibula_core] = np.float32(trabecular_hu)

    if notch_center_mm is not None and notch_depth_mm > 0:
        band = np.abs(z_mm - float(notch_center_mm)) <= float(notch_half_width_mm)
        removed = band & (x_mm > (tibia_radius_mm - float(notch_depth_mm))) & (tibia_radius <= tibia_radius_mm + 0.1)
        volume[removed] = np.float32(OUTSIDE_HU)

    return volume, (spacing, spacing, spacing)


def build_reference_volume(length_mm: float, spacing_mm: float) -> tuple[np.ndarray, tuple[float, float, float]]:
    return synthetic_tibfib_volume(length_mm=length_mm, spacing_mm=spacing_mm)
