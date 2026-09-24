"""Geometric tib/fib strength from clinical MRI.

Standard MRI does not measure bone mineral density. Cortical bone is the dark
ring inside the limb. It is given one cortical modulus and yield strength, and
the same properties are applied to the normal-leg cortex, so the percent
difference is geometry rather than a fabricated density.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .errors import StrengthAnalysisError


UNIFORM_CORTEX_MODULUS_MPA = 17000.0
UNIFORM_CORTEX_YIELD_MPA = 120.0


def _long_components(mask: np.ndarray, min_length_fraction: float = 0.45) -> np.ndarray:
    labeled, count = ndi.label(mask)
    if count == 0:
        return mask
    height = mask.shape[0]
    keep = np.zeros(count + 1, dtype=bool)
    slices = ndi.find_objects(labeled)
    for label, region in enumerate(slices, start=1):
        if region is None:
            continue
        z_extent = region[0].stop - region[0].start
        if z_extent >= min_length_fraction * height:
            keep[label] = True
    if not np.any(keep):
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        keep[int(np.argmax(sizes))] = True
    return keep[labeled]


def segment_mri_cortex(volume: np.ndarray) -> np.ndarray:
    """Segment the dark cortical ring enclosed by brighter marrow and muscle."""
    values = np.asarray(volume, dtype=float)
    finite = np.isfinite(values)
    if int(finite.sum()) < 32:
        raise StrengthAnalysisError(
            "MRI cortical bone could not be segmented, so no strength percent was calculated.",
            modality="mri",
        )
    sample = values[finite]
    low, high = (float(value) for value in np.percentile(sample, [5, 95]))
    spread = max(high - low, 1.0)
    tissue = sample[sample > low + 0.01 * spread]
    if tissue.size < 32:
        raise StrengthAnalysisError(
            "MRI cortical bone could not be segmented, so no strength percent was calculated.",
            modality="mri",
        )
    bright_cut = float(np.percentile(tissue, 40))
    if bright_cut <= low + 0.05 * spread:
        raise StrengthAnalysisError(
            "MRI cortical bone could not be segmented, so no strength percent was calculated.",
            modality="mri",
        )
    bright = finite & (values >= bright_cut)
    dark_max = low + 0.55 * (bright_cut - low)
    structure = np.ones((7, 7), dtype=bool)
    cortex = np.zeros(values.shape, dtype=bool)
    for index in range(values.shape[0]):
        closed = ndi.binary_closing(bright[index], structure=structure)
        filled = ndi.binary_fill_holes(closed)
        cortex[index] = (
            filled
            & ~bright[index]
            & finite[index]
            & (values[index] > low + 0.01 * spread)
            & (values[index] <= dark_max)
        )
    labeled, count = ndi.label(cortex)
    if count:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        cortex = (sizes >= 20)[labeled]
    cortex = _long_components(cortex, min_length_fraction=0.4)
    if int(cortex.sum()) < 30:
        raise StrengthAnalysisError(
            "MRI cortical bone could not be segmented, so no strength percent was calculated.",
            modality="mri",
        )
    return cortex
