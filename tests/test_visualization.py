from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cpt_predictor.models import SegmentationResult, StudyData
from cpt_predictor.visualization import ResultVisualizer


def _cylinder_case(shape=(64, 28, 22), radius=7.0):
    zz, yy, xx = np.indices(shape)
    cy, cx = shape[1] / 2.0, shape[2] / 2.0
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
    mask[:4] = False
    mask[-4:] = False
    study = StudyData(
        volume=np.zeros(shape, dtype=np.float32),
        hu_volume=np.where(mask, 900.0, -800.0).astype(np.float32),
        spacing_zyx=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    segmentation = SegmentationResult(mask=mask, method="test")
    rng = np.random.default_rng(0)
    inside = np.argwhere(mask)
    sample = inside[rng.choice(inside.shape[0], size=min(800, inside.shape[0]), replace=False)]
    centers = np.column_stack([sample[:, 2], sample[:, 1], sample[:, 0]]).astype(float)
    z = centers[:, 2]
    stress = 4.0 + 18.0 * np.exp(-((z - shape[0] / 2.0) ** 2) / (2 * 4.0**2))
    return study, segmentation, centers, stress


def test_ap_lateral_views_follow_bone_silhouette_not_a_filled_rectangle():
    visualizer = ResultVisualizer({})
    study, segmentation, centers, stress = _cylinder_case()
    views = visualizer.project_bone_stress_views(centers, stress, study=study, segmentation=segmentation)
    ap = views["ap"]
    lat = views["lateral"]
    assert ap.shape[0] > ap.shape[1]
    assert lat.shape[0] > lat.shape[1]

    ap_finite = np.isfinite(ap)
    lat_finite = np.isfinite(lat)
    assert 0.05 < ap_finite.mean() < 0.75
    assert 0.05 < lat_finite.mean() < 0.75
    # Corners of the bounding box are empty space, not interpolated stress.
    assert not ap_finite[0, 0]
    assert not ap_finite[0, -1]
    assert not lat_finite[-1, 0]
    # Mid-shaft rows should be wider than a 1-pixel horizontal stripe.
    row_widths = ap_finite.sum(axis=1)
    assert int(np.max(row_widths)) >= 8
    assert int(np.count_nonzero(row_widths)) > ap.shape[0] // 3
    # Peak stress is inside the silhouette, not a box-filling band.
    peak = np.nanargmax(np.nan_to_num(ap, nan=-1.0))
    pz, px = np.unravel_index(peak, ap.shape)
    assert ap_finite[pz, px]
    assert 0.25 * ap.shape[0] < pz < 0.75 * ap.shape[0]


def test_interior_extrema_ignore_boundary_condition_bands():
    visualizer = ResultVisualizer({})
    n = 100
    z = np.linspace(0.0, 100.0, n)
    centers = np.column_stack([np.zeros(n), np.zeros(n), z])
    stress = np.ones(n)
    stress[0] = 50.0
    stress[-1] = 40.0
    stress[50] = 12.0
    safety = np.full(n, 5.0)
    safety[2] = 0.1
    safety[55] = 1.2
    modulus = np.full(n, 5000.0)
    modulus[55] = 200.0
    hotspot, weakest = visualizer.select_interior_extrema(centers, stress, safety=safety, modulus=modulus)
    assert 0.12 < (z[hotspot] / 100.0) < 0.88
    assert hotspot == 50
    assert weakest == 50 or 0.12 < (z[weakest] / 100.0) < 0.88
    assert z[weakest] != z[2]
