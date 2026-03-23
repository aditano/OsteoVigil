from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODULE_NAME = "cpt_predictor.segmentation"


if importlib.util.find_spec(MODULE_NAME) is None:
    pytest.skip("segmentation module not yet implemented", allow_module_level=True)

segmentation = importlib.import_module(MODULE_NAME)


def _get_segmentation_fn():
    if hasattr(segmentation, "TibiaSegmenter"):
        return segmentation.TibiaSegmenter
    for name in [
        "classical_tibia_segmentation",
        "classical_segment_tibia",
        "segment_tibia_classical",
        "segment_tibia",
        "classical_segmentation",
    ]:
        if hasattr(segmentation, name):
            return getattr(segmentation, name)
    raise AttributeError("No expected tibia segmentation function found")


def _synthetic_ct_with_tibia_and_pseudarthrosis(shape=(72, 72, 96)):
    volume = np.full(shape, -900.0, dtype=np.float32)
    z, y, x = np.indices(shape)
    center_y = shape[1] // 2
    center_x = shape[2] // 2
    radius_outer = 12
    radius_inner = 4
    radial = (y - center_y) ** 2 + (x - center_x) ** 2
    tibia = radial <= radius_outer**2
    canal = radial <= radius_inner**2
    tibia &= ~canal

    volume[tibia] = 950.0

    gap_start = shape[0] // 2 - 4
    gap_end = shape[0] // 2 + 4
    volume[gap_start:gap_end, :, :][tibia[gap_start:gap_end, :, :]] = 120.0
    volume[:gap_start, :, :][tibia[:gap_start, :, :]] = 950.0
    volume[gap_end:, :, :][tibia[gap_end:, :, :]] = 950.0
    return volume


def test_classical_segmentation_recovers_tibial_structure():
    segmenter_or_fn = _get_segmentation_fn()
    volume = _synthetic_ct_with_tibia_and_pseudarthrosis()

    if inspect.isclass(segmenter_or_fn):
        try:
            segmenter = segmenter_or_fn(bone_threshold_hu=180, bridge_gap_slices=4)
        except TypeError:
            segmenter = segmenter_or_fn()
        mask = segmenter.segment(volume)
    else:
        try:
            mask = segmenter_or_fn(volume, bone_threshold_hu=180, bridge_gap_slices=4)
        except TypeError:
            mask = segmenter_or_fn(volume)

    mask = np.asarray(mask, dtype=bool)
    assert mask.shape == volume.shape
    assert mask.sum() > 0

    expected_bone = volume > 180
    overlap = np.logical_and(mask, expected_bone).sum()
    dice = 2.0 * overlap / (mask.sum() + expected_bone.sum())
    assert dice > 0.8


def test_segmentation_handles_gap_region_without_breaking_shape():
    segmenter_or_fn = _get_segmentation_fn()
    volume = _synthetic_ct_with_tibia_and_pseudarthrosis()
    gap_start = volume.shape[0] // 2 - 4
    gap_end = volume.shape[0] // 2 + 4

    if inspect.isclass(segmenter_or_fn):
        try:
            segmenter = segmenter_or_fn(bone_threshold_hu=180, bridge_gap_slices=6)
        except TypeError:
            segmenter = segmenter_or_fn()
        mask = segmenter.segment(volume)
    else:
        try:
            mask = segmenter_or_fn(volume, bone_threshold_hu=180, bridge_gap_slices=6)
        except TypeError:
            mask = segmenter_or_fn(volume)

    mask = np.asarray(mask, dtype=bool)
    z_profile = mask.sum(axis=(1, 2))
    assert z_profile[:10].max() > 0
    assert z_profile[-10:].max() > 0
    assert mask.mean() < 0.2
    assert mask.dtype == bool or mask.dtype == np.bool_
