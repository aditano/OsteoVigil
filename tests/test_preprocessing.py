from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cpt_predictor.models import StudyData
from cpt_predictor.preprocessing import preprocess_study


def _synthetic_bilateral_full_body_study() -> StudyData:
    shape = (72, 128, 128)
    volume = np.full(shape, -1000.0, dtype=np.float32)
    z, y, x = np.indices(shape)

    left_femur = (z < 20) & (((x - 34) ** 2 + (y - 48) ** 2) <= 8**2)
    right_femur = (z < 20) & (((x - 96) ** 2 + (y - 48) ** 2) <= 8**2)

    left_tibia = (z >= 20) & (z < 62) & (((x - 30) ** 2 + (y - 46) ** 2) <= 6**2)
    left_fibula = (z >= 20) & (z < 62) & (((x - 42) ** 2 + (y - 46) ** 2) <= 4**2)
    right_tibia = (z >= 20) & (z < 62) & (((x - 88) ** 2 + (y - 46) ** 2) <= 6**2)
    right_fibula = (z >= 20) & (z < 62) & (((x - 102) ** 2 + (y - 46) ** 2) <= 4**2)

    volume[left_femur | right_femur | left_tibia | left_fibula | right_tibia | right_fibula] = 950.0
    return StudyData(
        volume=volume.copy(),
        hu_volume=volume.copy(),
        spacing_zyx=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        metadata={"loader": "synthetic"},
    )


def _synthetic_single_leg_study() -> StudyData:
    shape = (64, 72, 72)
    volume = np.full(shape, -1000.0, dtype=np.float32)
    z, y, x = np.indices(shape)

    tibia = (z >= 12) & (z < 52) & (((x - 24) ** 2 + (y - 36) ** 2) <= 6**2)
    fibula = (z >= 12) & (z < 52) & (((x - 40) ** 2 + (y - 36) ** 2) <= 4**2)

    volume[tibia | fibula] = 950.0
    return StudyData(
        volume=volume.copy(),
        hu_volume=volume.copy(),
        spacing_zyx=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        metadata={"loader": "synthetic"},
    )


def _synthetic_bilateral_with_outlier_slice() -> StudyData:
    shape = (72, 128, 192)
    volume = np.full(shape, -1000.0, dtype=np.float32)
    z, y, x = np.indices(shape)

    left_femur = (z < 20) & (((x - 42) ** 2 + (y - 48) ** 2) <= 8**2)
    right_femur = (z < 20) & (((x - 150) ** 2 + (y - 48) ** 2) <= 8**2)

    left_tibia = (z >= 20) & (z < 62) & (((x - 40) ** 2 + (y - 46) ** 2) <= 6**2)
    left_fibula = (z >= 20) & (z < 62) & (((x - 54) ** 2 + (y - 46) ** 2) <= 4**2)
    right_tibia = (z >= 20) & (z < 62) & (((x - 146) ** 2 + (y - 46) ** 2) <= 6**2)
    right_fibula = (z >= 20) & (z < 62) & (((x - 160) ** 2 + (y - 46) ** 2) <= 4**2)
    left_outlier = (z == 38) & (((x - 94) ** 2 + (y - 46) ** 2) <= 5**2)

    volume[left_femur | right_femur | left_tibia | left_fibula | right_tibia | right_fibula | left_outlier] = 950.0
    return StudyData(
        volume=volume.copy(),
        hu_volume=volume.copy(),
        spacing_zyx=(1.0, 1.0, 1.0),
        origin_xyz=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        metadata={"loader": "synthetic"},
    )


def _config(target_leg_side: str) -> dict:
    return {
        "input": {
            "allow_dummy_if_missing": False,
            "target_leg_side": target_leg_side,
            "enable_leg_localization": True,
        },
        "preprocessing": {
            "clip_range_hu": [-1000, 2500],
            "target_spacing_mm": [1.0, 1.0, 1.0],
            "full_body_extent_threshold_mm": 60.0,
            "leg_localization_bone_threshold_hu": 220,
            "leg_localization_min_component_area_px": 30,
            "leg_localization_min_bilateral_run_slices": 6,
            "leg_localization_min_tibfib_run_slices": 10,
            "leg_localization_max_tibfib_components": 4,
            "leg_localization_margin_mm": 8.0,
            "leg_localization_min_interleg_gap_mm": 35.0,
        },
    }


def test_preprocess_allows_auto_on_single_leg_tibfib_scan():
    study = _synthetic_single_leg_study()
    processed = preprocess_study(study, _config("auto"))

    assert processed.hu_volume.shape == study.hu_volume.shape
    localization = processed.metadata.get("leg_localization", {})
    assert localization.get("cropped") is False
    assert localization.get("scan_scope") == "single_leg"


def test_preprocess_requires_leg_selection_for_bilateral_full_body_scan():
    study = _synthetic_bilateral_full_body_study()

    with pytest.raises(ValueError, match="choose the left or right leg"):
        preprocess_study(study, _config("auto"))


def test_preprocess_crops_selected_leg_for_bilateral_full_body_scan():
    study = _synthetic_bilateral_full_body_study()
    processed = preprocess_study(study, _config("left"))

    assert processed.hu_volume.shape[2] < study.hu_volume.shape[2]
    assert processed.hu_volume.shape[0] < study.hu_volume.shape[0]
    localization = processed.metadata.get("leg_localization", {})
    assert localization.get("cropped") is True
    assert localization.get("target_leg_side") == "left"
    assert localization.get("scan_scope") == "full_body_or_multileg"


def test_preprocess_ignores_single_slice_outlier_when_cropping_selected_leg():
    study = _synthetic_bilateral_with_outlier_slice()
    processed = preprocess_study(study, _config("left"))

    assert processed.hu_volume.shape[2] < int(study.hu_volume.shape[2] * 0.45)
    localization = processed.metadata.get("leg_localization", {})
    assert localization.get("cropped") is True
    assert localization.get("target_leg_side") == "left"
