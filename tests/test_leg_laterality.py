"""Patient left/right must follow LPS column direction, not image-left naming."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cpt_predictor.models import StudyData
from cpt_predictor.preprocessing import preprocess_study


def _bilateral_study() -> StudyData:
    """Patient-right shaft at low column index, patient-left at high index (LPS)."""
    shape = (72, 128, 128)
    volume = np.full(shape, -1000.0, dtype=np.float32)
    z, y, x = np.indices(shape)

    # ImageOrientationPatient [1,0,0, 0,1,0]: +columns = +X = patient left.
    patient_right_femur = (z < 20) & (((x - 34) ** 2 + (y - 48) ** 2) <= 8**2)
    patient_left_femur = (z < 20) & (((x - 96) ** 2 + (y - 48) ** 2) <= 8**2)
    patient_right_tibia = (z >= 20) & (z < 62) & (((x - 30) ** 2 + (y - 46) ** 2) <= 6**2)
    patient_right_fibula = (z >= 20) & (z < 62) & (((x - 42) ** 2 + (y - 46) ** 2) <= 4**2)
    patient_left_tibia = (z >= 20) & (z < 62) & (((x - 88) ** 2 + (y - 46) ** 2) <= 6**2)
    patient_left_fibula = (z >= 20) & (z < 62) & (((x - 102) ** 2 + (y - 46) ** 2) <= 4**2)

    volume[
        patient_right_femur
        | patient_left_femur
        | patient_right_tibia
        | patient_right_fibula
        | patient_left_tibia
        | patient_left_fibula
    ] = 950.0
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


def test_selecting_patient_left_keeps_high_column_shaft() -> None:
    study = _bilateral_study()
    processed = preprocess_study(study, _config("left"))
    bounds = processed.metadata["leg_localization"]["crop_bounds_zyx"]["x"]
    # Patient left lives near x≈88–102; the crop must cover that side, not x≈30.
    assert bounds[0] >= 60
    assert bounds[1] > 96


def test_selecting_patient_right_keeps_low_column_shaft() -> None:
    study = _bilateral_study()
    processed = preprocess_study(study, _config("right"))
    bounds = processed.metadata["leg_localization"]["crop_bounds_zyx"]["x"]
    assert bounds[1] <= 70
    assert bounds[0] < 40
