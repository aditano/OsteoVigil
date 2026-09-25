"""The Pyodide entry point returns the same public report as the local API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cpt_predictor.browser_api import analyze_upload, decode_next_dataset, prepare_study, solve_prepared_study
from cpt_predictor.io.dicom_export import write_dicom_series
from cpt_predictor.reference_leg import synthetic_tibfib_volume


def test_browser_api_analyzes_a_folder_of_dicom(tmp_path: Path) -> None:
    hu, spacing = synthetic_tibfib_volume(40, 2.5)
    folder = write_dicom_series(hu, spacing, tmp_path / "series", patient_weight_kg=70.0)
    payload = analyze_upload(str(folder), 70.0, True)
    assert "error" not in payload
    assert payload["modality"] == "ct"
    assert payload["method"] == "voxel_hexahedral_linear_elastic"
    assert "surrogate" not in payload["solver"]
    assert np.isfinite(payload["failure_load_n"])
    assert payload["failure_load_n"] > 0
    assert "weakness" in payload


def test_incremental_decode_matches_one_shot_upload(tmp_path: Path) -> None:
    hu, spacing = synthetic_tibfib_volume(40, 2.5)
    folder = write_dicom_series(hu, spacing, tmp_path / "series", patient_weight_kg=70.0)
    full = analyze_upload(str(folder), 70.0, True)
    prepared = prepare_study(str(folder))
    assert prepared["count"] == int(hu.shape[0])
    seen = 0
    finished = False
    while not finished:
        step = decode_next_dataset()
        assert "error" not in step
        assert step["done"] >= seen
        seen = step["done"]
        finished = bool(step["finished"])
        assert seen <= prepared["count"]
    assert seen == prepared["count"]
    solved = solve_prepared_study(70.0, True)
    assert solved["failure_load_n"] == full["failure_load_n"]
    assert solved["percent_vs_normal"] == full["percent_vs_normal"]
    assert solved["modality"] == "ct"
