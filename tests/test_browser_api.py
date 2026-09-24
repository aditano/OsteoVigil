"""The Pyodide entry point returns the same public report as the local API."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cpt_predictor.browser_api import analyze_upload
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
