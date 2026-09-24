"""DICOM routing, API, synthetic defect, and the CC0 distal CT."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cpt_predictor.api import create_app
from cpt_predictor.io.dicom_export import write_dicom_series
from cpt_predictor.reference_leg import synthetic_tibfib_volume
from cpt_predictor.strength_io import load_dicom_path
from cpt_predictor.strength_pipeline import analyze_ct_volume, analyze_dicom_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DISTAL_CT = REPO_ROOT / "data" / "demo" / "normal_real_talocrural"
DEFECT_CT = REPO_ROOT / "data" / "demo" / "abnormal_synthetic_cpt" / "dicom"


def _plan_download():
    path = REPO_ROOT / "scripts" / "download_full_limb_ct.py"
    spec = importlib.util.spec_from_file_location("download_full_limb_ct", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plan_download


def test_download_plan_skips_unknown_and_oversized_files() -> None:
    plan_download = _plan_download()
    assert plan_download([]) == "skip-unknown"
    assert plan_download([142, 0]) == "skip-unknown"
    assert plan_download([500 * 1024 * 1024]) == "skip-too-large"
    assert plan_download([142, 1_641_479_123]) == "skip-too-large"
    assert plan_download([12 * 1024 * 1024, 900 * 1024 * 1024]) == "download"


def test_dicom_round_trip_keeps_modality_and_patient_weight(tmp_path: Path) -> None:
    hu, spacing = synthetic_tibfib_volume(40, 2.5)
    ct_dir = write_dicom_series(hu, spacing, tmp_path / "ct", patient_weight_kg=82.0)
    ct = load_dicom_path(ct_dir)
    assert ct["kind"] == "ct"
    assert ct["patient_weight_kg"] == pytest.approx(82.0)
    assert ct["volume"].shape[0] == hu.shape[0]

    mri_dir = write_dicom_series(hu, spacing, tmp_path / "mr", modality="MR")
    assert load_dicom_path(mri_dir)["kind"] == "mri"

    dx_dir = write_dicom_series(hu[hu.shape[0] // 2], (spacing[0], spacing[1], spacing[2]), tmp_path / "dx", modality="DX", view_position="LAT")
    radiograph = load_dicom_path(dx_dir)
    assert radiograph["kind"] == "radiograph"
    assert "lateral" in radiograph["views"]


def _repair_low_density_band(hu: np.ndarray) -> tuple[np.ndarray, float]:
    """Replace the low-HU shaft band with a healthy neighboring slice."""
    bone = np.asarray(hu) >= 180.0
    medians = np.full(hu.shape[0], np.nan, dtype=float)
    for index in range(hu.shape[0]):
        values = hu[index][bone[index]]
        if values.size > 20:
            medians[index] = float(np.median(values))
    present = np.flatnonzero(np.isfinite(medians))
    if present.size < 8:
        raise AssertionError("The synthetic series did not contain a bone shaft.")
    z0 = int(present[0])
    z1 = int(present[-1])
    interior = np.zeros(hu.shape[0], dtype=bool)
    margin = max(2, int(0.08 * (z1 - z0)))
    interior[z0 + margin : z1 - margin + 1] = True
    baseline = float(np.nanmedian(medians[interior]))
    defect = interior & np.isfinite(medians) & (medians < baseline * 0.55)
    if not np.any(defect):
        raise AssertionError("No low-density band was found in the synthetic series.")
    donors = np.flatnonzero(interior & ~defect & np.isfinite(medians) & (medians > baseline * 0.85))
    donor = int(donors[len(donors) // 2])
    intact = np.array(hu, copy=True)
    donor_slice = hu[donor]
    donor_bone = bone[donor]
    for index in np.flatnonzero(defect):
        intact[index] = np.where(donor_bone, donor_slice, intact[index])
    center = float(np.mean(np.flatnonzero(defect)))
    fraction = (center - z0) / max(1.0, float(z1 - z0))
    return intact, fraction


def test_api_analyzes_a_tiny_ct_and_rejects_an_empty_upload(tmp_path: Path) -> None:
    hu, spacing = synthetic_tibfib_volume(40, 2.5)
    folder = write_dicom_series(hu, spacing, tmp_path / "series", patient_weight_kg=70.0)
    uploads = [
        ("files", (path.name, path.read_bytes(), "application/dicom"))
        for path in sorted(folder.glob("*.dcm"))
    ]
    client = TestClient(create_app(tmp_path / "no-dist"))
    response = client.post("/api/analyze", files=uploads, data={"body_mass_kg": "65", "use_dicom_weight": "1"})
    assert response.status_code == 200
    body = response.json()
    assert body["modality"] == "ct"
    assert body["method"] == "voxel_hexahedral_linear_elastic"
    assert "surrogate" not in body["solver"]
    assert body["body_mass_kg"] == pytest.approx(70.0)
    assert np.isfinite(body["failure_load_n"])
    assert "weakness" in body

    rejected = client.post(
        "/api/analyze",
        files=[("files", ("note.txt", b"not dicom", "text/plain"))],
        data={"body_mass_kg": "70", "use_dicom_weight": "0"},
    )
    assert rejected.status_code == 422
    assert "error" in rejected.json()
    health = client.get("/api/health")
    assert health.json()["analysis"] == "voxel_hexahedral_linear_elastic"


def test_synthetic_defect_is_weaker_than_its_intact_copy_and_marks_the_band() -> None:
    study = load_dicom_path(DEFECT_CT)
    defective = analyze_ct_volume(study["volume"], study["spacing_zyx"], 70.0)
    intact_hu, fraction = _repair_low_density_band(study["volume"])
    intact = analyze_ct_volume(intact_hu, study["spacing_zyx"], 70.0)
    assert defective.failure_load_n < intact.failure_load_n
    assert defective.signed_percent_weaker > intact.signed_percent_weaker
    assert defective.hotspot_z_mm is not None
    expected = fraction * defective.bone_length_mm
    assert abs(defective.hotspot_z_mm - expected) <= max(12.0, 0.12 * defective.bone_length_mm)


def test_real_distal_ct_finishes_in_voxel_fe_and_reports_the_field() -> None:
    report = analyze_dicom_path(DISTAL_CT, body_mass_kg=70.0)
    assert report.method == "voxel_hexahedral_linear_elastic"
    assert report.solver.startswith("voxel_hexahedral_")
    assert "surrogate" not in report.solver
    assert report.tibia_voxels > 0
    assert np.isfinite(report.failure_load_n) and report.failure_load_n > 0
    note = report.field_note.lower()
    assert "distal" in note
    assert "not a full tibial shaft" in note
