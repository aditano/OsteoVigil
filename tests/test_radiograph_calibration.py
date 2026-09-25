"""Radiograph cortical-index calibration, spacing tags, and the stronger-claim gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from cpt_predictor.radiograph import blocks_stronger_claim, measure_shaft, project_reference
from cpt_predictor.strength_io import load_dicom_path, radiograph_pixel_spacing
from cpt_predictor.strength_pipeline import analyze_radiograph_views


def _soft_tissue_radiograph(spacing: float = 0.5) -> np.ndarray:
    """Low-contrast leg silhouette. Cortex is only slightly brighter than soft tissue."""
    height = int(180 / spacing)
    width = int(90 / spacing)
    image = np.full((height, width), 40.0, dtype=np.float32)
    columns = np.arange(width)[None, :]
    tibia_center = width * 0.42
    soft = np.abs(columns - tibia_center) <= (31 / spacing)
    soft_rows = np.broadcast_to(soft, image.shape)
    image[soft_rows] = 150
    radius = np.abs(columns - tibia_center)
    cortex = (radius <= 13 / spacing) & (radius >= 8 / spacing)
    marrow = radius < 8 / spacing
    image[np.broadcast_to(cortex, image.shape)] = 168
    image[np.broadcast_to(marrow, image.shape)] = 158
    fibula = tibia_center + 24 / spacing
    fibula_radius = np.abs(columns - fibula)
    fibula_cortex = (fibula_radius <= 6.5 / spacing) & (fibula_radius >= 4.1 / spacing)
    fibula_marrow = fibula_radius < 4.1 / spacing
    image[np.broadcast_to(fibula_cortex, image.shape)] = 166
    image[np.broadcast_to(fibula_marrow, image.shape)] = 158
    image += np.random.default_rng(1).normal(0, 1.5, image.shape).astype(np.float32)
    image[6:18, 4:16] = 2400
    return image


def _flat_shaft(outer_mm: float, spacing: float = 0.5) -> np.ndarray:
    """Bright bar with no medullary valley. The old index scored this as a solid rod."""
    length_mm = 140.0
    rows = int(length_mm / spacing)
    cols = int((outer_mm + 16) / spacing)
    image = np.full((rows, cols), 40.0, dtype=np.float32)
    inset = int(6 / spacing)
    width_px = int(outer_mm / spacing)
    image[:, inset : inset + width_px] = 180
    image += np.random.default_rng(0).normal(0, 2.0, image.shape).astype(np.float32)
    return image


def test_soft_tissue_radiograph_does_not_claim_a_large_strength_advantage() -> None:
    spacing = (0.5, 0.5)
    image = _soft_tissue_radiograph()
    report = analyze_radiograph_views(
        {"ap": (image, spacing), "lateral": (_soft_tissue_radiograph(), spacing)},
        70,
        {"ap": False, "lateral": False},
    )
    measures = report.radiograph_measures
    assert measures is not None
    ratio = report.failure_load_n / report.reference_failure_load_n
    assert ratio < 1.5
    if report.comparison == "stronger":
        assert report.percent_vs_normal <= 15
        assert report.measurement_reliable
    else:
        assert report.comparison in {"weaker", "matched", "unreliable"}
    assert measures["cortical_area_mm2"] < 900
    outers = [float(view["outer_diameter_mm"]) for view in measures["views"].values()]
    assert min(outers) > 15


def test_unresolved_canal_is_not_scored_as_a_solid_rod() -> None:
    spacing = 0.5
    outer_mm = 40.0
    image = _flat_shaft(outer_mm, spacing)
    report = analyze_radiograph_views({"ap": (image, (spacing, spacing))}, 70, {"ap": False})
    measures = report.radiograph_measures
    assert measures is not None
    assert measures["canal_resolved"] is False
    view = measures["views"]["ap"]
    assert view["inner_diameter_mm"] > 0
    assert view["inner_estimated"] is True
    solid_disk = 0.25 * np.pi * outer_mm * outer_mm
    assert measures["cortical_area_mm2"] < 0.55 * solid_disk
    assert report.comparison != "stronger"
    assert "solid rod" in " ".join(report.assumptions).lower() or "not resolved" in report.measurement_note.lower()
    assert "tolerate unassisted walking" not in report.walking_text.lower()


def test_overwide_unresolved_shaft_cannot_claim_stronger_than_normal() -> None:
    image = _flat_shaft(120.0, 0.5)
    report = analyze_radiograph_views({"ap": (image, (0.5, 0.5))}, 70, {"ap": False})
    assert report.failure_load_n > report.reference_failure_load_n
    assert report.comparison == "unreliable"
    assert report.measurement_reliable is False
    assert "cannot claim stronger" in report.measurement_note.lower()
    assert "tolerate unassisted" not in report.walking_text.lower()
    payload = report.public_dict()
    assert payload["comparison"] == "unreliable"
    assert payload["radiograph_measures"]["views"]["ap"]["outer_diameter_mm"] > 35
    headline_fields = payload["measurement_note"].lower()
    assert "stronger" in headline_fields
    assert "344" not in payload["measurement_note"]


def test_blocks_stronger_claim_requires_canal_spacing_and_adult_diameter() -> None:
    assert blocks_stronger_claim(344.8, canal_resolved=False, outer_diameters_mm=[52.0], spacing_assumed=False)
    assert blocks_stronger_claim(18.0, canal_resolved=True, outer_diameters_mm=[26.0, 25.0], spacing_assumed=True)
    assert blocks_stronger_claim(18.0, canal_resolved=True, outer_diameters_mm=[18.0], spacing_assumed=False)
    assert blocks_stronger_claim(
        18.0,
        canal_resolved=True,
        outer_diameters_mm=[26.0],
        spacing_assumed=False,
        reference_resolved=False,
    )
    assert not blocks_stronger_claim(18.0, canal_resolved=True, outer_diameters_mm=[26.0, 25.0], spacing_assumed=False)
    assert not blocks_stronger_claim(10.0, canal_resolved=False, outer_diameters_mm=[52.0], spacing_assumed=True)


def test_clean_reference_projection_stays_reliable() -> None:
    ap, ap_spacing = project_reference(80, 0.5, "ap")
    lateral, lateral_spacing = project_reference(80, 0.5, "lateral")
    report = analyze_radiograph_views(
        {"ap": (ap, ap_spacing), "lateral": (lateral, lateral_spacing)},
        70,
        {"ap": False, "lateral": False},
    )
    assert report.comparison in {"matched", "weaker"}
    if report.comparison == "weaker":
        assert report.percent_vs_normal < 10
    assert report.measurement_reliable
    assert report.comparison != "unreliable"
    measures = report.radiograph_measures
    assert measures is not None
    assert measures["canal_resolved"] is True
    assert "elliptical" in report.assumptions[0].lower()
    payload = report.public_dict()
    assert payload["radiograph_measures"]["patient_failure_load_n"] == pytest.approx(report.failure_load_n)
    assert payload["radiograph_measures"]["reference_failure_load_n"] == pytest.approx(report.reference_failure_load_n)
    assert payload["radiograph_measures"]["cortical_area_mm2"] > 0
    for view in payload["radiograph_measures"]["views"].values():
        assert 20 <= view["outer_diameter_mm"] <= 40
        assert view["inner_diameter_mm"] > 0
        assert len(view["spacing_mm"]) == 2


def test_doubling_pixel_spacing_doubles_outer_diameter() -> None:
    image, spacing = project_reference(64, 1.0, "ap")
    base = measure_shaft(image, spacing)
    doubled = measure_shaft(image, (spacing[0] * 2, spacing[1] * 2))
    base_outer = float(np.median(base.outer_mm))
    wide_outer = float(np.median(doubled.outer_mm))
    assert wide_outer == pytest.approx(2 * base_outer, rel=0.15)


def test_radiograph_spacing_prefers_pixel_spacing_over_imager() -> None:
    class Header:
        PixelSpacing = [0.15, 0.14]
        ImagerPixelSpacing = [0.24, 0.22]
        EstimatedRadiographicMagnificationFactor = 1.2

    chosen = radiograph_pixel_spacing(Header())
    assert chosen.source == "PixelSpacing"
    assert chosen.assumed is False
    assert chosen.as_tuple() == pytest.approx((0.15, 0.14))


def test_imager_spacing_is_divided_by_magnification() -> None:
    class Header:
        PixelSpacing = None
        ImagerPixelSpacing = [0.24, 0.18]
        EstimatedRadiographicMagnificationFactor = 1.2

    chosen = radiograph_pixel_spacing(Header())
    assert chosen.source == "ImagerPixelSpacing/EstimatedRadiographicMagnificationFactor"
    assert chosen.as_tuple() == pytest.approx((0.20, 0.15))

    class ImagerOnly:
        ImagerPixelSpacing = [0.168, 0.168]

    imager = radiograph_pixel_spacing(ImagerOnly())
    assert imager.source == "ImagerPixelSpacing"
    assert imager.as_tuple() == pytest.approx((0.168, 0.168))
    assert imager.assumed is False


def test_missing_spacing_is_labeled_assumed() -> None:
    class Header:
        pass

    chosen = radiograph_pixel_spacing(Header())
    assert chosen.assumed is True
    assert chosen.source == "assumed"
    assert chosen.as_tuple() == (1.0, 1.0)


def _write_cr(path: Path, **attributes: object) -> None:
    pixels = np.zeros((8, 8), dtype=np.uint16)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CR"
    dataset.PatientName = "OsteoVigil^Spacing"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SeriesDescription = "AP tibia fibula"
    dataset.ViewPosition = "AP"
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixels.tobytes()
    for name, value in attributes.items():
        setattr(dataset, name, value)
    dataset.save_as(str(path), write_like_original=False)


def test_loaded_cr_records_pixel_spacing_and_ignores_imager_when_both_exist(tmp_path: Path) -> None:
    path = tmp_path / "cr.dcm"
    _write_cr(
        path,
        PixelSpacing=[0.143, 0.143],
        ImagerPixelSpacing=[0.2, 0.2],
        EstimatedRadiographicMagnificationFactor=1.15,
    )
    study = load_dicom_path(path)
    decoded, spacing = study["views"]["ap"]
    assert decoded.shape == (8, 8)
    assert spacing == pytest.approx((0.143, 0.143))
    meta = study["radiograph_spacing"]["ap"]
    assert meta["source"] == "PixelSpacing"
    assert meta["assumed"] is False
    image, spacing = project_reference(64, 1.0, "ap")
    report = analyze_radiograph_views(
        {"ap": (image, spacing)},
        70,
        study["view_assumed"],
        study["radiograph_spacing"],
    )
    assert report.radiograph_measures is not None
    assert report.radiograph_measures["spacing_assumed"] is False
    assert "PixelSpacing" in report.radiograph_measures["spacing_source"]
    assert any("PixelSpacing" in line for line in report.assumptions)


def test_loaded_cr_without_spacing_tags_is_labeled_on_the_report(tmp_path: Path) -> None:
    path = tmp_path / "unscaled.dcm"
    _write_cr(path)
    study = load_dicom_path(path)
    meta = study["radiograph_spacing"]["ap"]
    assert meta["assumed"] is True
    assert meta["source"] == "assumed"
    assert study["views"]["ap"][1] == pytest.approx((1.0, 1.0))
    image, spacing = project_reference(64, 1.0, "ap")
    report = analyze_radiograph_views(
        {"ap": (image, spacing)},
        70,
        {"ap": False},
        {"ap": {"row_mm": spacing[0], "col_mm": spacing[1], "source": "assumed", "assumed": True}},
    )
    assert report.radiograph_measures is not None
    assert report.radiograph_measures["spacing_assumed"] is True
    assert any("1.0 mm was assumed" in line for line in report.assumptions)
