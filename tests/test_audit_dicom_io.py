"""Regression tests for DICOM series selection, spacing, and geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from cpt_predictor.errors import StrengthAnalysisError
from cpt_predictor.strength_io import load_dicom_path


def _write_slice(
    path: Path,
    *,
    pixels: np.ndarray,
    series_uid: str,
    instance: int,
    z_mm: float,
    pixel_spacing: tuple[float, float] | None = (1.0, 1.0),
    slice_thickness: float | None = 1.0,
    study_uid: str | None = None,
) -> None:
    stored = np.clip(np.rint(pixels.astype(np.float32) + 1024.0), -32768, 32767).astype(np.int16)
    sop_class = "1.2.840.10008.5.1.4.1.1.2"
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "OsteoVigil^Audit"
    dataset.PatientID = "AUDIT"
    dataset.Modality = "CT"
    dataset.StudyInstanceUID = study_uid or generate_uid()
    dataset.SeriesInstanceUID = series_uid
    dataset.InstanceNumber = instance
    dataset.ImagePositionPatient = [0.0, 0.0, float(z_mm)]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    if pixel_spacing is not None:
        dataset.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    if slice_thickness is not None:
        dataset.SliceThickness = float(slice_thickness)
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = -1024.0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows, dataset.Columns = stored.shape
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.PixelData = stored.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(str(path), write_like_original=False)


def test_load_keeps_only_the_largest_series_instance_uid(tmp_path: Path) -> None:
    primary = generate_uid()
    scout = generate_uid()
    study = generate_uid()
    bone = np.full((16, 16), 900.0, dtype=np.float32)
    air = np.full((16, 16), -1000.0, dtype=np.float32)
    for index in range(8):
        _write_slice(
            tmp_path / f"primary_{index:02d}.dcm",
            pixels=bone,
            series_uid=primary,
            instance=index + 1,
            z_mm=float(index * 2.0),
            study_uid=study,
        )
    for index in range(2):
        _write_slice(
            tmp_path / f"scout_{index:02d}.dcm",
            pixels=air,
            series_uid=scout,
            instance=100 + index,
            z_mm=float(index * 50.0),
            study_uid=study,
        )

    study_data = load_dicom_path(tmp_path)
    assert study_data["kind"] == "ct"
    assert study_data["volume"].shape[0] == 8
    assert study_data["spacing_zyx"][0] == pytest.approx(2.0)


def test_missing_pixel_spacing_raises_for_ct(tmp_path: Path) -> None:
    series = generate_uid()
    for index in range(4):
        _write_slice(
            tmp_path / f"slice_{index:02d}.dcm",
            pixels=np.zeros((8, 8), dtype=np.float32),
            series_uid=series,
            instance=index + 1,
            z_mm=float(index),
            pixel_spacing=None,
        )

    with pytest.raises(StrengthAnalysisError, match="PixelSpacing"):
        load_dicom_path(tmp_path)


def test_uneven_slice_positions_preserve_physical_length(tmp_path: Path) -> None:
    """Uneven Z positions must not collapse onto median spacing.

    A stack with mostly 1 mm steps and one large gap used to report ~10 mm of
    length (median spacing × slice count) instead of the true ~100 mm extent,
    which warps the strength comparison by tens of percent.
    """
    series = generate_uid()
    # Positions: 0..8 mm at 1 mm, then a jump to 100 mm.
    positions = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]
    for index, z_mm in enumerate(positions):
        mark = np.full((12, 12), -1000.0, dtype=np.float32)
        mark[4:8, 4:8] = 1200.0 + float(index)
        _write_slice(
            tmp_path / f"slice_{index:02d}.dcm",
            pixels=mark,
            series_uid=series,
            instance=index + 1,
            z_mm=z_mm,
            pixel_spacing=(1.0, 1.0),
            slice_thickness=1.0,
        )

    study_data = load_dicom_path(tmp_path)
    spacing_z = float(study_data["spacing_zyx"][0])
    length_mm = float(study_data["volume"].shape[0] - 1) * spacing_z
    assert length_mm == pytest.approx(100.0, abs=1.0)
    # Uniform voxel grid after correction (no silent median collapse).
    assert study_data["volume"].shape[0] >= 50
