"""Write small DICOM series for tests and local fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def write_dicom_series(
    volume: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    output_dir: Path,
    *,
    modality: str = "CT",
    sop_class_uid: Optional[str] = None,
    patient_id: str = "STRENGTH",
    patient_weight_kg: Optional[float] = None,
    view_position: str = "",
    series_description: str = "OsteoVigil strength",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    array = np.asarray(volume, dtype=np.float32)
    images = [array] if array.ndim == 2 else list(array)
    if sop_class_uid is None:
        if modality.upper() in {"MR", "MRI"}:
            sop_class_uid = "1.2.840.10008.5.1.4.1.1.4"
            modality = "MR"
        elif modality.upper() in {"DX", "CR", "DR"}:
            sop_class_uid = "1.2.840.10008.5.1.4.1.1.1.1"
        else:
            sop_class_uid = "1.2.840.10008.5.1.4.1.1.2"
            modality = "CT"
    study_uid = generate_uid()
    series_uid = generate_uid()
    spacing_z, spacing_y, spacing_x = spacing_zyx
    for index, slice_array in enumerate(images):
        stored = np.clip(np.rint(slice_array + 1024.0), -32768, 32767).astype(np.int16)
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = sop_class_uid
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()
        file_path = output_dir / f"slice_{index:04d}.dcm"
        dataset = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = sop_class_uid
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.PatientName = "OsteoVigil^Strength"
        dataset.PatientID = patient_id
        dataset.Modality = modality
        if patient_weight_kg is not None:
            dataset.PatientWeight = f"{float(patient_weight_kg):.1f}"
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.SeriesDescription = series_description
        dataset.ViewPosition = view_position
        dataset.InstanceNumber = index + 1
        dataset.ImagePositionPatient = [0.0, 0.0, float(index * spacing_z)]
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dataset.PixelSpacing = [float(spacing_y), float(spacing_x)]
        dataset.SliceThickness = float(spacing_z)
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
        dataset.save_as(str(file_path), write_like_original=False)
    return output_dir
