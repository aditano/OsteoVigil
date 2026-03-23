"""Synthetic sample-data generation for demos and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..models import StudyData
from .dicom_loader import CTVolume


def generate_synthetic_ct_volume(config: Dict[str, Any]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    sample_cfg = config.get("sample_data", {})
    shape = tuple(int(v) for v in sample_cfg.get("volume_shape_zyx", [160, 128, 128]))
    spacing = tuple(float(v) for v in sample_cfg.get("spacing_mm_zyx", [1.2, 0.8, 0.8]))
    seed = int(config.get("project", {}).get("random_seed", 42))
    rng = np.random.default_rng(seed)

    z, y, x = np.indices(shape)
    cz = shape[0] / 2.0
    cy = shape[1] / 2.0
    cx = shape[2] / 2.0

    volume = np.full(shape, -950.0, dtype=np.float32)
    soft_tissue = 45.0 + rng.normal(0.0, 18.0, size=shape)
    volume += soft_tissue.astype(np.float32)

    z_norm = (z - cz) / (shape[0] * 0.38)
    y_norm = (y - cy) / (shape[1] * 0.16)
    x_norm = (x - cx) / (shape[2] * 0.12)
    shaft = (np.square(y_norm) + np.square(x_norm)) <= 1.0

    cortical = shaft & ((np.square((y - cy) / (shape[1] * 0.12)) + np.square((x - cx) / (shape[2] * 0.09))) > 0.45)
    trabecular = shaft & ~cortical
    defect_band = np.abs(z - (cz + shape[0] * 0.08)) < max(3, shape[0] * 0.025)
    periosteal_bulge = np.square((y - (cy + shape[1] * 0.02)) / (shape[1] * 0.18)) + np.square((x - (cx - shape[2] * 0.03)) / (shape[2] * 0.16)) <= 1.0

    volume[cortical] = 1250.0 + rng.normal(0.0, 60.0, size=int(cortical.sum()))
    volume[trabecular] = 350.0 + rng.normal(0.0, 40.0, size=int(trabecular.sum()))
    volume[shaft & defect_band] = 120.0 + rng.normal(0.0, 20.0, size=int((shaft & defect_band).sum()))
    volume[shaft & periosteal_bulge] += 180.0
    volume = np.clip(volume, -1000.0, 2500.0).astype(np.float32)
    return volume, spacing


def _stl_triangle(normal, v1, v2, v3) -> str:
    return (
        f"facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"
        "  outer loop\n"
        f"    vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n"
        f"    vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n"
        f"    vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n"
        "  endloop\n"
        "endfacet\n"
    )


def write_proxy_brace_stl(output_path: Path, volume_shape: Tuple[int, int, int], spacing_zyx: Tuple[float, float, float]) -> Path:
    z_size = volume_shape[0] * spacing_zyx[0]
    y_size = volume_shape[1] * spacing_zyx[1]
    x_size = volume_shape[2] * spacing_zyx[2]

    hx = x_size * 0.18
    hy = y_size * 0.22
    hz = z_size * 0.42
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=float,
    )
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("solid brace_proxy\n")
        for i1, i2, i3 in faces:
            v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
            normal = np.cross(v2 - v1, v3 - v1)
            norm = np.linalg.norm(normal) or 1.0
            handle.write(_stl_triangle(normal / norm, v1, v2, v3))
        handle.write("endsolid brace_proxy\n")
    return output_path


def write_synthetic_dicom_series(
    volume: Any,
    spacing_zyx: Optional[Tuple[float, float, float]] = None,
    output_dir: Optional[Path] = None,
    patient_id: str = "DEMO21",
) -> Path:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    if isinstance(volume, CTVolume):
        ct_volume = volume
        output_dir = spacing_zyx if isinstance(spacing_zyx, Path) else output_dir
        if output_dir is None:
            raise ValueError("output_dir is required when writing a CTVolume")
        spacing_zyx = ct_volume.spacing_zyx
        volume = ct_volume.array

    if output_dir is None or spacing_zyx is None:
        raise ValueError("spacing_zyx and output_dir are required")

    output_dir.mkdir(parents=True, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()

    stored = np.clip(np.rint(volume + 1024.0), -32768, 32767).astype(np.int16)
    spacing_z, spacing_y, spacing_x = spacing_zyx

    for index, slice_array in enumerate(stored):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        file_path = output_dir / f"slice_{index:04d}.dcm"
        dataset = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.PatientName = "OsteoVigil^Demo"
        dataset.PatientID = patient_id
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.FrameOfReferenceUID = frame_uid
        dataset.Modality = "CT"
        dataset.SeriesDescription = "Synthetic CPT demo"
        dataset.InstanceNumber = index + 1
        dataset.ImagePositionPatient = [0.0, 0.0, float(index * spacing_z)]
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dataset.PixelSpacing = [float(spacing_y), float(spacing_x)]
        dataset.SliceThickness = float(spacing_z)
        dataset.SpacingBetweenSlices = float(spacing_z)
        dataset.RescaleSlope = 1.0
        dataset.RescaleIntercept = -1024.0
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.Rows, dataset.Columns = slice_array.shape
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.PixelData = slice_array.tobytes()
        dataset.save_as(str(file_path), write_like_original=False)

    return output_dir


def generate_synthetic_ct_study(config: Dict[str, Any], output_dir: Path, write_dicom: bool = True) -> StudyData:
    output_dir.mkdir(parents=True, exist_ok=True)
    volume, spacing_zyx = generate_synthetic_ct_volume(config)
    dicom_dir = output_dir / "dicom"
    brace_path: Optional[Path] = None
    if write_dicom:
        write_synthetic_dicom_series(volume, spacing_zyx, dicom_dir)
    if config.get("sample_data", {}).get("create_brace_proxy", True):
        brace_path = write_proxy_brace_stl(output_dir / "afo_proxy.stl", volume.shape, spacing_zyx)

    metadata = {
        "loader": "synthetic",
        "brace_proxy": str(brace_path) if brace_path else "",
        "dicom_dir": str(dicom_dir) if write_dicom else "",
    }
    return StudyData(
        volume=volume.copy(),
        hu_volume=volume,
        spacing_zyx=spacing_zyx,
        origin_xyz=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        metadata=metadata,
        source_dir=dicom_dir if write_dicom else output_dir,
    )

def create_dummy_ct(
    shape_zyx: Tuple[int, int, int] = (160, 128, 128),
    spacing_mm_zyx: Tuple[float, float, float] = (1.2, 0.8, 0.8),
    seed: int = 42,
) -> CTVolume:
    config = {
        "project": {"random_seed": seed},
        "sample_data": {
            "volume_shape_zyx": list(shape_zyx),
            "spacing_mm_zyx": list(spacing_mm_zyx),
        },
    }
    volume, spacing = generate_synthetic_ct_volume(config)
    return CTVolume(
        array=volume,
        spacing_mm=spacing,
        origin_mm=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        source="synthetic",
    )
