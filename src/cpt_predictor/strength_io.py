"""Read a DICOM file, folder, or zip into a CT stack, MRI stack, or radiographs."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pydicom

from .errors import StrengthAnalysisError
from .modality import ModalityAssessment, classify_header, radiographic_view


def _datasets_from_files(files: Iterable[Path]) -> list[Any]:
    datasets = []
    for path in files:
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in {".json", ".txt", ".md", ".stl", ".npy", ".zip"}:
            continue
        try:
            dataset = pydicom.dcmread(str(path), force=True)
        except Exception:
            continue
        if not hasattr(dataset, "PixelData"):
            continue
        datasets.append(dataset)
    return datasets


def _pixel_values(dataset: Any) -> np.ndarray:
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    return dataset.pixel_array.astype(np.float32) * slope + intercept


def _pixel_spacing(dataset: Any) -> tuple[float, float]:
    spacing = getattr(dataset, "PixelSpacing", None) or getattr(dataset, "ImagerPixelSpacing", None)
    if spacing is None:
        return (1.0, 1.0)
    return (float(spacing[0]), float(spacing[1]))


def _patient_weight_kg(datasets: list[Any]) -> float | None:
    for dataset in datasets:
        raw = getattr(dataset, "PatientWeight", None)
        if raw in (None, ""):
            continue
        try:
            weight = float(raw)
        except (TypeError, ValueError):
            continue
        if weight > 0:
            return weight
    return None


def _stack_slices(datasets: list[Any]) -> tuple[np.ndarray, tuple[float, float, float]]:
    def sort_key(dataset: Any) -> float:
        position = getattr(dataset, "ImagePositionPatient", None)
        if position is not None and len(position) >= 3:
            return float(position[2])
        return float(getattr(dataset, "InstanceNumber", 0) or 0)

    ordered = sorted(datasets, key=sort_key)
    pixels = [_pixel_values(dataset) for dataset in ordered]
    if len(pixels) == 1 and pixels[0].ndim == 3:
        volume = pixels[0]
    else:
        if any(image.ndim != 2 for image in pixels):
            raise StrengthAnalysisError("The DICOM series mixes 2D and 3D images.")
        volume = np.stack(pixels, axis=0)
    spacing_y, spacing_x = _pixel_spacing(ordered[0])
    spacing_z = float(getattr(ordered[0], "SliceThickness", 1.0) or 1.0)
    positions = []
    for dataset in ordered:
        position = getattr(dataset, "ImagePositionPatient", None)
        if position is not None and len(position) >= 3:
            positions.append(float(position[2]))
    if len(positions) >= 2:
        diffs = np.abs(np.diff(np.asarray(positions, dtype=float)))
        diffs = diffs[diffs > 1.0e-4]
        if diffs.size:
            spacing_z = float(np.median(diffs))
    return volume.astype(np.float32), (spacing_z, spacing_y, spacing_x)


def _classify_datasets(datasets: list[Any]) -> ModalityAssessment:
    first = datasets[0]
    return classify_header(
        modality=str(getattr(first, "Modality", "") or ""),
        sop_class_uid=str(getattr(first, "SOPClassUID", "") or ""),
        number_of_frames=int(getattr(first, "NumberOfFrames", 1) or 1),
        photometric=str(getattr(first, "PhotometricInterpretation", "") or ""),
    )


def load_dicom_path(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        extract_root = path.parent / f"{path.stem}_unzipped"
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(extract_root)
        path = extract_root
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    datasets = _datasets_from_files(files)
    if not datasets:
        raise StrengthAnalysisError(f"No readable DICOM files were found in {path}.")
    header = _classify_datasets(datasets)
    if not header.supported or header.kind is None:
        raise StrengthAnalysisError(header.detail, modality="unknown")
    weight = _patient_weight_kg(datasets)
    if header.kind == "radiograph":
        views: dict[str, tuple[np.ndarray, tuple[float, float]]] = {}
        assumed: dict[str, bool] = {}
        for dataset in datasets:
            view, was_assumed = radiographic_view(
                str(getattr(dataset, "ViewPosition", "") or ""),
                str(getattr(dataset, "SeriesDescription", "") or ""),
                str(getattr(dataset, "ProtocolName", "") or ""),
            )
            views[view] = (_pixel_values(dataset), _pixel_spacing(dataset))
            assumed[view] = was_assumed
        return {
            "kind": "radiograph",
            "views": views,
            "view_assumed": assumed,
            "patient_weight_kg": weight,
            "header": header,
        }
    volume, spacing = _stack_slices(datasets)
    if volume.ndim != 3 or volume.shape[0] < 4:
        raise StrengthAnalysisError(
            "CT and MRI analysis needs a stack of slices, not a single image.",
            modality=header.kind,
        )
    return {
        "kind": header.kind,
        "volume": volume,
        "spacing_zyx": spacing,
        "patient_weight_kg": weight,
        "header": header,
    }
