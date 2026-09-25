"""Read a DICOM file, folder, or zip into a CT stack, MRI stack, or radiographs."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pydicom

from .dicom_codecs import COMPRESSED_PIXEL_ERROR, register_dicom_codecs
from .errors import StrengthAnalysisError
from .modality import ModalityAssessment, classify_header, radiographic_view

register_dicom_codecs()

_UNCOMPRESSED_TRANSFER_SYNTAXES = {
    "1.2.840.10008.1.2",
    "1.2.840.10008.1.2.1",
    "1.2.840.10008.1.2.1.99",
    "1.2.840.10008.1.2.2",
}


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


def _transfer_syntax_uid(dataset: Any) -> str:
    meta = getattr(dataset, "file_meta", None)
    uid = getattr(meta, "TransferSyntaxUID", None) if meta is not None else None
    if uid in (None, ""):
        uid = getattr(dataset, "TransferSyntaxUID", "")
    return str(uid or "")


def _pixel_values(dataset: Any) -> np.ndarray:
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    try:
        pixels = dataset.pixel_array
    except Exception as exc:
        syntax = _transfer_syntax_uid(dataset)
        if syntax and syntax not in _UNCOMPRESSED_TRANSFER_SYNTAXES:
            raise StrengthAnalysisError(COMPRESSED_PIXEL_ERROR) from exc
        raise
    return pixels.astype(np.float32) * slope + intercept


def _pixel_spacing(dataset: Any) -> tuple[float, float]:
    """CT and MRI spacing. Radiographs use radiograph_pixel_spacing instead."""
    spacing = getattr(dataset, "PixelSpacing", None) or getattr(dataset, "ImagerPixelSpacing", None)
    if spacing is None:
        return (1.0, 1.0)
    return (float(spacing[0]), float(spacing[1]))


@dataclass(frozen=True)
class RadiographSpacing:
    row_mm: float
    col_mm: float
    source: str
    assumed: bool

    def as_tuple(self) -> tuple[float, float]:
        return (float(self.row_mm), float(self.col_mm))

    def public_dict(self) -> dict[str, Any]:
        return {
            "row_mm": float(self.row_mm),
            "col_mm": float(self.col_mm),
            "source": self.source,
            "assumed": bool(self.assumed),
        }


def _positive_spacing_pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        row = float(value[0])
        col = float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if row <= 0 or col <= 0 or not np.isfinite(row) or not np.isfinite(col):
        return None
    return (row, col)


def _magnification_factor(dataset: Any) -> float | None:
    raw = getattr(dataset, "EstimatedRadiographicMagnificationFactor", None)
    if raw in (None, ""):
        return None
    try:
        factor = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(factor) or factor < 1.0:
        return None
    return factor


def radiograph_pixel_spacing(dataset: Any) -> RadiographSpacing:
    """Patient-plane millimetres per pixel for a plain radiograph.

    PixelSpacing is already in the patient plane. ImagerPixelSpacing is the
    detector pitch and is divided by the magnification factor when that tag
    is present and at least 1. A missing tag is 1.0 mm and is labeled assumed.
    """
    pixel = _positive_spacing_pair(getattr(dataset, "PixelSpacing", None))
    if pixel is not None:
        return RadiographSpacing(pixel[0], pixel[1], "PixelSpacing", False)
    imager = _positive_spacing_pair(getattr(dataset, "ImagerPixelSpacing", None))
    if imager is not None:
        factor = _magnification_factor(dataset)
        if factor is not None:
            return RadiographSpacing(
                imager[0] / factor,
                imager[1] / factor,
                "ImagerPixelSpacing/EstimatedRadiographicMagnificationFactor",
                False,
            )
        return RadiographSpacing(imager[0], imager[1], "ImagerPixelSpacing", False)
    return RadiographSpacing(1.0, 1.0, "assumed", True)


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


def _slice_sort_key(dataset: Any) -> float:
    position = getattr(dataset, "ImagePositionPatient", None)
    if position is not None and len(position) >= 3:
        return float(position[2])
    return float(getattr(dataset, "InstanceNumber", 0) or 0)


def _ordered_datasets(datasets: list[Any]) -> list[Any]:
    return sorted(datasets, key=_slice_sort_key)


def _volume_from_pixels(
    pixels: list[np.ndarray],
    ordered: list[Any],
) -> tuple[np.ndarray, tuple[float, float, float]]:
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


def resolve_dicom_datasets(path: Path) -> list[Any]:
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
    return datasets


class StudyDecoder:
    """Decode DICOM pixels one dataset at a time, then build the study dict.

    ``load_dicom_path`` decodes every dataset before returning. The browser
    solver uses the same decoder so it can report each image as it finishes.
    """

    def __init__(self, datasets: list[Any]) -> None:
        self.header = _classify_datasets(datasets)
        if not self.header.supported or self.header.kind is None:
            raise StrengthAnalysisError(self.header.detail, modality="unknown")
        self.kind = self.header.kind
        self.weight = _patient_weight_kg(datasets)
        if self.kind == "radiograph":
            self._datasets = list(datasets)
        else:
            self._datasets = _ordered_datasets(datasets)
        self._index = 0
        self._views: dict[str, tuple[np.ndarray, tuple[float, float]]] = {}
        self._assumed: dict[str, bool] = {}
        self._spacing: dict[str, dict[str, Any]] = {}
        self._pixels: list[np.ndarray] = []

    @property
    def total(self) -> int:
        return len(self._datasets)

    @property
    def done(self) -> int:
        return self._index

    def decode_next(self) -> None:
        if self._index >= len(self._datasets):
            return
        dataset = self._datasets[self._index]
        pixels = _pixel_values(dataset)
        if self.kind == "radiograph":
            view, was_assumed = radiographic_view(
                str(getattr(dataset, "ViewPosition", "") or ""),
                str(getattr(dataset, "SeriesDescription", "") or ""),
                str(getattr(dataset, "ProtocolName", "") or ""),
            )
            spacing = radiograph_pixel_spacing(dataset)
            self._views[view] = (pixels, spacing.as_tuple())
            self._assumed[view] = was_assumed
            self._spacing[view] = spacing.public_dict()
        else:
            self._pixels.append(pixels)
        self._index += 1

    def study(self) -> dict[str, Any]:
        if self._index < len(self._datasets):
            raise StrengthAnalysisError("The DICOM series is only partly decoded.", modality=self.kind)
        if self.kind == "radiograph":
            return {
                "kind": "radiograph",
                "views": self._views,
                "view_assumed": self._assumed,
                "radiograph_spacing": self._spacing,
                "patient_weight_kg": self.weight,
                "header": self.header,
            }
        volume, spacing = _volume_from_pixels(self._pixels, self._datasets)
        if volume.ndim != 3 or volume.shape[0] < 4:
            raise StrengthAnalysisError(
                "CT and MRI analysis needs a stack of slices, not a single image.",
                modality=self.kind,
            )
        return {
            "kind": self.kind,
            "volume": volume,
            "spacing_zyx": spacing,
            "patient_weight_kg": self.weight,
            "header": self.header,
        }


def load_dicom_path(path: Path) -> dict[str, Any]:
    decoder = StudyDecoder(resolve_dicom_datasets(path))
    while decoder.done < decoder.total:
        decoder.decode_next()
    return decoder.study()
