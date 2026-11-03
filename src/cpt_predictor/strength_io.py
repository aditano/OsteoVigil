"""Read a DICOM file, folder, or zip into a CT stack, MRI stack, or radiographs."""

from __future__ import annotations

import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pydicom
from scipy import ndimage as ndi

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

# Slice gaps beyond this relative spread are treated as uneven and resampled.
_UNEVEN_SPACING_RATIO = 1.05


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
    """CT and MRI in-plane spacing. Missing tags are an error, not 1 mm."""
    spacing = _positive_spacing_pair(getattr(dataset, "PixelSpacing", None))
    if spacing is not None:
        return spacing
    spacing = _positive_spacing_pair(getattr(dataset, "ImagerPixelSpacing", None))
    if spacing is not None:
        return spacing
    raise StrengthAnalysisError(
        "PixelSpacing is missing from the DICOM header. "
        "Refusing to assume 1.0 mm for CT/MRI strength analysis."
    )


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


def _series_uid(dataset: Any) -> str:
    uid = getattr(dataset, "SeriesInstanceUID", None)
    if uid in (None, ""):
        return ""
    return str(uid)


def _select_volume_series(datasets: list[Any]) -> list[Any]:
    """Keep one SeriesInstanceUID for CT/MRI stacks; mix no scout/localizer series."""
    if len(datasets) <= 1:
        return datasets
    grouped: dict[str, list[Any]] = defaultdict(list)
    for dataset in datasets:
        grouped[_series_uid(dataset)].append(dataset)
    if len(grouped) == 1:
        return datasets

    def score(series: list[Any]) -> tuple[int, int]:
        # Prefer the longest multi-slice series; blank UIDs sort last.
        uid = _series_uid(series[0])
        return (len(series), 1 if uid else 0)

    selected = max(grouped.values(), key=score)
    if len(selected) < 2:
        raise StrengthAnalysisError(
            "No multi-slice DICOM series was found after splitting by SeriesInstanceUID."
        )
    return selected


def _resample_uneven_volume(
    volume: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Rebuild a uniform Z grid from irregular ImagePositionPatient samples."""
    order = np.argsort(positions)
    positions = np.asarray(positions, dtype=float)[order]
    volume = np.asarray(volume, dtype=np.float32)[order]
    diffs = np.diff(positions)
    positive = diffs[np.abs(diffs) > 1.0e-4]
    if positive.size == 0:
        raise StrengthAnalysisError("Slice ImagePositionPatient values do not define a Z spacing.")
    target = float(np.median(np.abs(positive)))
    if target <= 1.0e-6:
        raise StrengthAnalysisError("Computed slice spacing is not positive.")
    span = float(abs(positions[-1] - positions[0]))
    count = max(2, int(round(span / target)) + 1)
    target_z = np.linspace(positions[0], positions[-1], count)
    source_index = np.interp(target_z, positions, np.arange(positions.size, dtype=float))
    coords = np.empty((3, count, volume.shape[1], volume.shape[2]), dtype=np.float64)
    coords[0] = source_index[:, None, None]
    coords[1] = np.arange(volume.shape[1], dtype=np.float64)[None, :, None]
    coords[2] = np.arange(volume.shape[2], dtype=np.float64)[None, None, :]
    resampled = ndi.map_coordinates(volume, coords, order=1, mode="nearest")
    spacing_z = float(abs(target_z[1] - target_z[0])) if count > 1 else target
    return resampled.astype(np.float32), spacing_z


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
    positions = []
    for dataset in ordered:
        position = getattr(dataset, "ImagePositionPatient", None)
        if position is not None and len(position) >= 3:
            positions.append(float(position[2]))
    spacing_z: float | None = None
    if len(positions) >= 2:
        position_array = np.asarray(positions, dtype=float)
        order = np.argsort(position_array)
        ordered_positions = position_array[order]
        diffs = np.diff(ordered_positions)
        positive = np.abs(diffs)
        positive = positive[positive > 1.0e-4]
        if positive.size == 0:
            raise StrengthAnalysisError(
                "Slice ImagePositionPatient values are missing or identical; "
                "refusing to assume 1.0 mm slice spacing."
            )
        median = float(np.median(positive))
        uneven = bool(
            float(np.max(positive) / median) > _UNEVEN_SPACING_RATIO
            or float(np.min(positive) / median) < (1.0 / _UNEVEN_SPACING_RATIO)
        )
        if uneven:
            if volume.ndim != 3 or volume.shape[0] != len(positions):
                raise StrengthAnalysisError("Uneven slice spacing could not be resampled.")
            volume = volume[order]
            volume, spacing_z = _resample_uneven_volume(volume, ordered_positions)
        else:
            volume = volume[order]
            spacing_z = median
    if spacing_z is None:
        raw_thickness = getattr(ordered[0], "SliceThickness", None)
        try:
            thickness = float(raw_thickness) if raw_thickness not in (None, "") else float("nan")
        except (TypeError, ValueError):
            thickness = float("nan")
        if not np.isfinite(thickness) or thickness <= 0:
            raise StrengthAnalysisError(
                "Slice spacing is missing from ImagePositionPatient and SliceThickness. "
                "Refusing to assume 1.0 mm for CT/MRI strength analysis."
            )
        spacing_z = thickness
    return volume.astype(np.float32), (float(spacing_z), spacing_y, spacing_x)


def _classify_datasets(datasets: list[Any]) -> ModalityAssessment:
    first = datasets[0]
    return classify_header(
        modality=str(getattr(first, "Modality", "") or ""),
        sop_class_uid=str(getattr(first, "SOPClassUID", "") or ""),
        number_of_frames=int(getattr(first, "NumberOfFrames", 1) or 1),
        photometric=str(getattr(first, "PhotometricInterpretation", "") or ""),
    )


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for info in archive.infolist():
        name = info.filename
        if not name or name.endswith("/"):
            continue
        target = (destination / name).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise StrengthAnalysisError(f"Refusing to extract unsafe zip member: {name}") from exc
        archive.extract(info, destination)


def resolve_dicom_datasets(path: Path) -> list[Any]:
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        extract_root = path.parent / f"{path.stem}_unzipped"
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            _safe_extract_zip(archive, extract_root)
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
        if self.kind == "radiograph":
            selected = list(datasets)
        else:
            selected = _select_volume_series(datasets)
            self.header = _classify_datasets(selected)
            if not self.header.supported or self.header.kind is None:
                raise StrengthAnalysisError(self.header.detail, modality="unknown")
            self.kind = self.header.kind
            selected = _ordered_datasets(selected)
        self.weight = _patient_weight_kg(selected)
        self._datasets = selected
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
