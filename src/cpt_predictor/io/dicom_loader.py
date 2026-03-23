"""DICOM CT loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover - optional dependency at import time
    sitk = None

try:
    import pydicom
except Exception:  # pragma: no cover - optional dependency at import time
    pydicom = None

from ..models import StudyData


@dataclass
class CTVolume:
    array: np.ndarray
    spacing_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float]
    direction: tuple[float, ...]
    source: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.array.shape)

    @property
    def spacing_zyx(self) -> tuple[float, float, float]:
        return self.spacing_mm

    @property
    def origin_xyz(self) -> tuple[float, float, float]:
        return (self.origin_mm[2], self.origin_mm[1], self.origin_mm[0])


def _iter_dicom_files(dicom_dir: Path) -> list[Path]:
    files = [path for path in dicom_dir.rglob("*") if path.is_file()]
    dicom_files = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix in {".dcm", ".dicom"} or not suffix:
            dicom_files.append(path)
    return sorted(dicom_files)


def _apply_slope_intercept(image: Any, array: np.ndarray) -> np.ndarray:
    slope = float(image.GetMetaData("0028|1053")) if image.HasMetaDataKey("0028|1053") else 1.0
    intercept = float(image.GetMetaData("0028|1052")) if image.HasMetaDataKey("0028|1052") else 0.0
    return array * slope + intercept


def _load_with_simpleitk(dicom_dir: Path) -> CTVolume:
    if sitk is None:
        raise ImportError("SimpleITK is not installed")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {dicom_dir}")

    series_files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(series_files)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    array = _apply_slope_intercept(image, array)

    spacing_xyz = image.GetSpacing()
    origin_xyz = image.GetOrigin()
    return CTVolume(
        array=array,
        spacing_mm=(float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0])),
        origin_mm=(float(origin_xyz[2]), float(origin_xyz[1]), float(origin_xyz[0])),
        direction=tuple(float(v) for v in image.GetDirection()),
        source="SimpleITK",
    )


def _sort_slices(datasets: Iterable[Any]) -> list[Any]:
    def position(dataset: Any) -> float:
        if hasattr(dataset, "ImagePositionPatient"):
            return float(dataset.ImagePositionPatient[2])
        return float(getattr(dataset, "InstanceNumber", 0))

    return sorted(list(datasets), key=position)


def _load_with_pydicom(dicom_dir: Path) -> CTVolume:
    if pydicom is None:
        raise ImportError("pydicom is not installed")

    datasets = []
    for path in _iter_dicom_files(dicom_dir):
        try:
            datasets.append(pydicom.dcmread(str(path), force=True))
        except Exception:
            continue
    datasets = _sort_slices(datasets)
    if not datasets:
        raise FileNotFoundError(f"No readable DICOM files found in {dicom_dir}")

    first = datasets[0]
    pixel_arrays = []
    for dataset in datasets:
        pixel_array = dataset.pixel_array.astype(np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        pixel_arrays.append(pixel_array * slope + intercept)

    volume = np.stack(pixel_arrays, axis=0)
    spacing_y = float(getattr(first, "PixelSpacing", [1.0, 1.0])[0])
    spacing_x = float(getattr(first, "PixelSpacing", [1.0, 1.0])[1])
    if len(datasets) > 1 and hasattr(datasets[0], "ImagePositionPatient") and hasattr(datasets[1], "ImagePositionPatient"):
        spacing_z = abs(float(datasets[1].ImagePositionPatient[2]) - float(datasets[0].ImagePositionPatient[2]))
    else:
        spacing_z = float(getattr(first, "SliceThickness", 1.0))

    origin = getattr(first, "ImagePositionPatient", (0.0, 0.0, 0.0))
    direction = tuple(float(v) for v in getattr(first, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]))
    return CTVolume(
        array=volume,
        spacing_mm=(float(spacing_z), spacing_y, spacing_x),
        origin_mm=(float(origin[2]), float(origin[1]), float(origin[0])),
        direction=direction,
        source="pydicom",
    )


def load_ct_volume(dicom_dir: Union[str, Path], prefer_simpleitk: bool = True) -> CTVolume:
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.exists() or not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM directory does not exist: {dicom_dir}")

    loaders = (_load_with_simpleitk, _load_with_pydicom) if prefer_simpleitk else (_load_with_pydicom, _load_with_simpleitk)
    errors = []
    for loader in loaders:
        try:
            return loader(dicom_dir)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{loader.__name__}: {exc}")
    raise RuntimeError("Could not load CT volume. Tried:\n" + "\n".join(errors))


def _config_get(config: Any, dotted_key: str, default: Any = None) -> Any:
    current = config
    for part in dotted_key.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
    return default if current is None else current


def _to_study_data(ct_volume: CTVolume, dicom_dir: Optional[Path]) -> StudyData:
    return StudyData(
        volume=ct_volume.array.astype(np.float32),
        hu_volume=ct_volume.array.astype(np.float32),
        spacing_zyx=ct_volume.spacing_zyx,
        origin_xyz=ct_volume.origin_xyz,
        direction=tuple(float(value) for value in ct_volume.direction),
        metadata={
            "spacing_zyx": ct_volume.spacing_zyx,
            "origin_xyz": ct_volume.origin_xyz,
            "shape_zyx": ct_volume.shape,
            "loader": ct_volume.source,
        },
        source_dir=dicom_dir,
    )


def load_ct_study(dicom_dir: Optional[Union[str, Path]], config: Any, output_dir: Union[str, Path]) -> StudyData:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_dir = Path(dicom_dir) if dicom_dir else None

    allow_dummy = bool(_config_get(config, "input.allow_dummy_if_missing", False))
    dicom_present = bool(study_dir and study_dir.exists() and any(study_dir.rglob("*")))

    if not dicom_present:
        if not allow_dummy:
            raise FileNotFoundError(f"DICOM directory does not exist or is empty: {study_dir}")

        from .sample_data import generate_synthetic_ct_study

        return generate_synthetic_ct_study(config, output_dir / "synthetic_case", write_dicom=True)

    return _to_study_data(load_ct_volume(study_dir, prefer_simpleitk=True), study_dir)
