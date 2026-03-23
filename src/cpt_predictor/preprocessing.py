"""Preprocessing helpers for CT normalization and resampling."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Tuple, Union

import numpy as np

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover - optional dependency at import time
    sitk = None

from .io.dicom_loader import CTVolume
from .models import StudyData


def clip_hu(volume: CTVolume, lower_hu: float = -1000.0, upper_hu: float = 2500.0) -> CTVolume:
    clipped = np.clip(volume.array.astype(np.float32), lower_hu, upper_hu)
    return replace(volume, array=clipped)


def normalize_hu(volume: CTVolume, lower_hu: float = -1000.0, upper_hu: float = 2500.0) -> np.ndarray:
    clipped = np.clip(volume.array.astype(np.float32), lower_hu, upper_hu)
    return (clipped - lower_hu) / (upper_hu - lower_hu)


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


def resample_isotropic(volume: CTVolume, target_spacing_mm: Union[float, Tuple[float, float, float]] = 1.0) -> CTVolume:
    if isinstance(target_spacing_mm, (int, float)):
        target_spacing = (float(target_spacing_mm),) * 3
    else:
        target_spacing = tuple(float(v) for v in target_spacing_mm)

    if sitk is not None:
        image = sitk.GetImageFromArray(volume.array.astype(np.float32))
        image.SetSpacing((float(volume.spacing_mm[2]), float(volume.spacing_mm[1]), float(volume.spacing_mm[0])))
        image.SetOrigin((float(volume.origin_mm[2]), float(volume.origin_mm[1]), float(volume.origin_mm[0])))
        if len(volume.direction) == 9:
            image.SetDirection(tuple(float(v) for v in volume.direction))

        original_spacing = np.array(image.GetSpacing(), dtype=float)
        original_size = np.array(image.GetSize(), dtype=float)
        target_spacing_xyz = np.array((target_spacing[2], target_spacing[1], target_spacing[0]), dtype=float)
        target_size = np.maximum(np.round(original_size * (original_spacing / target_spacing_xyz)).astype(int), 1)

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(tuple(target_spacing_xyz))
        resampler.SetSize([int(v) for v in target_size.tolist()])
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetDefaultPixelValue(-1024.0)
        resampled = resampler.Execute(image)
        array = sitk.GetArrayFromImage(resampled).astype(np.float32)
    else:
        from scipy.ndimage import zoom

        zoom_factors = tuple(float(old / new) for old, new in zip(volume.spacing_mm, target_spacing))
        array = zoom(volume.array.astype(np.float32), zoom=zoom_factors, order=1)

    return CTVolume(
        array=array,
        spacing_mm=target_spacing,
        origin_mm=volume.origin_mm,
        direction=volume.direction,
        source=f"{volume.source}+resampled",
    )


def preprocess_study(study: StudyData, config: Any) -> StudyData:
    clip_range = _config_get(config, "preprocessing.clip_range_hu", [-1000, 2500])
    target_spacing = _config_get(config, "preprocessing.target_spacing_mm", (1.0, 1.0, 1.0))

    ct_volume = CTVolume(
        array=study.hu_volume.astype(np.float32),
        spacing_mm=study.spacing_zyx,
        origin_mm=(study.origin_xyz[2], study.origin_xyz[1], study.origin_xyz[0]),
        direction=study.direction,
        source=study.metadata.get("loader", "StudyData"),
    )
    clipped = clip_hu(ct_volume, lower_hu=float(clip_range[0]), upper_hu=float(clip_range[1]))
    resampled = resample_isotropic(clipped, target_spacing_mm=target_spacing)
    normalized = normalize_hu(resampled, lower_hu=float(clip_range[0]), upper_hu=float(clip_range[1]))

    metadata = dict(study.metadata)
    metadata["preprocessing"] = {
        "clip_range_hu": list(clip_range),
        "target_spacing_mm": list(target_spacing),
    }
    return StudyData(
        volume=resampled.array.astype(np.float32),
        hu_volume=resampled.array.astype(np.float32),
        spacing_zyx=resampled.spacing_zyx,
        origin_xyz=resampled.origin_xyz,
        direction=tuple(float(v) for v in resampled.direction),
        metadata=metadata,
        normalized_volume=normalized.astype(np.float32),
        source_dir=study.source_dir,
    )
