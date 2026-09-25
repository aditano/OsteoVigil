"""Run the strength solver on a DICOM folder already stored on disk.

GitHub Pages has no Python server. The site loads this module inside Pyodide
and passes a folder of DICOM bytes that never leave the browser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import StrengthAnalysisError
from .strength_io import StudyDecoder, resolve_dicom_datasets
from .strength_pipeline import analyze_dicom_path, analyze_loaded_study

_DECODER: StudyDecoder | None = None


def _error_payload(exc: StrengthAnalysisError) -> dict[str, Any]:
    return {"error": str(exc), "modality": exc.modality}


def analyze_upload(study_dir: str, body_mass_kg: float, prefer_dicom_weight: bool) -> dict[str, Any]:
    try:
        report = analyze_dicom_path(
            Path(study_dir),
            body_mass_kg=float(body_mass_kg),
            prefer_dicom_weight=bool(prefer_dicom_weight),
        )
    except StrengthAnalysisError as exc:
        return _error_payload(exc)
    return report.public_dict()


def prepare_study(study_dir: str) -> dict[str, Any]:
    """Index a study without decoding pixels. The next calls decode one image at a time."""
    global _DECODER
    try:
        decoder = StudyDecoder(resolve_dicom_datasets(Path(study_dir)))
    except StrengthAnalysisError as exc:
        _DECODER = None
        return _error_payload(exc)
    _DECODER = decoder
    return {"count": decoder.total, "kind": decoder.kind}


def decode_next_dataset() -> dict[str, Any]:
    """Decode the next DICOM dataset already indexed by ``prepare_study``."""
    decoder = _DECODER
    if decoder is None:
        return {"error": "No study is open."}
    try:
        decoder.decode_next()
    except StrengthAnalysisError as exc:
        return _error_payload(exc)
    return {
        "done": decoder.done,
        "total": decoder.total,
        "finished": decoder.done >= decoder.total,
    }


def solve_prepared_study(body_mass_kg: float, prefer_dicom_weight: bool) -> dict[str, Any]:
    """Solve the study decoded by ``decode_next_dataset``."""
    global _DECODER
    decoder = _DECODER
    if decoder is None:
        return {"error": "No study is open."}
    try:
        report = analyze_loaded_study(
            decoder.study(),
            float(body_mass_kg),
            prefer_dicom_weight=bool(prefer_dicom_weight),
        )
    except StrengthAnalysisError as exc:
        return _error_payload(exc)
    finally:
        _DECODER = None
    return report.public_dict()
