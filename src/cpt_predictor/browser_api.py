"""Run the strength solver on a DICOM folder already stored on disk.

GitHub Pages has no Python server. The site loads this module inside Pyodide
and passes a folder of DICOM bytes that never leave the browser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import StrengthAnalysisError
from .strength_pipeline import analyze_dicom_path


def analyze_upload(study_dir: str, body_mass_kg: float, prefer_dicom_weight: bool) -> dict[str, Any]:
    try:
        report = analyze_dicom_path(
            Path(study_dir),
            body_mass_kg=float(body_mass_kg),
            prefer_dicom_weight=bool(prefer_dicom_weight),
        )
    except StrengthAnalysisError as exc:
        return {"error": str(exc), "modality": exc.modality}
    return report.public_dict()
