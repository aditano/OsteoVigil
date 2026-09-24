"""Classify a DICOM header as CT, MRI, or a plain radiograph."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ScanKind = Literal["ct", "mri", "radiograph"]

_CT_MODALITIES = {"CT"}
_MR_MODALITIES = {"MR", "MRI"}
_RADIOGRAPH_MODALITIES = {"DX", "CR", "DR", "RG", "XA", "RF"}

_SOP_KIND = {
    "1.2.840.10008.5.1.4.1.1.2": "ct",
    "1.2.840.10008.5.1.4.1.1.2.1": "ct",
    "1.2.840.10008.5.1.4.1.1.2.2": "ct",
    "1.2.840.10008.5.1.4.1.1.4": "mri",
    "1.2.840.10008.5.1.4.1.1.4.1": "mri",
    "1.2.840.10008.5.1.4.1.1.4.4": "mri",
    "1.2.840.10008.5.1.4.1.1.1": "radiograph",
    "1.2.840.10008.5.1.4.1.1.1.1": "radiograph",
    "1.2.840.10008.5.1.4.1.1.1.1.1": "radiograph",
    "1.2.840.10008.5.1.4.1.1.12.1": "radiograph",
}


@dataclass(frozen=True)
class ModalityAssessment:
    kind: ScanKind | None
    modality_tag: str
    sop_class_uid: str
    detail: str

    @property
    def supported(self) -> bool:
        return self.kind is not None


def classify_header(
    *,
    modality: str = "",
    sop_class_uid: str = "",
    number_of_frames: int = 1,
    photometric: str = "",
) -> ModalityAssessment:
    """Use the DICOM modality and SOP class. Pixel values are not required."""
    tag = str(modality or "").strip().upper()
    sop = str(sop_class_uid or "").strip()
    kind: ScanKind | None = None
    if tag in _CT_MODALITIES:
        kind = "ct"
    elif tag in _MR_MODALITIES:
        kind = "mri"
    elif tag in _RADIOGRAPH_MODALITIES:
        kind = "radiograph"
    elif sop in _SOP_KIND:
        kind = _SOP_KIND[sop]  # type: ignore[assignment]
    elif tag == "" and int(number_of_frames) > 1 and "MONOCHROME" in photometric.upper():
        kind = None

    if kind == "ct":
        detail = "Classified as CT from the DICOM modality or SOP class."
    elif kind == "mri":
        detail = "Classified as MRI from the DICOM modality or SOP class."
    elif kind == "radiograph":
        detail = "Classified as a radiograph from the DICOM modality or SOP class."
    else:
        detail = f"Unsupported DICOM modality {tag or 'missing'}."
    return ModalityAssessment(kind=kind, modality_tag=tag, sop_class_uid=sop, detail=detail)


def radiographic_view(*texts: str) -> tuple[str, bool]:
    """Return ('ap' or 'lateral', assumed)."""
    blob = " ".join(str(text or "") for text in texts).upper()
    if re.search(r"\b(LATERAL|LAT|LL|RL)\b", blob):
        return "lateral", False
    if re.search(r"\b(AP|PA)\b", blob):
        return "ap", False
    return "ap", True
