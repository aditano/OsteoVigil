"""Compressed DICOM pixels decode, and the browser solver installs those codecs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.pixel_data_handlers import pillow_handler
from pydicom.uid import generate_uid

from cpt_predictor.browser_api import analyze_upload
from cpt_predictor.dicom_codecs import COMPRESSED_PIXEL_ERROR
from cpt_predictor.errors import StrengthAnalysisError
from cpt_predictor.strength_io import load_dicom_path


REPO_ROOT = Path(__file__).resolve().parents[1]
BROWSER_SOLVER = REPO_ROOT / "web" / "src" / "browserSolver.ts"
VITE_CONFIG = REPO_ROOT / "web" / "vite.config.ts"
PAGES_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pages.yml"
INDEX_HTML = REPO_ROOT / "web" / "index.html"


def test_browser_solver_declares_dicom_codec_packages() -> None:
    source = BROWSER_SOLVER.read_text(encoding="utf-8")
    vite = VITE_CONFIG.read_text(encoding="utf-8")
    for package in (
        "pillow",
        "pydicom==2.4.4",
        "pylibjpeg==2.1.0",
        "pylibjpeg-libjpeg==2.4.0",
        "pylibjpeg-openjpeg==2.6.0",
    ):
        assert package in source
    assert "imagecodecs==" not in source
    assert "osteovigilDecodeDicomFrame" in source
    assert 'export const DICOM_CODEC_STATUS = "Loading DICOM image codecs…"' in source
    assert COMPRESSED_PIXEL_ERROR in source
    assert "cpt_predictor/dicom_codecs.py" in vite
    page = INDEX_HTML.read_text(encoding="utf-8")
    assert "Research biomechanical estimate" in page
    assert "not a diagnosis" in page
    assert "VITE_BASE: /OsteoVigil/" in PAGES_WORKFLOW.read_text(encoding="utf-8")


def _write_compressed(
    path: Path,
    pixels: np.ndarray,
    encoded: bytes,
    transfer_syntax: str,
    *,
    bits_allocated: int,
    bits_stored: int,
    pixel_representation: int,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = transfer_syntax
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CR"
    dataset.PatientName = "OsteoVigil^Codec"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SeriesDescription = "AP tibia fibula"
    dataset.ViewPosition = "AP"
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.BitsAllocated = bits_allocated
    dataset.BitsStored = bits_stored
    dataset.HighBit = bits_stored - 1
    dataset.PixelRepresentation = pixel_representation
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = 0.0
    dataset.PixelData = encapsulate([encoded])
    dataset["PixelData"].is_undefined_length = True
    dataset.save_as(str(path), write_like_original=False)


def test_compressed_decode_failure_is_a_short_message(tmp_path: Path) -> None:
    pixels = np.zeros((8, 8), dtype=np.uint16)
    path = tmp_path / "broken.dcm"
    _write_compressed(
        path,
        pixels,
        b"\xff\x4f\xff\x51not-a-jpeg2000-codestream",
        "1.2.840.10008.1.2.4.90",
        bits_allocated=16,
        bits_stored=12,
        pixel_representation=0,
    )
    with pytest.raises(StrengthAnalysisError, match="compressed") as caught:
        load_dicom_path(path)
    message = str(caught.value)
    assert message == COMPRESSED_PIXEL_ERROR
    assert "Traceback" not in message
    assert "RuntimeError" not in message
    payload = analyze_upload(str(path), 70.0, False)
    assert payload["error"] == COMPRESSED_PIXEL_ERROR


def test_compressed_cr_pixels_round_trip(tmp_path: Path) -> None:
    imagecodecs = pytest.importorskip("imagecodecs")
    pixels = (np.arange(40 * 32, dtype=np.uint16).reshape(40, 32) % 3500)
    lossless = imagecodecs.jpeg_encode(pixels, lossless=True, bitspersample=12)
    jpeg2000 = imagecodecs.jpeg2k_encode(pixels, codecformat="J2K", reversible=True)
    baseline = np.zeros((32, 48), dtype=np.uint8)
    baseline[8:24, 12:36] = 180
    baseline_bytes = imagecodecs.jpeg8_encode(baseline, level=95)
    cases = (
        ("ljpeg", pixels, lossless, "1.2.840.10008.1.2.4.70", 16, 12, 0, True),
        ("jpeg2000", pixels, jpeg2000, "1.2.840.10008.1.2.4.90", 16, 12, 0, True),
        ("jpeg", baseline, baseline_bytes, "1.2.840.10008.1.2.4.50", 8, 8, 0, False),
    )
    original_jpeg = pillow_handler.HAVE_JPEG
    original_jpeg2000 = pillow_handler.HAVE_JPEG2K
    pillow_handler.HAVE_JPEG = False
    pillow_handler.HAVE_JPEG2K = False
    try:
        for name, original, encoded, syntax, bits_allocated, bits_stored, signed, exact in cases:
            path = tmp_path / f"{name}.dcm"
            _write_compressed(
                path,
                original,
                encoded,
                syntax,
                bits_allocated=bits_allocated,
                bits_stored=bits_stored,
                pixel_representation=signed,
            )
            loaded = load_dicom_path(path)
            assert loaded["kind"] == "radiograph"
            decoded, _spacing = loaded["views"]["ap"]
            if exact:
                np.testing.assert_array_equal(decoded, original.astype(np.float32))
            else:
                assert decoded.shape == original.shape
                assert float(np.mean(np.abs(decoded - original.astype(np.float32)))) < 8.0
    finally:
        pillow_handler.HAVE_JPEG = original_jpeg
        pillow_handler.HAVE_JPEG2K = original_jpeg2000
