"""Decode compressed DICOM pixels in environments where GDCM is unavailable.

Pyodide's Pillow build includes baseline JPEG and not OpenJPEG. The pylibjpeg
plugins have no wasm wheels, and the imagecodecs wasm wheel aborts Pyodide when
called. Native runs use imagecodecs. The browser solver decodes JPEG lossless
and JPEG 2000 in JavaScript and passes the samples back through
``osteovigilDecodeDicomFrame`` so ``dataset.pixel_array`` still works.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pydicom
from pydicom.encaps import generate_pixel_data_frame
from pydicom.pixel_data_handlers.util import get_expected_length, pixel_dtype
from pydicom.uid import UID

try:
    import imagecodecs as _imagecodecs
except ImportError:
    _imagecodecs = None

try:
    import js as _js
except ImportError:
    _js = None

try:
    from pyodide.ffi import to_js as _to_js
except ImportError:
    _to_js = None

imagecodecs: Any = _imagecodecs

COMPRESSED_PIXEL_ERROR = "This DICOM is compressed and the image codecs could not decode it."

_JPEG_SYNTAXES = {
    "1.2.840.10008.1.2.4.50",  # JPEG Baseline
    "1.2.840.10008.1.2.4.51",  # JPEG Extended
}
_LJPEG_SYNTAXES = {
    "1.2.840.10008.1.2.4.57",  # JPEG Lossless, Process 14
    "1.2.840.10008.1.2.4.70",  # JPEG Lossless, Process 14 SV1
}
_JPEG2000_SYNTAXES = {
    "1.2.840.10008.1.2.4.90",  # JPEG 2000 Lossless
    "1.2.840.10008.1.2.4.91",  # JPEG 2000
}
_SUPPORTED_SYNTAXES = _JPEG_SYNTAXES | _LJPEG_SYNTAXES | _JPEG2000_SYNTAXES

HANDLER_NAME = "imagecodecs"
DEPENDENCIES = {
    "imagecodecs": ("https://pypi.org/project/imagecodecs/", "imagecodecs"),
    "numpy": ("https://numpy.org/", "NumPy"),
}

_registered = False


def _syntax_kind(transfer_syntax: str) -> str | None:
    if transfer_syntax in _JPEG_SYNTAXES:
        return "jpeg"
    if transfer_syntax in _LJPEG_SYNTAXES:
        return "ljpeg"
    if transfer_syntax in _JPEG2000_SYNTAXES:
        return "jpeg2k"
    return None


def _imagecodecs_usable() -> bool:
    if imagecodecs is None or sys.platform == "emscripten":
        return False
    return any(
        bool(getattr(getattr(imagecodecs, name, None), "available", False))
        for name in ("JPEG8", "LJPEG", "JPEG2K")
    )


def _javascript_decoder() -> Any:
    if _js is None:
        return None
    decoder = getattr(_js, "osteovigilDecodeDicomFrame", None)
    if decoder is None:
        return None
    return decoder


def is_available() -> bool:
    """Return True when a native or browser codec can decode compressed pixels."""
    return _imagecodecs_usable() or _javascript_decoder() is not None


def supports_transfer_syntax(transfer_syntax: UID) -> bool:
    """Return True for the compressed syntaxes this process can decode."""
    kind = _syntax_kind(str(transfer_syntax))
    if kind is None:
        return False
    if _imagecodecs_usable():
        return True
    if _javascript_decoder() is None:
        return False
    return kind in {"ljpeg", "jpeg2k"}


def needs_to_convert_to_RGB(_dataset: Any) -> bool:
    """Compressed grayscale radiographs stay in their stored color space."""
    return False


def should_change_PhotometricInterpretation_to_RGB(_dataset: Any) -> bool:
    """Leave Photometric Interpretation unchanged."""
    return False


def _decode_frame_imagecodecs(frame: bytes, kind: str) -> np.ndarray:
    if imagecodecs is None:
        raise ImportError("imagecodecs is not installed")
    if kind == "jpeg":
        decoded = imagecodecs.jpeg_decode(frame)
    elif kind == "ljpeg":
        decoded = imagecodecs.ljpeg_decode(frame)
    elif kind == "jpeg2k":
        decoded = imagecodecs.jpeg2k_decode(frame)
    else:
        raise NotImplementedError(f"No DICOM image codec for {kind}")
    return np.asarray(decoded)


def _decode_frame_javascript(decoder: Any, frame: bytes, ds: Any) -> np.ndarray:
    if _to_js is None:
        raise ImportError("The browser DICOM codec bridge is not available")
    raw = decoder(
        _to_js(frame),
        int(ds.Rows),
        int(ds.Columns),
        int(ds.BitsAllocated),
        int(getattr(ds, "SamplesPerPixel", 1) or 1),
        bool(int(getattr(ds, "PixelRepresentation", 0) or 0)),
        str(ds.file_meta.TransferSyntaxUID),
    )
    payload = raw.to_py()
    return np.frombuffer(bytes(payload), dtype=pixel_dtype(ds))


def _decode_frame(frame: bytes, kind: str, ds: Any) -> np.ndarray:
    javascript = _javascript_decoder()
    if javascript is not None and not _imagecodecs_usable():
        return _decode_frame_javascript(javascript, frame, ds)
    if _imagecodecs_usable():
        return _decode_frame_imagecodecs(frame, kind)
    raise ImportError("No DICOM image codec is available for this transfer syntax")


def _as_pixel_dtype(decoded: np.ndarray, ds: Any, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.asarray(decoded)
    samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    if array.ndim == 3 and samples == 1 and array.shape[-1] == 1:
        array = array[..., 0]
    flat = np.ascontiguousarray(array).reshape(-1)
    if flat.dtype != dtype:
        flat = flat.astype(dtype, copy=False)
    return flat


def get_pixeldata(ds: Any) -> np.ndarray:
    """Return encapsulated Pixel Data as a 1D array for ``dataset.pixel_array``."""
    transfer_syntax = str(ds.file_meta.TransferSyntaxUID)
    kind = _syntax_kind(transfer_syntax)
    if kind is None:
        raise NotImplementedError(f"Unsupported transfer syntax {transfer_syntax}")
    number_of_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    expected = int(get_expected_length(ds, "pixels"))
    frame_length = expected // number_of_frames
    dtype = pixel_dtype(ds)
    decoded_frames = np.empty(expected, dtype=dtype)
    frames = generate_pixel_data_frame(ds.PixelData, number_of_frames)
    for index, frame in enumerate(frames):
        flat = _as_pixel_dtype(_decode_frame(frame, kind, ds), ds, dtype)
        if flat.size != frame_length:
            raise RuntimeError(
                f"Decoded frame has {flat.size} samples and the DICOM header expects {frame_length}."
            )
        start = index * frame_length
        decoded_frames[start : start + frame_length] = flat
    return decoded_frames


def register_dicom_codecs() -> None:
    """Prefer this handler so ``pixel_array`` works without GDCM."""
    global _registered
    if _registered:
        return
    handlers = pydicom.config.pixel_data_handlers
    module = sys.modules[__name__]
    if module not in handlers:
        handlers.insert(0, module)
    _registered = True
