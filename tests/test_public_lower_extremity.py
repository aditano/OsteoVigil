"""Public lower-extremity checks for the radiograph and CT strength paths.

STS_028 is a TCIA Soft-tissue-Sarcoma calf CT (CC BY 3.0). The DICOM cache is
gitignored. Without it, the public_data tests skip. See data/PUBLIC_SOURCES.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
from scipy import ndimage as ndi

from cpt_predictor.errors import StrengthAnalysisError
from cpt_predictor.io.dicom_export import write_dicom_series
from cpt_predictor.radiograph import project_reference
from cpt_predictor.reference_leg import synthetic_tibfib_volume
from cpt_predictor.strength_pipeline import (
    analyze_ct_volume,
    analyze_dicom_path,
    analyze_radiograph_views,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STS028 = REPO_ROOT / "data" / "external" / "sts028_legs"


def _flat_shaft(outer_mm: float, spacing: float = 0.5) -> np.ndarray:
    rows = int(140.0 / spacing)
    cols = int((outer_mm + 16) / spacing)
    image = np.full((rows, cols), 40.0, dtype=np.float32)
    inset = int(6 / spacing)
    width_px = int(outer_mm / spacing)
    image[:, inset : inset + width_px] = 180
    image += np.random.default_rng(0).normal(0, 2.0, image.shape).astype(np.float32)
    return image


def _load_sts028() -> tuple[np.ndarray, tuple[float, float, float]]:
    files = sorted(STS028.glob("*.dcm"))
    if len(files) < 50:
        pytest.skip(
            "STS_028 is not cached. Run: python3 scripts/download_public_abnormal_cts.py --case sts028"
        )
    datasets = [pydicom.dcmread(str(path)) for path in files]
    datasets.sort(key=lambda dataset: float(dataset.ImagePositionPatient[2]))
    slope = float(getattr(datasets[0], "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(datasets[0], "RescaleIntercept", 0.0) or 0.0)
    volume = np.stack(
        [dataset.pixel_array.astype(np.float32) * slope + intercept for dataset in datasets],
        axis=0,
    )
    pixel = float(datasets[0].PixelSpacing[0])
    positions = np.array([float(dataset.ImagePositionPatient[2]) for dataset in datasets])
    slice_spacing = float(np.median(np.abs(np.diff(positions))))
    return volume, (slice_spacing, pixel, pixel)


def _cortical_notch(volume: np.ndarray) -> np.ndarray:
    """Remove one side of the dense cortex on the middle third of a public shaft."""
    notch = np.array(volume, copy=True)
    z0 = notch.shape[0] // 3
    z1 = (2 * notch.shape[0]) // 3
    columns = np.flatnonzero((notch[z0:z1] >= 1000).any(axis=(0, 1)))
    if columns.size == 0:
        return notch
    cut = int(np.median(columns))
    region = notch[z0:z1, :, cut:]
    notch[z0:z1, :, cut:] = np.where(region >= 800, 40, region)
    return notch


def _cortical_widths(volume: np.ndarray, pixel_mm: float) -> tuple[float, float, float]:
    """Median AP width, lateral width, and equivalent canal diameter of the largest cortex."""
    ap_widths: list[float] = []
    lateral_widths: list[float] = []
    inners: list[float] = []
    z0 = min(8, volume.shape[0] // 5)
    z1 = max(z0 + 1, volume.shape[0] - z0)
    for index in range(z0, z1):
        bone = volume[index] >= 800
        labeled, count = ndi.label(bone)
        if count == 0:
            continue
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        label = int(np.argmax(sizes))
        if int(sizes[label]) < 8:
            continue
        mask = labeled == label
        rows, cols = np.nonzero(mask)
        ap_widths.append(float(cols.max() - cols.min() + 1) * pixel_mm)
        lateral_widths.append(float(rows.max() - rows.min() + 1) * pixel_mm)
        marrow = ndi.binary_fill_holes(mask) & ~mask
        if int(marrow.sum()) >= 4:
            inners.append(2.0 * float(np.sqrt(float(marrow.sum()) * pixel_mm * pixel_mm / np.pi)))
    assert ap_widths and lateral_widths and inners
    return float(np.median(ap_widths)), float(np.median(lateral_widths)), float(np.median(inners))


@pytest.fixture(scope="module")
def sts028_shafts() -> dict[str, object]:
    volume, spacing = _load_sts028()
    # Image-left has the smaller soft-tissue field. Image-right carries more
    # soft tissue beside the tibia on this bilateral calf series.
    left = volume[100:136, :, 90:230]
    right = volume[100:136, :, 260:430]
    mass = 58.0
    intact = analyze_ct_volume(left, spacing, mass)
    tumor = analyze_ct_volume(right, spacing, mass)
    notch = analyze_ct_volume(_cortical_notch(left), spacing, mass)
    return {
        "volume": volume,
        "spacing": spacing,
        "left": left,
        "right": right,
        "intact": intact,
        "tumor": tumor,
        "notch": notch,
    }


@pytest.mark.public_data
def test_public_normal_shaft_is_near_matched(sts028_shafts: dict[str, object]) -> None:
    report = sts028_shafts["intact"]
    assert report.comparison in {"weaker", "matched"}
    assert report.measurement_reliable
    assert report.percent_vs_normal <= 25
    assert 20_000 <= report.failure_load_n <= 80_000
    assert report.bone_length_mm > 80
    assert report.tibia_voxels > 0


@pytest.mark.public_data
def test_public_sarcoma_side_is_not_hundreds_stronger(sts028_shafts: dict[str, object]) -> None:
    left = sts028_shafts["left"]
    right = sts028_shafts["right"]
    left_soft = int(((left > -30) & (left < 150)).sum())
    right_soft = int(((right > -30) & (right < 150)).sum())
    assert right_soft > left_soft
    report = sts028_shafts["tumor"]
    assert report.comparison in {"weaker", "matched", "unreliable"}
    assert report.failure_load_n < 3.0 * report.reference_failure_load_n
    if report.comparison == "weaker":
        assert report.percent_vs_normal < 80
    assert "hundreds" not in report.measurement_note.lower()


@pytest.mark.public_data
def test_derived_cortical_notch_is_weaker_than_the_public_shaft(sts028_shafts: dict[str, object]) -> None:
    intact = sts028_shafts["intact"]
    notch = sts028_shafts["notch"]
    assert notch.comparison == "weaker"
    assert notch.measurement_reliable
    assert notch.signed_percent_weaker > intact.signed_percent_weaker + 15
    assert notch.failure_load_n < intact.failure_load_n * 0.8
    assert notch.failure_load_n > 5_000


@pytest.mark.public_data
def test_synthetic_radiograph_tracks_public_ct_diameters(sts028_shafts: dict[str, object]) -> None:
    volume = sts028_shafts["volume"]
    spacing = sts028_shafts["spacing"]
    crop = volume[80:160, 180:380, 100:220]
    pixel = float(spacing[1])
    truth_ap, truth_lateral, truth_inner = _cortical_widths(crop, pixel)
    density = np.clip(crop, 0.0, None)
    ap = (density.sum(axis=1) * pixel).astype(np.float32)
    lateral = (density.sum(axis=2) * pixel).astype(np.float32)
    film_spacing = (float(spacing[0]), pixel)
    report = analyze_radiograph_views(
        {"ap": (ap, film_spacing), "lateral": (lateral, film_spacing)},
        58.0,
        {"ap": False, "lateral": False},
    )
    measures = report.radiograph_measures
    assert measures is not None
    assert measures["canal_resolved"] is True
    assert report.measurement_reliable
    assert report.comparison in {"weaker", "matched"}
    if report.comparison == "weaker":
        assert report.percent_vs_normal < 40
    assert 10_000 <= report.failure_load_n <= 80_000
    views = measures["views"]
    assert views["ap"]["outer_diameter_mm"] == pytest.approx(truth_ap, abs=4.0)
    assert views["lateral"]["outer_diameter_mm"] == pytest.approx(truth_lateral, abs=4.0)
    assert views["ap"]["inner_diameter_mm"] == pytest.approx(truth_inner, abs=4.0)
    assert views["lateral"]["inner_diameter_mm"] == pytest.approx(truth_inner, abs=4.0)


def test_unresolved_adult_shaft_is_not_a_precise_percent() -> None:
    image = _flat_shaft(22.0)
    report = analyze_radiograph_views({"ap": (image, (0.5, 0.5))}, 70.0, {"ap": False})
    assert report.comparison == "unreliable"
    assert report.unreliable_reason == "canal"
    assert report.measurement_reliable is False
    assert "not resolved" in report.measurement_note.lower()
    assert "%" not in report.measurement_note
    assert "weaker than" not in report.measurement_note.lower()
    assert "tolerate unassisted" not in report.walking_text.lower()


def test_missing_pixel_spacing_is_not_a_walking_clearance() -> None:
    image, spacing = project_reference(64, 1.0, "ap")
    report = analyze_radiograph_views(
        {"ap": (image, spacing)},
        70.0,
        {"ap": False},
        {"ap": {"row_mm": spacing[0], "col_mm": spacing[1], "source": "assumed", "assumed": True}},
    )
    assert report.comparison == "unreliable"
    assert report.unreliable_reason == "spacing"
    assert report.measurement_reliable is False
    assert "clearance to walk" in report.measurement_note.lower()
    assert "tolerate unassisted" not in report.walking_text.lower()
    assert "not estimated" in report.walking_text.lower()


def test_unsupported_modality_does_not_score_walking(tmp_path: Path) -> None:
    folder = write_dicom_series(np.zeros((4, 16, 16), dtype=np.float32), (2.0, 2.0, 2.0), tmp_path, modality="OT")
    for path in folder.glob("*.dcm"):
        dataset = pydicom.dcmread(str(path))
        dataset.Modality = "OT"
        dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.67"
        dataset.save_as(str(path))
    with pytest.raises(StrengthAnalysisError) as caught:
        analyze_dicom_path(folder, body_mass_kg=70.0)
    message = str(caught.value).lower()
    assert caught.value.modality == "unknown"
    assert "walk" not in message
    assert "clearance" not in message


def test_missing_header_weight_is_not_called_measured(tmp_path: Path) -> None:
    hu, spacing = synthetic_tibfib_volume(72, 2.5)
    folder = write_dicom_series(hu, spacing, tmp_path / "entered")
    entered = analyze_dicom_path(folder, body_mass_kg=70.0, prefer_dicom_weight=True)
    assert entered.body_mass_kg == pytest.approx(70.0)
    assert entered.body_mass_source == "entered"
    assert any("not a measured patient weight" in line for line in entered.assumptions)
    assert entered.comparison == "matched"

    labeled = write_dicom_series(hu, spacing, tmp_path / "header", patient_weight_kg=62.0)
    from_header = analyze_dicom_path(labeled, body_mass_kg=70.0, prefer_dicom_weight=True)
    assert from_header.body_mass_kg == pytest.approx(62.0)
    assert from_header.body_mass_source == "dicom"
    assert from_header.comparison == "matched"
