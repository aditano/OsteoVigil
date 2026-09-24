"""Solver, material, modality, and comparison tests for tib/fib strength."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from cpt_predictor.comparison import project_views, walking_assessment
from cpt_predictor.errors import StrengthAnalysisError
from cpt_predictor.materials import apparent_density_g_cm3, youngs_modulus_from_density, yield_strength_from_density
from cpt_predictor.modality import classify_header
from cpt_predictor.radiograph import project_reference
from cpt_predictor.reference_leg import synthetic_tibfib_volume
from cpt_predictor.segmentation import segment_tibia_and_fibula
from cpt_predictor.strength_pipeline import analyze_ct_volume, analyze_mri_volume, analyze_radiograph_views, projections_for_report
from cpt_predictor.voxel_fea import solve_voxel_elasticity
from scipy import ndimage as ndi


def test_cortical_hu_window_is_12_to_18_gpa() -> None:
    modulus_1200 = float(youngs_modulus_from_density(apparent_density_g_cm3(1200)))
    modulus_1800 = float(youngs_modulus_from_density(apparent_density_g_cm3(1800)))
    assert 11_000 <= modulus_1200 <= 13_000
    assert 17_000 <= modulus_1800 <= 19_000
    assert modulus_1200 < modulus_1800


def _uniform_cylinder(radius_mm: float = 6.0, length_mm: float = 80.0, spacing: float = 2.0, hu: float = 1500.0):
    nz = int(round(length_mm / spacing))
    nxy = int(np.ceil((radius_mm * 2 + 8) / spacing))
    _z, y, x = np.indices((nz, nxy, nxy))
    center = (nxy - 1) / 2.0
    mask = (y - center) ** 2 + (x - center) ** 2 <= (radius_mm / spacing) ** 2
    values = np.where(mask, hu, -500.0)
    density = apparent_density_g_cm3(values)
    return (
        mask,
        youngs_modulus_from_density(density),
        yield_strength_from_density(density),
        (spacing, spacing, spacing),
    )


def test_voxel_cylinder_matches_analytical_axial_stiffness() -> None:
    mask, modulus, strength, spacing = _uniform_cylinder()
    force = 1000.0
    result = solve_voxel_elasticity(modulus, strength, mask, spacing, force)
    spacing_z = spacing[0]
    mid = mask.shape[0] // 2
    area = float(mask[mid].sum()) * spacing[1] * spacing[2]
    young = float(np.median(modulus[mask]))
    length = mask.shape[0] * spacing_z
    analytical = young * area / length
    assert result.residual_norm < 1.0e-6
    assert abs(result.stiffness_n_per_mm / analytical - 1.0) < 0.05
    assert result.mean_proximal_displacement_mm < 0


def test_lower_hu_cylinder_is_weaker_by_about_the_yield_ratio() -> None:
    force = 800.0
    high_mask, high_mod, high_yield, spacing = _uniform_cylinder(hu=1600)
    low_mask, low_mod, low_yield, _spacing = _uniform_cylinder(hu=1100)
    high = solve_voxel_elasticity(high_mod, high_yield, high_mask, spacing, force)
    low = solve_voxel_elasticity(low_mod, low_yield, low_mask, spacing, force)
    expected = float(np.median(low_yield[low_mask]) / np.median(high_yield[high_mask]))
    actual = low.failure_load_n / high.failure_load_n
    assert actual < 0.9
    assert abs(actual - expected) < 0.08


def test_reference_leg_matches_itself_and_keeps_the_fibula() -> None:
    hu, spacing = synthetic_tibfib_volume(72, 2.5)
    masks = segment_tibia_and_fibula(hu, spacing)
    assert masks.tibia_voxels > masks.fibula_voxels > 0
    report = analyze_ct_volume(hu, spacing, 70)
    assert report.comparison == "matched"
    assert report.percent_vs_normal == 0
    assert report.method == "voxel_hexahedral_linear_elastic"
    assert "surrogate" not in report.solver
    assert report.walking_verdict == "tolerated"
    assert report.hotspot_z_mm is None


def test_notch_is_weaker_and_both_views_peak_on_it() -> None:
    notch_mm = 46.0
    hu, spacing = synthetic_tibfib_volume(72, 2.5, notch_center_mm=notch_mm, notch_depth_mm=9.0, notch_half_width_mm=3.0)
    report = analyze_ct_volume(hu, spacing, 70)
    assert report.comparison == "weaker"
    assert report.percent_vs_normal > 15
    assert report.hotspot_z_mm is not None
    assert abs(report.hotspot_z_mm - notch_mm) <= 6.0
    ap, lateral = projections_for_report(report)
    assert ap is not None and lateral is not None
    ap_z = (int(np.argmax(np.max(ap, axis=1))) + 0.5) * report.weakness_spacing_zyx[0]
    lat_z = (int(np.argmax(np.max(lateral, axis=1))) + 0.5) * report.weakness_spacing_zyx[0]
    assert abs(ap_z - notch_mm) <= 6.0
    assert abs(lat_z - notch_mm) <= 6.0


def test_header_classification_for_ct_mr_dx_and_cr() -> None:
    assert classify_header(modality="CT").kind == "ct"
    assert classify_header(modality="MR").kind == "mri"
    assert classify_header(modality="DX").kind == "radiograph"
    assert classify_header(modality="CR").kind == "radiograph"
    assert classify_header(modality="", sop_class_uid="1.2.840.10008.5.1.4.1.1.2").kind == "ct"
    assert classify_header(modality="OT").kind is None


def test_walking_thresholds_follow_orthoload() -> None:
    low = walking_assessment(2.0 * 70 * 9.80665, 70)
    mid = walking_assessment(3.2 * 70 * 9.80665, 70)
    high = walking_assessment(5.0 * 70 * 9.80665, 70)
    assert low["walking_verdict"] == "not_expected"
    assert "not expected to tolerate level walking" in str(low["walking_text"]).lower()
    assert mid["walking_verdict"] == "limited"
    assert "stairs" in str(mid["walking_text"]).lower()
    assert high["walking_verdict"] == "tolerated"
    assert "ordinary daily loads" in str(high["walking_text"]).lower()


def test_severely_thin_bone_is_not_cleared_for_unassisted_walking() -> None:
    hu, spacing = synthetic_tibfib_volume(
        48,
        2.0,
        cortical_hu=220,
        trabecular_hu=40,
        tibia_radius_mm=4.5,
        tibia_cortex_mm=2.0,
        include_fibula=False,
    )
    report = analyze_ct_volume(hu, spacing, 70)
    assert report.failure_load_bodyweights < 2.8
    assert report.walking_verdict == "not_expected"


def test_mri_geometry_matches_normal_and_noise_is_rejected() -> None:
    hu, spacing = synthetic_tibfib_volume(56, 2.0)
    cortex = hu >= 1000
    marrow = (hu > 0) & ~cortex
    mri = np.zeros(hu.shape, dtype=np.float32)
    mri[cortex] = 25
    mri[marrow] = 1400
    mri[ndi.binary_dilation(hu > 0, iterations=2) & ~(hu > 0)] = 800
    report = analyze_mri_volume(mri, spacing, 70)
    assert report.modality == "mri"
    assert report.comparison == "matched"
    assert "cannot measure bone density" in report.assumptions[0]
    with pytest.raises(StrengthAnalysisError):
        analyze_mri_volume(np.random.default_rng(1).random((20, 20, 20)).astype(np.float32), (2, 2, 2), 70)


def test_radiograph_of_the_reference_matches_and_thin_cortex_is_weaker() -> None:
    image, spacing = project_reference(64, 1.0, "ap")
    normal = analyze_radiograph_views({"ap": (image, spacing)}, 70, {"ap": False})
    assert normal.comparison == "matched"
    assert "circular" in normal.assumptions[0].lower()
    thin_volume, thin_spacing = synthetic_tibfib_volume(64, 1.0, tibia_cortex_mm=2.0, fibula_cortex_mm=1.2)
    thin = (np.clip(thin_volume, 0, None).sum(axis=1) * thin_spacing[1]).astype(np.float32)
    weaker = analyze_radiograph_views({"ap": (thin, (thin_spacing[0], thin_spacing[2]))}, 70, {"ap": False})
    assert weaker.comparison == "weaker"
    assert weaker.percent_vs_normal > 10
    ap, lateral = project_views(np.zeros((4, 3, 5)))
    assert ap.shape == (4, 5)
    assert lateral.shape == (4, 3)


def test_radiograph_rasters_include_the_image_and_cortical_bone() -> None:
    image, spacing = project_reference(64, 1.0, "ap")
    lateral, lateral_spacing = project_reference(64, 1.0, "lateral")
    report = analyze_radiograph_views(
        {"ap": (image, spacing), "lateral": (lateral, lateral_spacing)},
        70,
        {"ap": False, "lateral": False},
    )
    assert report.weakness is None
    assert report.views_acquired == {"ap": True, "lateral": True}
    payload = report.public_dict()
    assert "weakness" not in payload
    for name in ("ap", "lateral"):
        raster = report.rasters[name]
        bone = raster >= 2
        assert float(bone.mean()) > 0.01
        assert float((~bone).mean()) > 0.01
        fraction = raster[bone] - np.floor(raster[bone])
        assert float(np.std(fraction)) > 0.02
        encoded = base64.b64decode(payload["rasters"][name]["b64"])
        restored = np.frombuffer(encoded, dtype="<f4").reshape(payload["rasters"][name]["shape"])
        assert restored.shape == raster.shape
        assert np.allclose(restored, raster)
    thin_volume, thin_spacing = synthetic_tibfib_volume(64, 1.0, tibia_cortex_mm=2.0, fibula_cortex_mm=1.2)
    thin = (np.clip(thin_volume, 0, None).sum(axis=1) * thin_spacing[1]).astype(np.float32)
    weaker = analyze_radiograph_views({"ap": (thin, (thin_spacing[0], thin_spacing[2]))}, 70, {"ap": False})
    normal_step = np.floor(report.rasters["ap"][report.rasters["ap"] >= 2] - 2)
    weak_step = np.floor(weaker.rasters["ap"][weaker.rasters["ap"] >= 2] - 2)
    assert float(np.median(weak_step)) > float(np.median(normal_step)) + 0.5
