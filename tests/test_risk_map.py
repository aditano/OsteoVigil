from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cpt_predictor.analysis import RiskAnalyzer
from cpt_predictor.models import SimulationResult
from cpt_predictor.risk_map import (
    VISIBLE_UTILIZATION_HEALTHY,
    assess_defect_from_samples,
    attach_clinical_risk_fields,
    failure_overlay_rgba,
    failure_utilization,
)


class _Mesh:
    def __init__(self, z, hu, von_mises, yield_strength, modulus=None):
        n = len(z)
        self.n_cells = n
        self.points = np.column_stack([np.zeros(n), np.zeros(n), np.asarray(z, dtype=float)])
        self.cell_data = {
            "HU": np.asarray(hu, dtype=float),
            "von_mises_mpa": np.asarray(von_mises, dtype=float),
            "yield_strength_mpa": np.asarray(yield_strength, dtype=float),
            "youngs_modulus_mpa": np.asarray(modulus if modulus is not None else np.where(np.asarray(hu) >= 300, 12000.0, 800.0), dtype=float),
            "safety_factor": np.asarray(yield_strength, dtype=float) / np.maximum(np.asarray(von_mises, dtype=float), 0.1),
            "fatigue_cycles": np.full(n, 1.0e12, dtype=float),
        }
        self.field_data = {}

    def cell_centers(self):
        return SimpleNamespace(points=self.points.copy())


def _healthy_mesh():
    z = np.linspace(0.0, 100.0, 80)
    hu = np.full(z.shape, 1100.0)
    hu[(z > 20) & (z < 80)] = 1050.0
    stress = np.full(z.shape, 8.0)
    strength = np.full(z.shape, 120.0)
    return _Mesh(z, hu, stress, strength)


def _defect_mesh():
    z = np.linspace(0.0, 100.0, 80)
    hu = np.full(z.shape, 1100.0)
    hu[(z > 42) & (z < 52)] = 90.0
    stress = np.full(z.shape, 8.0)
    stress[(z > 42) & (z < 52)] = 6.0
    strength = np.full(z.shape, 120.0)
    strength[(z > 42) & (z < 52)] = 4.0
    return _Mesh(z, hu, stress, strength)


def test_healthy_length_is_not_flagged_as_a_defect():
    z = np.linspace(0.0, 120.0, 200)
    hu = np.full(z.shape, 350.0)
    hu += 40.0 * np.sin(z / 8.0)
    assessment = assess_defect_from_samples(z, hu)
    assert assessment.has_defect is False


def test_localized_low_hu_band_is_flagged_as_a_defect():
    z = np.linspace(0.0, 120.0, 200)
    hu = np.full(z.shape, 360.0)
    hu[(z > 52) & (z < 64)] = 90.0
    assessment = assess_defect_from_samples(z, hu)
    assert assessment.has_defect is True
    assert 48.0 < float(assessment.defect_z_mm) < 70.0


def test_healthy_mesh_overlay_stays_empty_and_risk_is_lower(tmp_path):
    mesh = _healthy_mesh()
    assessment = attach_clinical_risk_fields(mesh)
    overlay = np.asarray(mesh.cell_data["clinical_utilization"], dtype=float)
    assert assessment.has_defect is False
    assert not np.isfinite(overlay).any()
    analyzer = RiskAnalyzer({"patient": {"steps_per_day": 6000}})
    result = SimulationResult(
        mode="linear_tet_fea",
        mesh=mesh,
        mesh_path=tmp_path / "mesh.vtu",
        summary={"years_to_failure_estimate": 0.1, "governing_phase": "mid_stance", "has_structural_defect": False},
        log_path=tmp_path / "log.json",
    )
    risk = analyzer.analyze(result, tmp_path, "none")
    assert risk.summary["risk_category"] == "lower"
    assert risk.summary["has_structural_defect"] is False


def test_defect_mesh_lights_up_near_yield_and_raises_risk(tmp_path):
    mesh = _defect_mesh()
    assessment = attach_clinical_risk_fields(mesh)
    overlay = np.asarray(mesh.cell_data["clinical_utilization"], dtype=float)
    assert assessment.has_defect is True
    assert np.isfinite(overlay).any()
    assert float(np.nanmax(overlay)) > 0.9
    # Cortex away from the band should remain unshaded.
    z = mesh.points[:, 2]
    far = np.isfinite(overlay) & ((z < 20.0) | (z > 80.0))
    assert not far.any()
    analyzer = RiskAnalyzer({"patient": {"steps_per_day": 6000}})
    result = SimulationResult(
        mode="linear_tet_fea",
        mesh=mesh,
        mesh_path=tmp_path / "mesh.vtu",
        summary={"years_to_failure_estimate": 10.0, "governing_phase": "mid_stance", "has_structural_defect": True},
        log_path=tmp_path / "log.json",
    )
    risk = analyzer.analyze(result, tmp_path, "none")
    assert risk.summary["has_structural_defect"] is True
    assert risk.summary["risk_category"] in {"high", "elevated"}
    assert risk.summary["min_safety_factor"] < 1.0


def test_failure_overlay_rgba_hides_healthy_utilization_and_marks_yield():
    image = np.array([[0.10, 0.50], [np.nan, 1.05]], dtype=float)
    rgba = failure_overlay_rgba(image, min_visible=0.33, vmax=1.0)
    assert rgba[0, 0, 3] == 0.0
    assert rgba[0, 1, 3] > 0.3
    assert rgba[1, 0, 3] == 0.0
    assert rgba[1, 1, 3] == 1.0
    assert float(failure_utilization(np.array([6.0]), np.array([3.0]))[0]) == 2.0
    assert VISIBLE_UTILIZATION_HEALTHY >= 0.8
