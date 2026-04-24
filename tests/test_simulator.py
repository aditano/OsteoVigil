from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cpt_predictor.models import FEBioSetup, SimulationResult
from cpt_predictor.simulator import FEBioRunner


def _config() -> dict:
    return {
        "patient": {"body_mass_kg": 55.0, "steps_per_day": 6000},
        "simulation": {
            "prefer_febio": True,
            "febio_exe": None,
            "surrogate_if_febio_unavailable": True,
            "fatigue_constant": 500000.0,
            "fatigue_exponent": 7.5,
        },
        "loads": {
            "gait_phases": [
                {
                    "name": "mid_stance",
                    "axial_bodyweight_multiplier": 2.0,
                    "bending_moment_nm": 10.0,
                    "torsion_nm": 2.0,
                }
            ]
        },
    }


class _FakeFEBioMesh:
    def __init__(self) -> None:
        self.points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.point_data = {
            "displacement": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.1],
                ],
                dtype=float,
            )
        }
        self.cell_data = {
            "stress": np.asarray([[10.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 6.0]], dtype=float),
            "Lagrange_strain": np.asarray([[0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.03]], dtype=float),
        }
        self.n_cells = 1
        self.n_points = 4

    def save(self, path: Path) -> None:
        Path(path).write_text("fake-febio-mesh", encoding="utf-8")


def test_febio_command_uses_model_path_relative_to_output_dir(tmp_path: Path) -> None:
    runner = FEBioRunner(_config())
    febio_setup = FEBioSetup(
        feb_path=tmp_path / "model.feb",
        manifest_path=tmp_path / "simulation_manifest.json",
        node_sets={},
        load_summary={},
    )

    command = runner._febio_command("/tmp/febio4", febio_setup, tmp_path)
    assert command == ["/tmp/febio4", "-i", "model.feb"]


def test_run_marks_successful_febio_completion_without_plain_surrogate_mode(
    tmp_path: Path, monkeypatch
) -> None:
    runner = FEBioRunner(_config())
    febio_setup = FEBioSetup(
        feb_path=tmp_path / "model.feb",
        manifest_path=tmp_path / "simulation_manifest.json",
        node_sets={},
        load_summary={},
    )
    febio_setup.feb_path.write_text("<febio />", encoding="utf-8")

    monkeypatch.setattr(runner, "_resolve_febio_executable", lambda: "/tmp/febio4")

    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    captured: dict[str, object] = {}

    def fake_run(command, cwd, capture_output, text, check):
        captured["command"] = command
        captured["cwd"] = cwd
        return Completed()

    monkeypatch.setattr("cpt_predictor.simulator.subprocess.run", fake_run)
    surrogate_called = {"value": False}

    def fake_surrogate(*args, **kwargs):
        surrogate_called["value"] = True
        raise AssertionError("surrogate path should not be used on successful FEBio import")

    monkeypatch.setattr(runner, "_run_surrogate", fake_surrogate)

    expected_result = SimulationResult(
        mode="febio_results_vtk",
        mesh=SimpleNamespace(cell_data={}, n_cells=0),
        mesh_path=tmp_path / "simulation_mesh.vtu",
        summary={"mode": "febio_results_vtk", "years_to_failure_estimate": 1.0},
    )
    monkeypatch.setattr(runner, "_load_febio_results", lambda *args, **kwargs: expected_result)

    material_result = SimpleNamespace(mesh=SimpleNamespace(), mesh_path=tmp_path / "mesh.vtu")
    result = runner.run(
        febio_setup,
        study=SimpleNamespace(),
        segmentation=SimpleNamespace(),
        material_result=material_result,
        brace=SimpleNamespace(),
        output_dir=tmp_path,
    )

    assert captured["command"] == ["/tmp/febio4", "-i", "model.feb"]
    assert surrogate_called["value"] is False
    assert result.mode == "febio_results_vtk"
    assert result.summary["mode"] == "febio_results_vtk"
    assert result.summary["febio_return_code"] == 0


def test_run_marks_febio_failure_fallback_explicitly(tmp_path: Path, monkeypatch) -> None:
    runner = FEBioRunner(_config())
    febio_setup = FEBioSetup(
        feb_path=tmp_path / "model.feb",
        manifest_path=tmp_path / "simulation_manifest.json",
        node_sets={},
        load_summary={},
    )
    febio_setup.feb_path.write_text("<febio />", encoding="utf-8")

    monkeypatch.setattr(runner, "_resolve_febio_executable", lambda: "/tmp/febio4")

    class Completed:
        returncode = 7
        stdout = "bad"
        stderr = "failed"

    def fake_run(command, cwd, capture_output, text, check):
        return Completed()

    monkeypatch.setattr("cpt_predictor.simulator.subprocess.run", fake_run)

    expected_result = SimulationResult(
        mode="surrogate",
        mesh=SimpleNamespace(cell_data={}, n_cells=0),
        mesh_path=tmp_path / "simulation_mesh.vtu",
        summary={"mode": "surrogate", "years_to_failure_estimate": 1.0},
    )
    monkeypatch.setattr(runner, "_run_surrogate", lambda *args, **kwargs: expected_result)

    material_result = SimpleNamespace(mesh=SimpleNamespace(), mesh_path=tmp_path / "mesh.vtu")
    result = runner.run(
        febio_setup,
        study=SimpleNamespace(),
        segmentation=SimpleNamespace(),
        material_result=material_result,
        brace=SimpleNamespace(),
        output_dir=tmp_path,
    )

    assert result.mode == "surrogate_febio_failed"
    assert result.summary["mode"] == "surrogate_febio_failed"
    assert result.summary["febio_return_code"] == 7


def test_build_febio_result_from_mesh_normalizes_real_febio_fields(tmp_path: Path) -> None:
    runner = FEBioRunner(_config())
    febio_setup = FEBioSetup(
        feb_path=tmp_path / "model.feb",
        manifest_path=tmp_path / "simulation_manifest.json",
        node_sets={},
        load_summary={
            "peak_phase": {"name": "mid_stance"},
            "febio_cell_order": [0],
        },
    )

    material_result = SimpleNamespace(
        mesh=SimpleNamespace(
            n_cells=1,
            cell_data={
                "yield_strength_mpa": np.asarray([100.0], dtype=float),
                "youngs_modulus_mpa": np.asarray([1200.0], dtype=float),
                "density_g_cm3": np.asarray([1.8], dtype=float),
                "material_bin": np.asarray([1], dtype=int),
            },
        )
    )

    result = runner._build_febio_result_from_mesh(
        _FakeFEBioMesh(),
        material_result=material_result,
        febio_setup=febio_setup,
        output_dir=tmp_path,
        vtk_path=tmp_path / "febio_results.0.vtk",
    )

    assert result.mode == "febio_results_vtk"
    assert np.isclose(result.mesh.points[3, 2], 1.1)
    assert np.isclose(result.mesh.cell_data["von_mises_mpa"][0], np.sqrt(12.0))
    assert np.isclose(result.mesh.cell_data["principal_strain"][0], 0.03)
    assert result.summary["governing_phase"] == "mid_stance"
    assert result.summary["vtk_result_path"].endswith("febio_results.0.vtk")
    assert result.mesh_path.exists()
