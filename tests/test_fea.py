from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from cpt_predictor.fea import FEBioModelBuilder
from cpt_predictor.models import BraceModel, MaterialResult


class _FakeMesh:
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
        self.cells = np.asarray([4, 0, 1, 2, 3], dtype=np.int64)
        self.cell_data = {
            "material_bin": np.asarray([1], dtype=np.int64),
            "youngs_modulus_mpa": np.asarray([1200.0], dtype=float),
            "density_g_cm3": np.asarray([1.8], dtype=float),
        }


def _config() -> dict:
    return {
        "patient": {"body_mass_kg": 70.0},
        "loads": {
            "distal_fixation_fraction": 0.2,
            "proximal_load_fraction": 0.2,
            "gait_phases": [
                {
                    "name": "mid_stance",
                    "axial_bodyweight_multiplier": 2.5,
                    "bending_moment_nm": 12.0,
                    "torsion_nm": 2.0,
                }
            ],
        },
        "materials": {"poisson_ratio": 0.3},
        "simulation": {
            "time_steps": 10,
            "step_size": 0.1,
        },
    }


def test_write_model_emits_febio_v4_control_boundary_and_load_syntax(tmp_path: Path) -> None:
    builder = FEBioModelBuilder(_config())
    mesh = _FakeMesh()
    material_result = MaterialResult(
        mesh=mesh,
        mesh_path=tmp_path / "mesh.vtu",
        materials_table=[],
    )
    brace = BraceModel(
        enabled=False,
        surface=None,
        surface_path=None,
        source="none",
    )

    setup = builder.write_model(
        mesh_result=None,
        material_result=material_result,
        brace=brace,
        output_dir=tmp_path,
    )

    tree = ET.parse(setup.feb_path)
    root = tree.getroot()

    assert root.findtext("Control/analysis") == "STATIC"
    solver = root.find("Control/solver")
    assert solver is not None
    assert solver.attrib == {}
    assert solver.find("qn_method") is not None
    assert root.find("LoadData") is None

    distal_nodes = root.find("Mesh/NodeSet[@name='distal_nodes']")
    proximal_nodes = root.find("Mesh/NodeSet[@name='proximal_nodes']")
    assert distal_nodes is not None and distal_nodes.text == "1,2,3"
    assert proximal_nodes is not None and proximal_nodes.text == "4"

    boundary_bc = root.find("Boundary/bc[@type='zero displacement']")
    assert boundary_bc is not None
    assert boundary_bc.attrib["node_set"] == "distal_nodes"
    assert boundary_bc.findtext("x_dof") == "1"
    assert boundary_bc.findtext("y_dof") == "1"
    assert boundary_bc.findtext("z_dof") == "1"

    loads = root.findall("Loads/nodal_load")
    assert len(loads) == 2
    assert all(load.attrib["type"] == "nodal_load" for load in loads)
    assert [load.findtext("dof") for load in loads] == ["z", "x"]
    assert all(load.find("scale") is not None for load in loads)


def test_write_model_uses_cg_solver_for_large_meshes_or_forced_config(tmp_path: Path) -> None:
    config = _config()
    config["simulation"]["febio_solver_type"] = "CG-solid"
    builder = FEBioModelBuilder(config)
    mesh = _FakeMesh()
    material_result = MaterialResult(
        mesh=mesh,
        mesh_path=tmp_path / "mesh.vtu",
        materials_table=[],
    )
    brace = BraceModel(
        enabled=False,
        surface=None,
        surface_path=None,
        source="none",
    )

    setup = builder.write_model(
        mesh_result=None,
        material_result=material_result,
        brace=brace,
        output_dir=tmp_path,
    )

    tree = ET.parse(setup.feb_path)
    root = tree.getroot()
    solver = root.find("Control/solver")

    assert solver is not None
    assert solver.attrib == {"type": "CG-solid"}
    assert solver.findtext("lsiter") == "10"
    assert solver.findtext("cgmethod") == "0"
    assert solver.findtext("preconditioner") == "1"
    assert solver.find("qn_method") is None
    assert setup.stats["solver_type"] == "CG-solid"
