from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .meshing import tetra_connectivity
from .models import BraceModel, FEBioSetup, MaterialResult, MeshResult


def _indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + ("  " * level)
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indent + "  "
        for child in element:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indent


class FEBioModelBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _resolve_solver_type(self, element_count: int) -> str:
        requested = str(self.config["simulation"].get("febio_solver_type", "auto")).strip() or "auto"
        requested_lower = requested.lower()
        if requested_lower != "auto":
            return requested
        iterative_threshold = int(self.config["simulation"].get("febio_iterative_solver_min_elements", 200000))
        if element_count >= iterative_threshold:
            return "CG-solid"
        return "solid"

    def _build_solver_tag(self, control: ET.Element, element_count: int) -> str:
        solver_type = self._resolve_solver_type(element_count)
        solver_attrs = {"type": solver_type} if solver_type != "solid" else {}
        solver = ET.SubElement(control, "solver", solver_attrs)

        ET.SubElement(solver, "dtol").text = f"{float(self.config['simulation'].get('febio_dtol', 1.0e-3)):.6g}"
        ET.SubElement(solver, "etol").text = f"{float(self.config['simulation'].get('febio_etol', 1.0e-2)):.6g}"
        ET.SubElement(solver, "rtol").text = f"{float(self.config['simulation'].get('febio_rtol', 0.0)):.6g}"
        ET.SubElement(solver, "lstol").text = f"{float(self.config['simulation'].get('febio_lstol', 0.9)):.6g}"
        ET.SubElement(solver, "min_residual").text = f"{float(self.config['simulation'].get('febio_min_residual', 1.0e-20)):.6g}"

        if solver_type == "CG-solid":
            ET.SubElement(solver, "lsmin").text = f"{float(self.config['simulation'].get('febio_lsmin', 1.0e-15)):.6g}"
            ET.SubElement(solver, "lsiter").text = str(int(self.config["simulation"].get("febio_lsiter", 10)))
            ET.SubElement(solver, "cgmethod").text = str(int(self.config["simulation"].get("febio_cg_method", 0)))
            ET.SubElement(solver, "preconditioner").text = str(
                int(self.config["simulation"].get("febio_preconditioner", 1))
            )
            return solver_type

        ET.SubElement(solver, "max_refs").text = str(int(self.config["simulation"].get("febio_max_refs", 25)))
        ET.SubElement(solver, "diverge_reform").text = "1"
        ET.SubElement(solver, "reform_each_time_step").text = "1"
        qn_method = ET.SubElement(
            solver,
            "qn_method",
            {"type": str(self.config["simulation"].get("febio_qn_method", "BFGS"))},
        )
        ET.SubElement(qn_method, "max_ups").text = str(int(self.config["simulation"].get("febio_qn_max_ups", 10)))
        return solver_type

    def _build_node_sets(self, mesh: Any, brace: BraceModel) -> Dict[str, List[int]]:
        points = np.asarray(mesh.points)
        z_coords = points[:, 2]
        z_min = float(z_coords.min())
        z_max = float(z_coords.max())
        length = max(1e-6, z_max - z_min)

        fixation_fraction = float(self.config["loads"].get("distal_fixation_fraction", 0.08))
        load_fraction = float(self.config["loads"].get("proximal_load_fraction", 0.08))

        distal_nodes = np.where(z_coords <= (z_min + fixation_fraction * length))[0] + 1
        proximal_nodes = np.where(z_coords >= (z_max - load_fraction * length))[0] + 1

        node_sets = {
            "distal_nodes": distal_nodes.astype(int).tolist(),
            "proximal_nodes": proximal_nodes.astype(int).tolist(),
        }

        if brace.enabled and brace.support_bounds_xyz:
            x0, x1, y0, y1, z0, z1 = brace.support_bounds_xyz
            brace_mask = (
                (points[:, 0] >= x0)
                & (points[:, 0] <= x1)
                & (points[:, 1] >= y0)
                & (points[:, 1] <= y1)
                & (points[:, 2] >= z0)
                & (points[:, 2] <= z1)
            )
            brace_nodes = np.where(brace_mask)[0] + 1
            if len(brace_nodes):
                node_sets["brace_support_nodes"] = brace_nodes.astype(int).tolist()

        return node_sets

    def write_model(
        self,
        mesh_result: MeshResult,
        material_result: MaterialResult,
        brace: BraceModel,
        output_dir: Path,
    ) -> FEBioSetup:
        output_dir.mkdir(parents=True, exist_ok=True)

        mesh = material_result.mesh
        points = np.asarray(mesh.points)
        cells = tetra_connectivity(mesh)
        material_bins = np.asarray(mesh.cell_data["material_bin"]).astype(int)
        modulus = np.asarray(mesh.cell_data["youngs_modulus_mpa"]).astype(float)
        density = np.asarray(mesh.cell_data["density_g_cm3"]).astype(float)

        node_sets = self._build_node_sets(mesh, brace)
        patient_mass = float(self.config["patient"]["body_mass_kg"])
        body_weight_n = patient_mass * 9.81
        peak_phase = max(
            self.config["loads"]["gait_phases"],
            key=lambda phase: float(phase["axial_bodyweight_multiplier"]),
        )
        peak_force_n = body_weight_n * float(peak_phase["axial_bodyweight_multiplier"])
        lateral_force_n = (float(peak_phase["bending_moment_nm"]) * 1000.0) / max(
            1.0, points[:, 2].max() - points[:, 2].min()
        )

        root = ET.Element("febio_spec", {"version": "4.0"})
        ET.SubElement(root, "Module", {"type": "solid"})

        control = ET.SubElement(root, "Control")
        ET.SubElement(control, "analysis").text = "STATIC"
        ET.SubElement(control, "time_steps").text = str(int(self.config["simulation"].get("time_steps", 10)))
        ET.SubElement(control, "step_size").text = str(float(self.config["simulation"].get("step_size", 0.1)))
        ET.SubElement(control, "plot_level").text = "PLOT_MAJOR_ITRS"
        solver_type = self._build_solver_tag(control, int(cells.shape[0]))

        material_tag = ET.SubElement(root, "Material")
        unique_bins = sorted({int(value) for value in material_bins.tolist()})
        material_names: Dict[int, str] = {}
        for material_id, bin_id in enumerate(unique_bins, start=1):
            bin_mask = material_bins == bin_id
            material_name = f"bone_mat_{bin_id}"
            material_names[bin_id] = material_name
            entry = ET.SubElement(
                material_tag,
                "material",
                {"id": str(material_id), "name": material_name, "type": "isotropic elastic"},
            )
            ET.SubElement(entry, "E").text = f"{float(np.mean(modulus[bin_mask])):.6f}"
            ET.SubElement(entry, "v").text = f"{float(self.config['materials'].get('poisson_ratio', 0.30)):.4f}"
            ET.SubElement(entry, "density").text = f"{float(np.mean(density[bin_mask]) * 1e-9):.12f}"

        mesh_tag = ET.SubElement(root, "Mesh")
        nodes_tag = ET.SubElement(mesh_tag, "Nodes", {"name": "bone_nodes"})
        for node_id, point in enumerate(points, start=1):
            ET.SubElement(nodes_tag, "node", {"id": str(node_id)}).text = ",".join(f"{coord:.6f}" for coord in point)

        element_counter = 1
        for bin_id in unique_bins:
            elems_tag = ET.SubElement(mesh_tag, "Elements", {"type": "tet4", "name": f"bone_domain_bin_{bin_id}"})
            for cell in cells[material_bins == bin_id]:
                ET.SubElement(elems_tag, "elem", {"id": str(element_counter)}).text = ",".join(
                    str(int(node_index) + 1) for node_index in cell
                )
                element_counter += 1

        for set_name, node_ids in node_sets.items():
            set_tag = ET.SubElement(mesh_tag, "NodeSet", {"name": set_name})
            set_tag.text = ",".join(str(int(node_id)) for node_id in node_ids)

        domains_tag = ET.SubElement(root, "MeshDomains")
        for bin_id in unique_bins:
            ET.SubElement(
                domains_tag,
                "SolidDomain",
                {"name": f"bone_domain_bin_{bin_id}", "mat": material_names[bin_id]},
            )

        boundary = ET.SubElement(root, "Boundary")
        distal_bc = ET.SubElement(boundary, "bc", {"type": "zero displacement", "node_set": "distal_nodes"})
        ET.SubElement(distal_bc, "x_dof").text = "1"
        ET.SubElement(distal_bc, "y_dof").text = "1"
        ET.SubElement(distal_bc, "z_dof").text = "1"
        if "brace_support_nodes" in node_sets:
            brace_bc = ET.SubElement(
                boundary,
                "bc",
                {"type": "zero displacement", "node_set": "brace_support_nodes"},
            )
            ET.SubElement(brace_bc, "x_dof").text = "1"
            ET.SubElement(brace_bc, "y_dof").text = "1"
            ET.SubElement(brace_bc, "z_dof").text = "0"

        loads = ET.SubElement(root, "Loads")
        proximal_count = max(1, len(node_sets["proximal_nodes"]))
        axial_load = ET.SubElement(loads, "nodal_load", {"type": "nodal_load", "node_set": "proximal_nodes"})
        ET.SubElement(axial_load, "dof").text = "z"
        ET.SubElement(axial_load, "scale").text = f"{(-peak_force_n / proximal_count):.6f}"
        lateral_load = ET.SubElement(loads, "nodal_load", {"type": "nodal_load", "node_set": "proximal_nodes"})
        ET.SubElement(lateral_load, "dof").text = "x"
        ET.SubElement(lateral_load, "scale").text = f"{(lateral_force_n / proximal_count):.6f}"

        output = ET.SubElement(root, "Output")
        plotfile = ET.SubElement(output, "plotfile", {"type": "febio"})
        ET.SubElement(plotfile, "var", {"type": "displacement"})
        ET.SubElement(plotfile, "var", {"type": "stress"})
        ET.SubElement(plotfile, "var", {"type": "Lagrange strain"})

        _indent_xml(root)

        feb_path = output_dir / "model.feb"
        tree = ET.ElementTree(root)
        tree.write(feb_path, encoding="utf-8", xml_declaration=True)

        manifest = {
            "node_sets": node_sets,
            "peak_phase": peak_phase,
            "peak_force_n": peak_force_n,
            "lateral_force_n": lateral_force_n,
            "brace_enabled": brace.enabled,
            "brace_source": brace.source,
            "febio_solver_type": solver_type,
            "material_table": material_result.materials_table,
        }
        manifest_path = output_dir / "simulation_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return FEBioSetup(
            feb_path=feb_path,
            manifest_path=manifest_path,
            node_sets=node_sets,
            load_summary=manifest,
            stats={
                "element_count": int(cells.shape[0]),
                "node_count": int(points.shape[0]),
                "solver_type": solver_type,
            },
        )
