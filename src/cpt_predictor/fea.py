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
        ET.SubElement(control, "analysis", {"type": "static"})
        ET.SubElement(control, "time_steps").text = str(int(self.config["simulation"].get("time_steps", 10)))
        ET.SubElement(control, "step_size").text = str(float(self.config["simulation"].get("step_size", 0.1)))
        ET.SubElement(control, "plot_level").text = "PLOT_MAJOR_ITRS"

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
            for node_id in node_ids:
                ET.SubElement(set_tag, "node", {"id": str(int(node_id))})

        domains_tag = ET.SubElement(root, "MeshDomains")
        for bin_id in unique_bins:
            ET.SubElement(
                domains_tag,
                "SolidDomain",
                {"name": f"bone_domain_bin_{bin_id}", "mat": material_names[bin_id]},
            )

        boundary = ET.SubElement(root, "Boundary")
        ET.SubElement(boundary, "fix", {"bc": "x,y,z", "node_set": "distal_nodes"})
        if "brace_support_nodes" in node_sets:
            ET.SubElement(boundary, "fix", {"bc": "x,y", "node_set": "brace_support_nodes"})

        loads = ET.SubElement(root, "Loads")
        proximal_count = max(1, len(node_sets["proximal_nodes"]))
        ET.SubElement(loads, "nodal_load", {"bc": "z", "node_set": "proximal_nodes", "lc": "1"}).text = (
            f"{(-peak_force_n / proximal_count):.6f}"
        )
        ET.SubElement(loads, "nodal_load", {"bc": "x", "node_set": "proximal_nodes", "lc": "1"}).text = (
            f"{(lateral_force_n / proximal_count):.6f}"
        )

        loaddata = ET.SubElement(root, "LoadData")
        controller = ET.SubElement(loaddata, "load_controller", {"id": "1", "type": "loadcurve"})
        ET.SubElement(controller, "interpolate").text = "LINEAR"
        points_tag = ET.SubElement(controller, "points")
        ET.SubElement(points_tag, "pt").text = "0,0"
        ET.SubElement(points_tag, "pt").text = "1,1"

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
            "material_table": material_result.materials_table,
        }
        manifest_path = output_dir / "simulation_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return FEBioSetup(
            feb_path=feb_path,
            manifest_path=manifest_path,
            node_sets=node_sets,
            load_summary=manifest,
            stats={"element_count": int(cells.shape[0]), "node_count": int(points.shape[0])},
        )

