from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .fem_solver import solve_mesh_linear_tet_fea
from .models import BraceModel, FEBioSetup, MaterialResult, MeshResult, SegmentationResult, SimulationResult, StudyData
from .segmentation import locate_pseudarthrosis_slice
from .utils.febio_manager import resolve_managed_febio_executable


class FEBioRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _resolve_febio_executable(self) -> Optional[str]:
        configured = self.config["simulation"].get("febio_exe")
        env_value = os.getenv("FEBIO_EXE")
        executable = configured or env_value
        if executable and Path(executable).exists():
            return executable
        managed = resolve_managed_febio_executable()
        if managed and managed.exists():
            return str(managed)
        for candidate in ("febio4", "febio4.exe", "febio3", "febio3.exe"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return None

    def _febio_command(self, executable: str, febio_setup: FEBioSetup, output_dir: Path) -> list[str]:
        feb_path = Path(febio_setup.feb_path)
        try:
            model_arg = str(feb_path.relative_to(output_dir))
        except ValueError:
            model_arg = feb_path.name if feb_path.parent == output_dir else str(feb_path)
        return [executable, "-i", model_arg]

    @staticmethod
    def _normalize_data_name(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    def _match_result_array(self, data: Any, candidates: list[str]) -> Optional[np.ndarray]:
        normalized_lookup = {
            self._normalize_data_name(str(key)): key
            for key in getattr(data, "keys", lambda: [])()
        }
        for candidate in candidates:
            key = normalized_lookup.get(self._normalize_data_name(candidate))
            if key is not None:
                return np.asarray(data[key], dtype=float)
        return None

    @staticmethod
    def _coerce_tensor_array(values: np.ndarray, item_count: int, label: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim == 3 and array.shape[1:] == (3, 3) and array.shape[0] == item_count:
            return array
        if array.ndim == 2 and array.shape == (item_count, 9):
            return array.reshape((item_count, 3, 3))
        if array.ndim == 2 and array.shape == (item_count, 6):
            tensor = np.zeros((item_count, 3, 3), dtype=float)
            tensor[:, 0, 0] = array[:, 0]
            tensor[:, 1, 1] = array[:, 1]
            tensor[:, 2, 2] = array[:, 2]
            tensor[:, 0, 1] = tensor[:, 1, 0] = array[:, 3]
            tensor[:, 1, 2] = tensor[:, 2, 1] = array[:, 4]
            tensor[:, 0, 2] = tensor[:, 2, 0] = array[:, 5]
            return tensor
        raise RuntimeError(f"Unexpected FEBio {label} tensor shape: {array.shape}")

    @staticmethod
    def _von_mises_from_tensors(stress_tensors: np.ndarray) -> np.ndarray:
        sx = stress_tensors[:, 0, 0]
        sy = stress_tensors[:, 1, 1]
        sz = stress_tensors[:, 2, 2]
        sxy = stress_tensors[:, 0, 1]
        syz = stress_tensors[:, 1, 2]
        sxz = stress_tensors[:, 0, 2]
        return np.sqrt(
            0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
            + 3.0 * (sxy**2 + syz**2 + sxz**2)
        )

    def _find_latest_vtk_result(self, febio_setup: FEBioSetup, output_dir: Path) -> Path:
        plotfile_info = febio_setup.load_summary.get("febio_plotfile", {})
        base_name = str(plotfile_info.get("base_name", Path(febio_setup.feb_path).stem)).strip()
        if not base_name:
            base_name = Path(febio_setup.feb_path).stem

        indexed_paths = []
        for path in output_dir.glob(f"{base_name}.*.vtk"):
            suffix = path.name[len(base_name) + 1 : -4]
            if suffix.isdigit():
                indexed_paths.append((int(suffix), path))
        if indexed_paths:
            return sorted(indexed_paths, key=lambda item: item[0])[-1][1]

        direct_path = output_dir / f"{base_name}.vtk"
        if direct_path.exists():
            return direct_path

        raise FileNotFoundError(f"No FEBio VTK result files matching {base_name} were found in {output_dir}.")

    def _build_febio_result_from_mesh(
        self,
        result_mesh: Any,
        material_result: MaterialResult,
        febio_setup: FEBioSetup,
        output_dir: Path,
        vtk_path: Optional[Path] = None,
    ) -> SimulationResult:
        mesh = result_mesh

        displacement = self._match_result_array(mesh.point_data, ["displacement"])
        if displacement is not None and displacement.shape == (mesh.n_points, 3):
            mesh.points = np.asarray(mesh.points, dtype=float) + displacement

        source_mesh = material_result.mesh
        export_order = np.asarray(febio_setup.load_summary.get("febio_cell_order", []), dtype=int)
        if export_order.size == 0:
            export_order = np.arange(int(getattr(source_mesh, "n_cells", mesh.n_cells)), dtype=int)
        if export_order.size != mesh.n_cells:
            raise RuntimeError(
                "FEBio result cell count does not match the recorded export order: "
                f"{mesh.n_cells} vs {export_order.size}."
            )

        for field_name in ("HU", "density_g_cm3", "youngs_modulus_mpa", "yield_strength_mpa", "material_bin"):
            source_values = np.asarray(source_mesh.cell_data.get(field_name, []))
            if source_values.shape[:1] == (export_order.size,):
                mesh.cell_data[field_name] = source_values[export_order]

        stress_values = self._match_result_array(mesh.cell_data, ["stress", "cauchy_stress"])
        strain_values = self._match_result_array(mesh.cell_data, ["lagrange_strain", "lagrange strain"])
        if stress_values is None:
            raise RuntimeError("FEBio VTK results did not contain a stress tensor field.")
        if strain_values is None:
            raise RuntimeError("FEBio VTK results did not contain a Lagrange strain tensor field.")

        stress_tensors = self._coerce_tensor_array(stress_values, mesh.n_cells, "stress")
        strain_tensors = self._coerce_tensor_array(strain_values, mesh.n_cells, "Lagrange strain")
        strain_tensors = 0.5 * (strain_tensors + np.swapaxes(strain_tensors, 1, 2))

        von_mises = self._von_mises_from_tensors(stress_tensors)
        principal_strain = np.max(np.linalg.eigvalsh(strain_tensors), axis=1)

        strength = np.asarray(mesh.cell_data.get("yield_strength_mpa", np.full(mesh.n_cells, np.nan)), dtype=float)
        if strength.shape != (mesh.n_cells,):
            strength = np.full(mesh.n_cells, np.nan, dtype=float)

        stress_floor = np.maximum(von_mises, 0.1)
        safety_factor = np.divide(
            strength,
            stress_floor,
            out=np.full(mesh.n_cells, np.inf, dtype=float),
            where=np.isfinite(strength),
        )

        fatigue_constant = float(self.config["simulation"].get("fatigue_constant", 500000.0))
        fatigue_exponent = float(self.config["simulation"].get("fatigue_exponent", 7.5))
        fatigue_ratio = np.divide(
            strength,
            stress_floor,
            out=np.full(mesh.n_cells, np.inf, dtype=float),
            where=np.isfinite(strength),
        )
        fatigue_cycles = fatigue_constant * np.power(np.maximum(fatigue_ratio, 0.1), fatigue_exponent)
        fatigue_cycles[~np.isfinite(fatigue_cycles)] = np.inf

        mesh.cell_data["von_mises_mpa"] = von_mises
        mesh.cell_data["principal_strain"] = principal_strain
        mesh.cell_data["safety_factor"] = safety_factor
        mesh.cell_data["fatigue_cycles"] = fatigue_cycles

        derived = {
            "max_von_mises_mpa": float(np.max(von_mises)) if von_mises.size else 0.0,
            "min_safety_factor": float(np.min(safety_factor)) if safety_factor.size else float("inf"),
            "min_fatigue_cycles": float(np.min(fatigue_cycles)) if fatigue_cycles.size else float("inf"),
        }
        derived = self._finalize_clinical_fields(mesh, derived)

        mesh_path = output_dir / "simulation_mesh.vtu"
        mesh.save(mesh_path)
        peak_phase = febio_setup.load_summary.get("peak_phase", {})
        summary = {
            "mode": "febio_results_vtk",
            **derived,
            "governing_phase": str(peak_phase.get("name", "peak_load_case")),
            "result_mesh_source": "febio_vtk",
            "vtk_result_path": str(vtk_path) if vtk_path else "",
        }

        summary_path = output_dir / "simulation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return SimulationResult(
            mode="febio_results_vtk",
            mesh=mesh,
            mesh_path=mesh_path,
            summary=summary,
            log_path=summary_path,
        )

    def _load_febio_results(
        self,
        febio_setup: FEBioSetup,
        material_result: MaterialResult,
        output_dir: Path,
    ) -> SimulationResult:
        import pyvista as pv

        vtk_path = self._find_latest_vtk_result(febio_setup, output_dir)
        vtk_mesh = pv.read(vtk_path)
        return self._build_febio_result_from_mesh(vtk_mesh, material_result, febio_setup, output_dir, vtk_path=vtk_path)

    def _finalize_clinical_fields(self, mesh: Any, derived: Dict[str, Any]) -> Dict[str, Any]:
        from .risk_map import attach_clinical_risk_fields, clinical_summary

        assessment = attach_clinical_risk_fields(mesh)
        extra = clinical_summary(mesh, assessment)
        steps_per_day = max(1.0, float(self.config["patient"].get("steps_per_day", 6000)))
        min_cycles = float(extra["min_clinical_fatigue_cycles"])
        years_to_failure = float(min_cycles / (steps_per_day * 365.0)) if np.isfinite(min_cycles) else float("inf")
        merged = dict(derived)
        merged["min_safety_factor"] = float(extra["min_clinical_safety_factor"])
        merged["min_fatigue_cycles"] = min_cycles
        merged["years_to_failure_estimate"] = max(0.0, years_to_failure) if np.isfinite(years_to_failure) else float("inf")
        merged["max_clinical_utilization"] = float(extra["max_clinical_utilization"])
        merged["has_structural_defect"] = bool(extra["has_structural_defect"])
        merged["defect_reason"] = extra["defect_reason"]
        merged["defect_z_mm"] = extra["defect_z_mm"]
        merged["overlay_caption"] = extra["overlay_caption"]
        return merged

    def _attach_derived_fields(self, mesh: Any, von_mises: np.ndarray, principal_strain: np.ndarray) -> Dict[str, Any]:
        n_cells = int(getattr(mesh, "n_cells", von_mises.size))
        strength = np.asarray(mesh.cell_data.get("yield_strength_mpa", np.full(n_cells, np.nan)), dtype=float)
        if strength.shape != (n_cells,):
            strength = np.full(n_cells, np.nan, dtype=float)

        stress_floor = np.maximum(np.asarray(von_mises, dtype=float), 0.1)
        safety_factor = np.divide(
            strength,
            stress_floor,
            out=np.full(n_cells, np.inf, dtype=float),
            where=np.isfinite(strength),
        )
        fatigue_constant = float(self.config["simulation"].get("fatigue_constant", 500000.0))
        fatigue_exponent = float(self.config["simulation"].get("fatigue_exponent", 7.5))
        fatigue_ratio = np.divide(
            strength,
            stress_floor,
            out=np.full(n_cells, np.inf, dtype=float),
            where=np.isfinite(strength),
        )
        fatigue_cycles = fatigue_constant * np.power(np.maximum(fatigue_ratio, 0.1), fatigue_exponent)
        fatigue_cycles[~np.isfinite(fatigue_cycles)] = np.inf

        mesh.cell_data["von_mises_mpa"] = np.asarray(von_mises, dtype=float)
        mesh.cell_data["principal_strain"] = np.asarray(principal_strain, dtype=float)
        mesh.cell_data["safety_factor"] = safety_factor
        mesh.cell_data["fatigue_cycles"] = fatigue_cycles

        derived = {
            "max_von_mises_mpa": float(np.max(von_mises)) if von_mises.size else 0.0,
            "min_safety_factor": float(np.min(safety_factor)) if safety_factor.size else float("inf"),
            "min_fatigue_cycles": float(np.min(fatigue_cycles)) if fatigue_cycles.size else float("inf"),
        }
        return self._finalize_clinical_fields(mesh, derived)

    def _run_linear_tet_fea(
        self,
        study: StudyData,
        segmentation: SegmentationResult,
        material_result: MaterialResult,
        brace: BraceModel,
        febio_setup: FEBioSetup,
        output_dir: Path,
        mode: str = "linear_tet_fea",
    ) -> SimulationResult:
        mesh = material_result.mesh.copy(deep=True)
        fea = solve_mesh_linear_tet_fea(mesh, febio_setup, self.config)
        mesh.point_data["displacement"] = fea.displacement
        derived = self._attach_derived_fields(mesh, fea.von_mises, fea.principal_strain)

        mesh_path = output_dir / "simulation_mesh.vtu"
        mesh.save(mesh_path)

        peak_phase = febio_setup.load_summary.get("peak_phase", {}) if febio_setup.load_summary else {}
        summary = {
            "mode": mode,
            **derived,
            "governing_phase": str(peak_phase.get("name", "peak_load_case")),
            "engine": "linear_tetrahedron_fea",
            "fea_solver": fea.solver,
            "fea_residual_norm": float(fea.residual_norm),
            "fea_iterations": int(fea.iterations),
            "brace_enabled": bool(getattr(brace, "enabled", False)),
        }
        if study is not None and segmentation is not None:
            summary["defect_slice_index"] = int(
                locate_pseudarthrosis_slice(segmentation.mask, study.hu_volume)
            )
        summary.update(fea.stats)

        summary_path = output_dir / "simulation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return SimulationResult(
            mode=mode,
            mesh=mesh,
            mesh_path=mesh_path,
            summary=summary,
            log_path=summary_path,
        )

    def _allow_internal_fea(self) -> bool:
        sim_cfg = self.config.get("simulation", {})
        if "internal_fea_if_febio_unavailable" in sim_cfg:
            return bool(sim_cfg.get("internal_fea_if_febio_unavailable"))
        # Legacy key: a false value used to disable any non-FEBio path.
        if "surrogate_if_febio_unavailable" in sim_cfg:
            return bool(sim_cfg.get("surrogate_if_febio_unavailable"))
        return True

    def run(
        self,
        febio_setup: FEBioSetup,
        study: StudyData,
        segmentation: SegmentationResult,
        material_result: MaterialResult,
        brace: BraceModel,
        output_dir: Path,
    ) -> SimulationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        executable = self._resolve_febio_executable()
        allow_internal_fea = self._allow_internal_fea()
        should_try_febio = bool(self.config["simulation"].get("prefer_febio", True)) and executable

        if should_try_febio:
            command = self._febio_command(executable, febio_setup, output_dir)
            completed = subprocess.run(
                command,
                cwd=output_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            log_path = output_dir / "febio_stdout.log"
            log_path.write_text(
                (completed.stdout or "") + "\n\nSTDERR\n" + (completed.stderr or ""),
                encoding="utf-8",
            )
            if completed.returncode == 0:
                try:
                    febio_result = self._load_febio_results(febio_setup, material_result, output_dir)
                except Exception as exc:
                    if not allow_internal_fea:
                        raise RuntimeError(
                            "FEBio completed, but its result files could not be imported. See "
                            f"{log_path} for execution details."
                        ) from exc
                    fea_result = self._run_linear_tet_fea(
                        study,
                        segmentation,
                        material_result,
                        brace,
                        febio_setup,
                        output_dir,
                        mode="linear_tet_fea_after_febio_import_failure",
                    )
                    summary = dict(fea_result.summary)
                    summary.update(
                        {
                            "mode": "linear_tet_fea_after_febio_import_failure",
                            "febio_return_code": int(completed.returncode),
                            "febio_command": command,
                            "febio_log_path": str(log_path),
                            "febio_import_error": str(exc),
                        }
                    )
                    (output_dir / "simulation_summary.json").write_text(
                        json.dumps(summary, indent=2), encoding="utf-8"
                    )
                    fea_result.mode = "linear_tet_fea_after_febio_import_failure"
                    fea_result.summary = summary
                    fea_result.log_path = log_path
                    fea_result.raw_stdout = completed.stdout
                    return fea_result

                febio_result.summary.update(
                    {
                        "mode": "febio_results_vtk",
                        "engine": "febio",
                        "febio_return_code": int(completed.returncode),
                        "febio_command": command,
                        "febio_log_path": str(log_path),
                    }
                )
                (output_dir / "simulation_summary.json").write_text(
                    json.dumps(febio_result.summary, indent=2), encoding="utf-8"
                )
                febio_result.log_path = log_path
                febio_result.raw_stdout = completed.stdout
                return febio_result

            if not allow_internal_fea:
                raise RuntimeError(f"FEBio execution failed. See {log_path} for details.")

            fea_result = self._run_linear_tet_fea(
                study,
                segmentation,
                material_result,
                brace,
                febio_setup,
                output_dir,
                mode="linear_tet_fea_after_febio_failure",
            )
            summary = dict(fea_result.summary)
            summary.update(
                {
                    "mode": "linear_tet_fea_after_febio_failure",
                    "febio_return_code": int(completed.returncode),
                    "febio_command": command,
                    "febio_log_path": str(log_path),
                }
            )
            (output_dir / "simulation_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            fea_result.mode = "linear_tet_fea_after_febio_failure"
            fea_result.summary = summary
            fea_result.log_path = log_path
            fea_result.raw_stdout = completed.stdout
            return fea_result

        if bool(self.config["simulation"].get("prefer_febio", True)) and not allow_internal_fea:
            raise RuntimeError("FEBio was requested, but no FEBio executable was found.")

        return self._run_linear_tet_fea(
            study, segmentation, material_result, brace, febio_setup, output_dir, mode="linear_tet_fea"
        )

