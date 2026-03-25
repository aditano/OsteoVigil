from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .models import BraceModel, FEBioSetup, MaterialResult, MeshResult, SegmentationResult, SimulationResult, StudyData
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

    def _slice_area_profile(self, segmentation: SegmentationResult, study: StudyData) -> np.ndarray:
        return segmentation.mask.sum(axis=(1, 2)).astype(float) * study.spacing_zyx[1] * study.spacing_zyx[2]

    def _run_surrogate(
        self,
        study: StudyData,
        segmentation: SegmentationResult,
        material_result: MaterialResult,
        brace: BraceModel,
        output_dir: Path,
    ) -> SimulationResult:
        mesh = material_result.mesh.copy(deep=True)
        centers = np.asarray(mesh.cell_centers().points)
        modulus = np.asarray(mesh.cell_data["youngs_modulus_mpa"], dtype=float)
        strength = np.asarray(mesh.cell_data["yield_strength_mpa"], dtype=float)

        slice_area = self._slice_area_profile(segmentation, study)
        nonzero = np.where(slice_area > 1e-3)[0]
        if len(nonzero):
            defect_slice = int(nonzero[np.argmin(slice_area[nonzero])])
            min_area = max(1.0, float(slice_area[defect_slice]))
        else:
            defect_slice = 0
            min_area = 100.0

        defect_z = study.origin_xyz[2] + defect_slice * study.spacing_zyx[0]
        z_positions = study.origin_xyz[2] + np.arange(study.hu_volume.shape[0]) * study.spacing_zyx[0]
        area_profile = np.maximum(slice_area, min_area)
        radius_profile = np.sqrt(area_profile / np.pi)

        x_center = 0.5 * (float(mesh.bounds[0]) + float(mesh.bounds[1]))
        y_center = 0.5 * (float(mesh.bounds[2]) + float(mesh.bounds[3]))
        radial_distance = np.sqrt((centers[:, 0] - x_center) ** 2 + (centers[:, 1] - y_center) ** 2) + 1e-3

        body_mass = float(self.config["patient"]["body_mass_kg"])
        fatigue_constant = float(self.config["simulation"].get("fatigue_constant", 500000.0))
        fatigue_exponent = float(self.config["simulation"].get("fatigue_exponent", 7.5))
        brace_factor = float(brace.metadata.get("stress_reduction_factor", 0.72)) if brace.enabled else 1.0

        phase_names = []
        phase_stresses = []
        phase_strains = []
        phase_cycles = []

        local_area = np.interp(centers[:, 2], z_positions, area_profile)
        local_radius = np.interp(centers[:, 2], z_positions, radius_profile)
        section_modulus = np.maximum(np.pi * np.power(local_radius, 3) / 4.0, 1.0)
        polar_moment = np.maximum(np.pi * np.power(local_radius, 4) / 2.0, 1.0)

        brace_mask = np.ones(mesh.n_cells, dtype=float)
        if brace.enabled and brace.support_bounds_xyz:
            x0, x1, y0, y1, z0, z1 = brace.support_bounds_xyz
            in_brace = (
                (centers[:, 0] >= x0)
                & (centers[:, 0] <= x1)
                & (centers[:, 1] >= y0)
                & (centers[:, 1] <= y1)
                & (centers[:, 2] >= z0)
                & (centers[:, 2] <= z1)
            )
            brace_mask[in_brace] = brace_factor

        defect_sigma = max(study.spacing_zyx[0] * 3.0, 6.0)
        defect_factor = 1.0 + 1.6 * np.exp(-np.square(centers[:, 2] - defect_z) / (2.0 * defect_sigma**2))

        for phase in self.config["loads"]["gait_phases"]:
            phase_names.append(phase["name"])
            axial_force = body_mass * 9.81 * float(phase["axial_bodyweight_multiplier"])
            bending_moment = float(phase["bending_moment_nm"]) * 1000.0
            torsion = float(phase["torsion_nm"]) * 1000.0

            sigma_axial = axial_force / np.maximum(local_area, 1.0)
            sigma_bending = (bending_moment * radial_distance) / section_modulus
            tau_torsion = (torsion * radial_distance) / polar_moment
            stress = np.sqrt(np.square(sigma_axial + sigma_bending) + (3.0 * np.square(tau_torsion)))
            stress = stress * defect_factor * brace_mask

            strain = stress / np.maximum(modulus, 100.0)
            cycles = fatigue_constant * np.power(np.maximum(strength / np.maximum(stress, 0.1), 0.1), fatigue_exponent)

            phase_stresses.append(stress)
            phase_strains.append(strain)
            phase_cycles.append(cycles)

        stacked_stress = np.vstack(phase_stresses)
        stacked_strain = np.vstack(phase_strains)
        stacked_cycles = np.vstack(phase_cycles)
        governing_index = np.argmax(stacked_stress, axis=0)
        cell_ids = np.arange(mesh.n_cells)

        von_mises = stacked_stress[governing_index, cell_ids]
        max_principal_strain = stacked_strain[governing_index, cell_ids]
        fatigue_cycles = stacked_cycles[governing_index, cell_ids]
        safety_factor = strength / np.maximum(von_mises, 0.1)

        mesh.cell_data["von_mises_mpa"] = von_mises
        mesh.cell_data["principal_strain"] = max_principal_strain
        mesh.cell_data["safety_factor"] = safety_factor
        mesh.cell_data["fatigue_cycles"] = fatigue_cycles
        mesh.cell_data["governing_phase_index"] = governing_index.astype(int)

        mesh_path = output_dir / "simulation_mesh.vtu"
        mesh.save(mesh_path)

        steps_per_day = max(1.0, float(self.config["patient"].get("steps_per_day", 6000)))
        years_to_failure = float(np.min(fatigue_cycles) / (steps_per_day * 365.0))
        summary = {
            "mode": "surrogate",
            "max_von_mises_mpa": float(np.max(von_mises)),
            "min_safety_factor": float(np.min(safety_factor)),
            "min_fatigue_cycles": float(np.min(fatigue_cycles)),
            "years_to_failure_estimate": max(0.0, years_to_failure),
            "defect_slice_index": defect_slice,
            "governing_phase": phase_names[int(np.argmax([np.max(v) for v in phase_stresses]))],
        }

        summary_path = output_dir / "simulation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return SimulationResult(mode="surrogate", mesh=mesh, mesh_path=mesh_path, summary=summary, log_path=summary_path)

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
        allow_surrogate_fallback = bool(self.config["simulation"].get("surrogate_if_febio_unavailable", True))
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
                surrogate_result = self._run_surrogate(study, segmentation, material_result, brace, output_dir)
                summary = dict(surrogate_result.summary)
                summary.update(
                    {
                        "mode": "febio_completed_surrogate_postprocess",
                        "febio_return_code": int(completed.returncode),
                        "febio_command": command,
                    }
                )
                return SimulationResult(
                    mode="febio_completed_surrogate_postprocess",
                    mesh=surrogate_result.mesh,
                    mesh_path=surrogate_result.mesh_path,
                    summary=summary,
                    log_path=log_path,
                    raw_stdout=completed.stdout,
                )

            if not allow_surrogate_fallback:
                raise RuntimeError(
                    "FEBio execution failed. See "
                    f"{log_path}"
                    " for details."
                )

            surrogate_result = self._run_surrogate(study, segmentation, material_result, brace, output_dir)
            summary = dict(surrogate_result.summary)
            summary.update(
                {
                    "mode": "surrogate_febio_failed",
                    "febio_return_code": int(completed.returncode),
                    "febio_command": command,
                    "febio_log_path": str(log_path),
                }
            )
            return SimulationResult(
                mode="surrogate_febio_failed",
                mesh=surrogate_result.mesh,
                mesh_path=surrogate_result.mesh_path,
                summary=summary,
                log_path=log_path,
                raw_stdout=completed.stdout,
            )

        elif bool(self.config["simulation"].get("prefer_febio", True)) and not allow_surrogate_fallback:
            raise RuntimeError("FEBio was requested, but no FEBio executable was found.")

        surrogate_result = self._run_surrogate(study, segmentation, material_result, brace, output_dir)
        if bool(self.config["simulation"].get("prefer_febio", True)) and not executable:
            summary = dict(surrogate_result.summary)
            summary.update({"mode": "surrogate_no_febio"})
            return SimulationResult(
                mode="surrogate_no_febio",
                mesh=surrogate_result.mesh,
                mesh_path=surrogate_result.mesh_path,
                summary=summary,
                log_path=surrogate_result.log_path,
                raw_stdout=surrogate_result.raw_stdout,
            )
        return surrogate_result
