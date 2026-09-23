from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .models import RiskAssessment, SimulationResult, to_jsonable


class RiskAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _categorize_risk(self, min_safety_factor: float, years_to_failure: float) -> str:
        if min_safety_factor < 1.0 or years_to_failure < 0.25:
            return "high"
        if min_safety_factor < 1.5 or years_to_failure < 1.0:
            return "elevated"
        if min_safety_factor < 2.0 or years_to_failure < 3.0:
            return "moderate"
        return "lower"

    def _recommendations(self, category: str, brace_mode: str) -> List[str]:
        recommendations = [
            "Review segmentation, loads, and brace alignment with an orthopaedic specialist before acting on the estimate.",
            "Treat the output as a biomechanical decision-support scenario, not a clinical diagnosis.",
        ]
        if category in {"high", "elevated"}:
            recommendations.append("Consider reducing high-impact activity volume until subject-specific validation is complete.")
            recommendations.append("Request a patient-specific review of brace fit and mediolateral support coverage.")
        if brace_mode == "proxy":
            recommendations.append("Replace the proxy brace with the real STL geometry for more credible support estimates.")
        if category == "lower":
            recommendations.append("Monitor interval CT or radiographs if symptoms or activity level change.")
        return recommendations

    def analyze(self, simulation: SimulationResult, output_dir: Path, brace_mode: str) -> RiskAssessment:
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh = simulation.mesh
        stress = np.asarray(mesh.cell_data.get("von_mises_mpa", np.zeros(mesh.n_cells)), dtype=float)
        strain = np.asarray(mesh.cell_data.get("principal_strain", np.zeros(mesh.n_cells)), dtype=float)
        safety = np.asarray(mesh.cell_data.get("safety_factor", np.ones(mesh.n_cells)), dtype=float)
        cycles = np.asarray(mesh.cell_data.get("fatigue_cycles", np.full(mesh.n_cells, np.inf)), dtype=float)

        from .risk_map import VISIBLE_UTILIZATION_HEALTHY, attach_clinical_risk_fields, clinical_summary

        if "clinical_safety_factor" not in getattr(mesh, "cell_data", {}):
            attach_clinical_risk_fields(mesh)
        extra = clinical_summary(mesh)
        clinical_sf = float(extra["min_clinical_safety_factor"])
        max_util = float(extra["max_clinical_utilization"])
        has_defect = bool(extra["has_structural_defect"] or simulation.summary.get("has_structural_defect"))
        years = float(simulation.summary.get("years_to_failure_estimate", extra["min_clinical_fatigue_cycles"]))
        if np.isfinite(extra["min_clinical_fatigue_cycles"]):
            steps_per_day = max(1.0, float(self.config.get("patient", {}).get("steps_per_day", 6000)))
            years = float(extra["min_clinical_fatigue_cycles"] / (steps_per_day * 365.0))

        if (not has_defect) and float(extra.get("yielded_volume_fraction", 0.0)) < 0.05:
            category = "lower"
            years = float("inf")
            statement = (
                "No localized pseudarthrosis-like weak zone was detected. Isolated hot voxels on otherwise normal bone are not treated as fracture risk."
            )
        elif not has_defect and max_util < VISIBLE_UTILIZATION_HEALTHY:
            category = "lower"
            years = float("inf")
            statement = (
                "No localized pseudarthrosis-like weak zone was detected, and cortical bone did not approach yield "
                "under the modeled load."
            )
        else:
            category = self._categorize_risk(clinical_sf, years)
            if np.isfinite(years):
                statement = (
                    f"Estimated fatigue failure horizon at the highlighted weak tissue: {years:.2f} years "
                    "under the configured activity assumptions."
                )
            else:
                statement = "Fatigue failure was not reached within the configured model horizon at the clinically relevant tissue."

        hotspot_cutoff = float(np.quantile(stress, 0.95)) if len(stress) else 0.0
        hotspot_count = int(np.sum(stress >= hotspot_cutoff)) if len(stress) else 0
        min_sf = clinical_sf if np.isfinite(clinical_sf) else float("inf")

        summary = {
            "risk_category": category,
            "max_von_mises_mpa": float(np.max(stress)) if len(stress) else 0.0,
            "mean_von_mises_mpa": float(np.mean(stress)) if len(stress) else 0.0,
            "max_principal_strain": float(np.max(strain)) if len(strain) else 0.0,
            "min_safety_factor": min_sf,
            "min_fatigue_cycles": float(extra["min_clinical_fatigue_cycles"]),
            "years_to_failure_estimate": years,
            "fracture_likely_statement": statement,
            "hotspot_cell_count": hotspot_count,
            "governing_phase": simulation.summary.get("governing_phase", "unknown"),
            "simulation_mode": simulation.mode,
            "has_structural_defect": has_defect,
            "max_clinical_utilization": max_util,
            "defect_reason": extra["defect_reason"],
            "defect_z_mm": extra["defect_z_mm"],
            "overlay_caption": extra["overlay_caption"],
        }
        summary["yielded_volume_fraction"] = float(extra.get("yielded_volume_fraction", 0.0))
        recommendations = self._recommendations(category, brace_mode)
        if not has_defect and category == "lower":
            recommendations.append(
                "Maps stay dark on healthy bone on purpose: color is reserved for abnormal or near-failure tissue."
            )
        elif has_defect:
            recommendations.append(
                "The highlighted band is the detected weak zone. Values near 1.0 mean that site is at yield under the modeled load."
            )

        summary_path = output_dir / "risk_summary.json"
        summary_path.write_text(
            json.dumps(to_jsonable({"summary": summary, "recommendations": recommendations}), indent=2),
            encoding="utf-8",
        )
        return RiskAssessment(summary=summary, recommendations=recommendations, summary_path=summary_path)
