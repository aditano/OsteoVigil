from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .models import RiskAssessment, SimulationResult


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

        hotspot_cutoff = float(np.quantile(stress, 0.95)) if len(stress) else 0.0
        hotspot_count = int(np.sum(stress >= hotspot_cutoff)) if len(stress) else 0
        years = float(simulation.summary.get("years_to_failure_estimate", np.inf))
        category = self._categorize_risk(float(np.min(safety)), years)

        summary = {
            "risk_category": category,
            "max_von_mises_mpa": float(np.max(stress)) if len(stress) else 0.0,
            "mean_von_mises_mpa": float(np.mean(stress)) if len(stress) else 0.0,
            "max_principal_strain": float(np.max(strain)) if len(strain) else 0.0,
            "min_safety_factor": float(np.min(safety)) if len(safety) else 0.0,
            "min_fatigue_cycles": float(np.min(cycles)) if len(cycles) else float("inf"),
            "years_to_failure_estimate": years,
            "fracture_likely_statement": f"Fracture likely in {years:.2f} years under the configured activity assumptions."
            if np.isfinite(years)
            else "Fatigue failure was not reached within the configured surrogate model horizon.",
            "hotspot_cell_count": hotspot_count,
            "governing_phase": simulation.summary.get("governing_phase", "unknown"),
            "simulation_mode": simulation.mode,
        }
        recommendations = self._recommendations(category, brace_mode)

        summary_path = output_dir / "risk_summary.json"
        summary_path.write_text(
            json.dumps({"summary": summary, "recommendations": recommendations}, indent=2),
            encoding="utf-8",
        )
        return RiskAssessment(summary=summary, recommendations=recommendations, summary_path=summary_path)

