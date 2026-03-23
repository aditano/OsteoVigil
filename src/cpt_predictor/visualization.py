from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .models import RiskAssessment, SimulationResult


class ResultVisualizer:
    def __init__(self, config: Dict):
        self.config = config

    def _save_dashboard(self, simulation: SimulationResult, risk: RiskAssessment, output_dir: Path) -> Path:
        mesh = simulation.mesh
        stress = np.asarray(mesh.cell_data.get("von_mises_mpa", np.zeros(mesh.n_cells)))
        safety = np.asarray(mesh.cell_data.get("safety_factor", np.ones(mesh.n_cells)))
        strain = np.asarray(mesh.cell_data.get("principal_strain", np.zeros(mesh.n_cells)))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].hist(stress, bins=30, color="#c0392b", alpha=0.9)
        axes[0].set_title("Von Mises Stress")
        axes[0].set_xlabel("MPa")

        axes[1].hist(safety, bins=30, color="#27ae60", alpha=0.9)
        axes[1].set_title("Safety Factor")
        axes[1].set_xlabel("Safety Factor")

        axes[2].scatter(stress, strain, s=10, alpha=0.4, color="#2980b9")
        axes[2].set_title(f"Risk: {risk.summary['risk_category']}")
        axes[2].set_xlabel("Stress (MPa)")
        axes[2].set_ylabel("Principal Strain")

        fig.tight_layout()
        dashboard_path = output_dir / "risk_dashboard.png"
        fig.savefig(dashboard_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return dashboard_path

    def _save_pyvista_plot(self, simulation: SimulationResult, output_dir: Path) -> Dict[str, str]:
        try:
            import pyvista as pv

            pv.start_xvfb()
            mesh = simulation.mesh
            screenshot_path = output_dir / "stress_map.png"
            html_path = output_dir / "interactive_mesh.html"

            plotter = pv.Plotter(off_screen=True)
            plotter.set_background("white")
            plotter.add_mesh(
                mesh,
                scalars="safety_factor",
                cmap="RdYlGn",
                clim=[0.0, max(2.5, float(np.nanmax(mesh.cell_data.get("safety_factor", [1.0]))))],
                show_edges=False,
                scalar_bar_args={"title": "Safety Factor"},
            )
            plotter.add_text("OsteoVigil CPT Risk Map", position="upper_left", font_size=12, color="black")
            plotter.show(screenshot=str(screenshot_path), auto_close=False)

            try:
                plotter.export_html(str(html_path))
                exported_html = str(html_path)
            except Exception:
                exported_html = ""
            plotter.close()
            return {"stress_map": str(screenshot_path), "interactive_html": exported_html}
        except Exception:
            return {}

    def create_outputs(self, simulation: SimulationResult, risk: RiskAssessment, output_dir: Path) -> Dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        outputs["risk_dashboard"] = str(self._save_dashboard(simulation, risk, output_dir))
        outputs.update(self._save_pyvista_plot(simulation, output_dir))
        return {key: value for key, value in outputs.items() if value}

