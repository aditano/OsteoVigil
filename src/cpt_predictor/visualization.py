from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from .models import RiskAssessment, SimulationResult


class ResultVisualizer:
    def __init__(self, config: Dict):
        self.config = config

    @staticmethod
    def _should_disable_pyvista_windowing() -> bool:
        if sys.platform != "darwin":
            return False

        flag = os.environ.get("OSTEOVIGIL_DISABLE_PYVISTA_WINDOWING", "").strip().lower()
        if flag in {"1", "true", "yes", "on"}:
            return True

        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
        except Exception:
            return False

        return get_script_run_ctx() is not None

    @staticmethod
    def _get_stress_values(mesh) -> np.ndarray:
        stress = np.asarray(mesh.cell_data.get("von_mises_mpa", []), dtype=float)
        if stress.size == mesh.n_cells:
            return stress
        point_stress = np.asarray(mesh.point_data.get("von_mises_mpa", []), dtype=float)
        if point_stress.size == mesh.n_points:
            return point_stress
        if mesh.n_cells:
            return np.zeros(mesh.n_cells, dtype=float)
        return np.zeros(1, dtype=float)

    @staticmethod
    def _build_projection_image(x: np.ndarray, y: np.ndarray, stress: np.ndarray) -> tuple[np.ndarray | None, tuple[float, float, float, float]]:
        x_pad = max(float(np.ptp(x)) * 0.05, 1.0)
        y_pad = max(float(np.ptp(y)) * 0.05, 1.0)
        extent = (
            float(np.min(x)) - x_pad,
            float(np.max(x)) + x_pad,
            float(np.min(y)) - y_pad,
            float(np.max(y)) + y_pad,
        )

        try:
            grid_x, grid_y = np.mgrid[
                extent[0] : extent[1] : 200j,
                extent[2] : extent[3] : 200j,
            ]
            projected = np.column_stack([x, y])
            grid_linear = griddata(projected, stress, (grid_x, grid_y), method="linear")
            grid_nearest = griddata(projected, stress, (grid_x, grid_y), method="nearest")
            heatmap = np.where(np.isnan(grid_linear), grid_nearest, grid_linear)
            return heatmap, (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
        except Exception:
            return None, extent

    def _plot_projection(
        self,
        ax,
        *,
        centers: np.ndarray,
        stress: np.ndarray,
        hotspot_index: int,
        axis_a: int,
        axis_b: int,
        title: str,
        x_label: str,
        y_label: str,
        vmin: float,
        vmax: float,
        show_legend: bool,
    ):
        projected = centers[:, [axis_a, axis_b]]
        x = projected[:, 0]
        y = projected[:, 1]
        hotspot_point = projected[hotspot_index]

        heatmap, extent = self._build_projection_image(x, y, stress)
        if heatmap is not None:
            image = ax.imshow(
                heatmap.T,
                origin="lower",
                extent=extent,
                cmap="inferno",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            image = ax.scatter(
                x,
                y,
                c=stress,
                cmap="inferno",
                s=26,
                alpha=0.8,
                edgecolors="none",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_facecolor("#140f0a")

        ax.scatter(
            [hotspot_point[0]],
            [hotspot_point[1]],
            s=140,
            facecolors="none",
            edgecolors="cyan",
            linewidths=2.0,
            marker="o",
            label="Peak stress",
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if show_legend:
            ax.legend(loc="upper right", frameon=True)
        return image

    def _save_hotspot_heatmap(self, simulation: SimulationResult, output_dir: Path) -> Path:
        mesh = simulation.mesh
        centers = np.asarray(mesh.cell_centers().points if mesh.n_cells else np.zeros((0, 3)), dtype=float)
        stress = self._get_stress_values(mesh)

        if centers.ndim != 2 or centers.shape[0] == 0:
            heatmap_path = output_dir / "stress_heatmap_2d.png"
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No stress data available", ha="center", va="center", fontsize=14)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return heatmap_path

        if stress.size != centers.shape[0]:
            if stress.size > 0:
                stress = np.resize(stress, centers.shape[0])
            else:
                stress = np.zeros(centers.shape[0], dtype=float)

        finite_mask = np.isfinite(stress)
        if not finite_mask.any():
            stress = np.zeros_like(stress)
            finite_mask = np.ones_like(stress, dtype=bool)
        hotspot_index = int(np.nanargmax(np.where(finite_mask, stress, -np.inf)))

        finite_stress = stress[finite_mask]
        vmin = float(np.min(finite_stress))
        vmax = float(np.max(finite_stress))
        if vmax <= vmin:
            vmax = vmin + 1.0

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5))
        projection_specs = (
            ("AP View", 0, 2, "X position", "Z position"),
            ("Lateral View", 1, 2, "Y position", "Z position"),
        )
        image = None
        for index, (title, axis_a, axis_b, x_label, y_label) in enumerate(projection_specs):
            image = self._plot_projection(
                axes[index],
                centers=centers,
                stress=stress,
                hotspot_index=hotspot_index,
                axis_a=axis_a,
                axis_b=axis_b,
                title=title,
                x_label=x_label,
                y_label=y_label,
                vmin=vmin,
                vmax=vmax,
                show_legend=index == 0,
            )

        fig.suptitle("2D Stress Hotspot Heatmaps", fontsize=14)
        fig.subplots_adjust(top=0.88, right=0.88, wspace=0.24)
        cbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.04)
        cbar.set_label("Von Mises stress (MPa)")
        heatmap_path = output_dir / "stress_heatmap_2d.png"
        fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return heatmap_path

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
            if self._should_disable_pyvista_windowing():
                placeholder_path = output_dir / "stress_map_disabled.png"
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.axis("off")
                ax.text(
                    0.5,
                    0.62,
                    "3D VTK preview disabled on macOS Streamlit",
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                )
                ax.text(
                    0.5,
                    0.42,
                    "PyVista rendering is skipped here to avoid AppKit window creation\n"
                    "from a background thread. The 2D stress map and report still render normally.",
                    ha="center",
                    va="center",
                    fontsize=11,
                )
                fig.tight_layout()
                fig.savefig(placeholder_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
                return {"stress_map": str(placeholder_path), "interactive_html": ""}

            import pyvista as pv

            if hasattr(pv, "system_supports_plotting") and not pv.system_supports_plotting():
                return {}
            if sys.platform.startswith("linux"):
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
        outputs["stress_heatmap_2d"] = str(self._save_hotspot_heatmap(simulation, output_dir))
        outputs.update(self._save_pyvista_plot(simulation, output_dir))
        return {key: value for key, value in outputs.items() if value}
