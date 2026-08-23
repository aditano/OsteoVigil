from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter

from .models import RiskAssessment, SegmentationResult, SimulationResult, StudyData


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
    def _cell_data(mesh) -> Dict:
        return getattr(mesh, "cell_data", None) or getattr(mesh, "cell_data", {})

    @staticmethod
    def _point_data(mesh) -> Dict:
        return getattr(mesh, "point_data", None) or getattr(mesh, "point_data", {})

    @staticmethod
    def select_interior_extrema(
        centers: np.ndarray,
        stress: np.ndarray,
        safety: Optional[np.ndarray] = None,
        modulus: Optional[np.ndarray] = None,
        margin: float = 0.12,
        min_modulus_mpa: float = 1000.0,
    ) -> Tuple[int, int]:
        """Pick peak-stress and lowest-SF cells away from proximal/distal load bands."""
        centers = np.asarray(centers, dtype=float)
        stress = np.asarray(stress, dtype=float)
        n = int(centers.shape[0])
        if n == 0:
            return 0, 0
        z = centers[:, 2]
        span = max(float(np.max(z) - np.min(z)), 1e-6)
        zrel = (z - float(np.min(z))) / span
        interior = (zrel >= float(margin)) & (zrel <= 1.0 - float(margin))
        eligible = interior & np.isfinite(stress)
        if not eligible.any():
            eligible = np.isfinite(stress)
        if modulus is not None:
            modulus = np.asarray(modulus, dtype=float)
            if modulus.size == n:
                cortical = eligible & np.isfinite(modulus) & (modulus >= float(min_modulus_mpa))
                if cortical.any():
                    eligible = cortical
        hotspot = int(np.nanargmax(np.where(eligible, stress, -np.inf)))
        if safety is None:
            return hotspot, hotspot
        safety = np.asarray(safety, dtype=float)
        if safety.size != n or not np.isfinite(safety[eligible]).any():
            return hotspot, hotspot
        weakest = int(np.nanargmin(np.where(eligible & np.isfinite(safety), safety, np.inf)))
        return hotspot, weakest

    def _get_stress_values(self, mesh) -> np.ndarray:
        cell_data = self._cell_data(mesh)
        stress = np.asarray(cell_data.get("von_mises_mpa", []), dtype=float)
        if stress.size == getattr(mesh, "n_cells", 0):
            return stress
        point_data = self._point_data(mesh)
        point_stress = np.asarray(point_data.get("von_mises_mpa", []), dtype=float)
        if point_stress.size == getattr(mesh, "n_points", 0):
            return point_stress
        if getattr(mesh, "n_cells", 0):
            return np.zeros(mesh.n_cells, dtype=float)
        return np.zeros(1, dtype=float)

    @staticmethod
    def _world_to_index(points: np.ndarray, origin_xyz, spacing_zyx, shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        origin = np.asarray(origin_xyz, dtype=float)
        spacing = np.asarray(spacing_zyx, dtype=float)
        iz = np.rint((points[:, 2] - origin[2]) / max(spacing[0], 1e-6)).astype(int)
        iy = np.rint((points[:, 1] - origin[1]) / max(spacing[1], 1e-6)).astype(int)
        ix = np.rint((points[:, 0] - origin[0]) / max(spacing[2], 1e-6)).astype(int)
        valid = (
            (iz >= 0) & (iz < shape[0])
            & (iy >= 0) & (iy < shape[1])
            & (ix >= 0) & (ix < shape[2])
        )
        return iz[valid], iy[valid], ix[valid], valid

    def project_bone_stress_views(
        self,
        centers: np.ndarray,
        stress: np.ndarray,
        study: Optional[StudyData] = None,
        segmentation: Optional[SegmentationResult] = None,
    ) -> Dict[str, np.ndarray]:
        """Return AP and lateral maximum-intensity projections of bone stress."""
        stress = np.asarray(stress, dtype=float)
        centers = np.asarray(centers, dtype=float)
        finite = np.isfinite(stress)
        if not finite.any():
            stress = np.zeros_like(stress)
            finite = np.ones_like(stress, dtype=bool)
        stress = np.where(finite, stress, np.nan)

        mask = None
        if segmentation is not None and getattr(segmentation, "mask", None) is not None:
            mask = np.asarray(segmentation.mask, dtype=bool)

        if study is not None and mask is not None and mask.ndim == 3:
            return self._mip_from_volume(centers, stress, study, mask)
        return self._mip_from_points(centers, stress)

    def _mip_from_volume(
        self,
        centers: np.ndarray,
        stress: np.ndarray,
        study: StudyData,
        mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        shape = mask.shape
        volume = np.full(shape, np.nan, dtype=np.float32)
        iz, iy, ix, valid = self._world_to_index(centers, study.origin_xyz, study.spacing_zyx, shape)
        valid_stress = np.asarray(stress[valid], dtype=np.float32)
        finite = np.isfinite(valid_stress)
        iz, iy, ix, valid_stress = iz[finite], iy[finite], ix[finite], valid_stress[finite]
        if valid_stress.size:
            flat = np.full(mask.size, -np.inf, dtype=np.float32)
            np.maximum.at(flat, np.ravel_multi_index((iz, iy, ix), shape), valid_stress)
            volume = flat.reshape(shape)
            volume[volume == -np.inf] = np.nan

        filled = np.nan_to_num(volume, nan=0.0)
        for _ in range(3):
            dilated = maximum_filter(filled, size=3)
            empty = ~np.isfinite(volume) & mask
            volume[empty] = np.where(dilated[empty] > 0, dilated[empty], np.nan)
            filled = np.nan_to_num(volume, nan=0.0)
        volume[~mask] = np.nan

        with np.errstate(all="ignore"):
            ap = np.nanmax(np.where(np.isfinite(volume), volume, -np.inf), axis=1)
            lateral = np.nanmax(np.where(np.isfinite(volume), volume, -np.inf), axis=2)
            hu = np.asarray(study.hu_volume, dtype=float)
            hu = np.where(mask, hu, -np.inf)
            ap_hu = np.max(hu, axis=1)
            lat_hu = np.max(hu, axis=2)
        ap[ap == -np.inf] = np.nan
        lateral[lateral == -np.inf] = np.nan
        ap_hu[ap_hu == -np.inf] = np.nan
        lat_hu[lat_hu == -np.inf] = np.nan
        ap_sil = np.any(mask, axis=1)
        lat_sil = np.any(mask, axis=2)
        ap[~ap_sil] = np.nan
        lateral[~lat_sil] = np.nan
        ap_hu[~ap_sil] = np.nan
        lat_hu[~lat_sil] = np.nan

        origin = np.asarray(study.origin_xyz, dtype=float)
        spacing = np.asarray(study.spacing_zyx, dtype=float)
        ap_extent = (
            float(origin[0]),
            float(origin[0] + shape[2] * spacing[2]),
            float(origin[2]),
            float(origin[2] + shape[0] * spacing[0]),
        )
        lat_extent = (
            float(origin[1]),
            float(origin[1] + shape[1] * spacing[1]),
            float(origin[2]),
            float(origin[2] + shape[0] * spacing[0]),
        )
        return {
            "ap": ap,
            "lateral": lateral,
            "ap_hu": ap_hu,
            "lateral_hu": lat_hu,
            "ap_extent": np.asarray(ap_extent, dtype=float),
            "lateral_extent": np.asarray(lat_extent, dtype=float),
        }

    def _mip_from_points(self, centers: np.ndarray, stress: np.ndarray, bins: int = 220) -> Dict[str, np.ndarray]:
        def _project(horizontal: np.ndarray, vertical: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
            h0, h1 = float(np.min(horizontal)), float(np.max(horizontal))
            v0, v1 = float(np.min(vertical)), float(np.max(vertical))
            if h1 <= h0:
                h1 = h0 + 1.0
            if v1 <= v0:
                v1 = v0 + 1.0
            pad_h = 0.04 * (h1 - h0)
            pad_v = 0.04 * (v1 - v0)
            h0, h1, v0, v1 = h0 - pad_h, h1 + pad_h, v0 - pad_v, v1 + pad_v
            n_h = bins
            n_v = max(bins, int(bins * (v1 - v0) / max(h1 - h0, 1e-6)))
            ih = np.clip(((horizontal - h0) / (h1 - h0) * (n_h - 1)).astype(int), 0, n_h - 1)
            iv = np.clip(((vertical - v0) / (v1 - v0) * (n_v - 1)).astype(int), 0, n_v - 1)
            image = np.full(n_v * n_h, -np.inf, dtype=float)
            occ = np.zeros(n_v * n_h, dtype=np.uint8)
            finite = np.isfinite(stress)
            np.maximum.at(image, iv[finite] * n_h + ih[finite], stress[finite])
            np.add.at(occ, iv[finite] * n_h + ih[finite], 1)
            image = image.reshape(n_v, n_h)
            occ = occ.reshape(n_v, n_h) > 0
            occ = binary_dilation(occ, iterations=1)
            image[~occ] = np.nan
            image[image == -np.inf] = np.nan
            return image, (h0, h1, v0, v1)

        ap, ap_extent = _project(centers[:, 0], centers[:, 2])
        lateral, lat_extent = _project(centers[:, 1], centers[:, 2])
        return {
            "ap": ap,
            "lateral": lateral,
            "ap_hu": np.full_like(ap, np.nan),
            "lateral_hu": np.full_like(lateral, np.nan),
            "ap_extent": np.asarray(ap_extent, dtype=float),
            "lateral_extent": np.asarray(lat_extent, dtype=float),
        }

    def _plot_bone_view(self, ax, image: np.ndarray, hu: np.ndarray, extent, title: str, x_label: str, vmin: float, vmax: float, hotspot_xy=None, weakest_xy=None):
        ax.set_facecolor("#0b0d10")
        if np.isfinite(hu).any():
            hu_display = np.ma.masked_invalid(hu)
            ax.imshow(
                hu_display,
                origin="lower",
                extent=extent,
                cmap="gray",
                aspect="equal",
                vmin=float(np.nanpercentile(hu, 5)) if np.isfinite(hu).any() else 0.0,
                vmax=float(np.nanpercentile(hu, 98)) if np.isfinite(hu).any() else 1.0,
                alpha=0.55,
            )
        stress_display = np.ma.masked_invalid(image)
        mapped = ax.imshow(
            stress_display,
            origin="lower",
            extent=extent,
            cmap="inferno",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            alpha=0.92,
        )
        if np.isfinite(hu).any():
            silhouette = np.isfinite(hu).astype(float)
            ax.contour(
                silhouette,
                levels=[0.5],
                extent=extent,
                origin="lower",
                colors="#8ecae6",
                linewidths=0.7,
                alpha=0.85,
            )
        if hotspot_xy is not None:
            ax.scatter(
                [hotspot_xy[0]],
                [hotspot_xy[1]],
                s=90,
                facecolors="none",
                edgecolors="cyan",
                linewidths=1.6,
                marker="o",
                label="Peak von Mises",
            )
        if weakest_xy is not None:
            ax.scatter(
                [weakest_xy[0]],
                [weakest_xy[1]],
                s=70,
                facecolors="none",
                edgecolors="lime",
                linewidths=1.6,
                marker="s",
                label="Lowest safety factor",
            )
        ax.set_title(title, color="white")
        ax.set_xlabel(x_label, color="white")
        ax.set_ylabel("Superior–inferior (mm)", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#8a8f98")
        ax.set_aspect("equal", adjustable="box")
        return mapped

    def _save_hotspot_heatmap(
        self,
        simulation: SimulationResult,
        output_dir: Path,
        study: Optional[StudyData] = None,
        segmentation: Optional[SegmentationResult] = None,
    ) -> Path:
        mesh = simulation.mesh
        centers = np.asarray(mesh.cell_centers().points if mesh.n_cells else np.zeros((0, 3)), dtype=float)
        stress = self._get_stress_values(mesh)
        heatmap_path = output_dir / "stress_heatmap_2d.png"

        if centers.ndim != 2 or centers.shape[0] == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No stress data available", ha="center", va="center", fontsize=14)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return heatmap_path

        if stress.size != centers.shape[0]:
            stress = np.resize(stress, centers.shape[0]) if stress.size else np.zeros(centers.shape[0])

        views = self.project_bone_stress_views(centers, stress, study=study, segmentation=segmentation)
        finite = stress[np.isfinite(stress)]
        vmin = float(np.min(finite)) if finite.size else 0.0
        vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0

        cell_data = self._cell_data(mesh)
        safety = np.asarray(cell_data.get("safety_factor", np.full(centers.shape[0], np.nan)), dtype=float)
        modulus = np.asarray(cell_data.get("youngs_modulus_mpa", np.full(centers.shape[0], np.nan)), dtype=float)
        hotspot_index, weakest_index = self.select_interior_extrema(centers, stress, safety=safety, modulus=modulus)

        fig, axes = plt.subplots(1, 2, figsize=(9.2, 10.5), facecolor="#111318")
        image = None
        specs = (
            (axes[0], views["ap"], views["ap_hu"], views["ap_extent"], "AP view", "Medial–lateral (mm)", 0),
            (axes[1], views["lateral"], views["lateral_hu"], views["lateral_extent"], "Lateral view", "Anterior–posterior (mm)", 1),
        )
        for ax, image_2d, hu_2d, extent, title, x_label, axis in specs:
            hotspot_xy = (float(centers[hotspot_index, axis]), float(centers[hotspot_index, 2]))
            weakest_xy = (float(centers[weakest_index, axis]), float(centers[weakest_index, 2]))
            image = self._plot_bone_view(
                ax,
                image_2d,
                hu_2d,
                extent,
                title,
                x_label,
                vmin,
                vmax,
                hotspot_xy=hotspot_xy,
                weakest_xy=weakest_xy,
            )
        axes[0].legend(loc="upper right", frameon=True, fontsize=8)
        fig.suptitle("Bone stress map  ·  AP and lateral", color="white", fontsize=14, fontweight="bold")
        fig.subplots_adjust(top=0.90, bottom=0.08, wspace=0.18, right=0.86)
        cbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
        cbar.set_label("Von Mises stress (MPa)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        fig.savefig(heatmap_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return heatmap_path

    def _save_dashboard(self, simulation: SimulationResult, risk: RiskAssessment, output_dir: Path) -> Path:
        mesh = simulation.mesh
        cell_data = self._cell_data(mesh)
        stress = np.asarray(cell_data.get("von_mises_mpa", np.zeros(mesh.n_cells)))
        safety = np.asarray(cell_data.get("safety_factor", np.ones(mesh.n_cells)))
        strain = np.asarray(cell_data.get("principal_strain", np.zeros(mesh.n_cells)))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].hist(stress[np.isfinite(stress)], bins=30, color="#c0392b", alpha=0.9)
        axes[0].set_title("Von Mises Stress")
        axes[0].set_xlabel("MPa")

        finite_safety = safety[np.isfinite(safety)]
        axes[1].hist(finite_safety, bins=30, color="#27ae60", alpha=0.9)
        axes[1].set_title("Safety Factor")
        axes[1].set_xlabel("Safety Factor")

        axes[2].scatter(stress, strain, s=10, alpha=0.4, color="#2980b9")
        axes[2].set_title(f"Risk: {risk.summary.get('risk_category', 'unknown')}")
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
                try:
                    pv.start_xvfb()
                except Exception:
                    pass
            mesh = simulation.mesh
            screenshot_path = output_dir / "stress_map.png"
            html_path = output_dir / "interactive_mesh.html"
            cell_data = self._cell_data(mesh)
            stress = np.asarray(cell_data.get("von_mises_mpa", np.zeros(getattr(mesh, "n_cells", 1))), dtype=float)
            finite = stress[np.isfinite(stress)]
            vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
            vmin = 0.0
            scalar_name = "von_mises_mpa" if "von_mises_mpa" in cell_data else None

            plotter = pv.Plotter(off_screen=True)
            plotter.set_background("white")
            plotter.add_mesh(
                mesh,
                scalars=scalar_name,
                cmap="inferno",
                clim=[vmin, max(vmax, vmin + 1e-3)],
                show_edges=False,
                scalar_bar_args={"title": "Von Mises (MPa)"},
            )
            plotter.add_text("OsteoVigil bone stress map", position="upper_left", font_size=12, color="black")
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

    def create_outputs(
        self,
        simulation: SimulationResult,
        risk: RiskAssessment,
        output_dir: Path,
        study: Optional[StudyData] = None,
        segmentation: Optional[SegmentationResult] = None,
    ) -> Dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        outputs["risk_dashboard"] = str(self._save_dashboard(simulation, risk, output_dir))
        outputs["stress_heatmap_2d"] = str(
            self._save_hotspot_heatmap(simulation, output_dir, study=study, segmentation=segmentation)
        )
        outputs.update(self._save_pyvista_plot(simulation, output_dir))
        return {key: value for key, value in outputs.items() if value}
