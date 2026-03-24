from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .analysis import RiskAnalyzer
from .brace import BraceContactBuilder
from .config import ensure_output_dir, load_config
from .fea import FEBioModelBuilder
from .io.dicom_loader import load_ct_study
from .logging_utils import setup_logging
from .materials import BoneMaterialMapper
from .meshing import MeshBuilder
from .models import PipelineArtifacts
from .preprocessing import preprocess_study
from .reporting import PDFReportBuilder
from .segmentation import TibiaSegmenter
from .simulator import FEBioRunner
from .visualization import ResultVisualizer


ProgressCallback = Optional[Callable[[str, float], None]]


class CPTFracturePipeline:
    def __init__(
        self,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        **compat_kwargs: Any,
    ):
        merged_overrides = dict(overrides or {})
        if "allow_dummy_if_missing" in compat_kwargs:
            merged_overrides.setdefault("input", {})["allow_dummy_if_missing"] = bool(
                compat_kwargs["allow_dummy_if_missing"]
            )
        if "body_mass_kg" in compat_kwargs:
            merged_overrides.setdefault("patient", {})["body_mass_kg"] = float(compat_kwargs["body_mass_kg"])
        if "steps_per_day" in compat_kwargs:
            merged_overrides.setdefault("patient", {})["steps_per_day"] = int(compat_kwargs["steps_per_day"])

        self.config = load_config(config_path, overrides=merged_overrides or None)
        self.output_dir = ensure_output_dir(output_dir, self.config)
        self.logger = setup_logging(self.output_dir)

        self.loader = load_ct_study
        self.preprocessor = preprocess_study
        self.segmenter = TibiaSegmenter(self.config)
        self.mesh_builder = MeshBuilder(self.config)
        self.material_mapper = BoneMaterialMapper(self.config)
        self.brace_builder = BraceContactBuilder(self.config)
        self.febio_builder = FEBioModelBuilder(self.config)
        self.simulator = FEBioRunner(self.config)
        self.analyzer = RiskAnalyzer(self.config)
        self.visualizer = ResultVisualizer(self.config)
        self.report_builder = PDFReportBuilder(self.config)

    def _emit(self, progress: ProgressCallback, stage: str, fraction: float) -> None:
        self.logger.info("%s (%.0f%%)", stage, fraction * 100.0)
        if progress:
            progress(stage, fraction)

    def run(
        self,
        dicom_dir: Optional[str] = None,
        brace_stl: Optional[str] = None,
        use_agents: bool = False,
        human_in_the_loop: bool = False,
        progress: ProgressCallback = None,
        **_: Any,
    ) -> PipelineArtifacts:
        if use_agents:
            try:
                from .agents.crew import PipelineCrewOrchestrator

                orchestrator = PipelineCrewOrchestrator(self)
                return orchestrator.run(
                    dicom_dir=dicom_dir,
                    brace_stl=brace_stl,
                    human_in_the_loop=human_in_the_loop,
                    progress=progress,
                )
            except Exception as exc:
                self.logger.warning("Falling back to direct pipeline execution: %s", exc)

        artifacts = PipelineArtifacts(output_dir=self.output_dir)

        self._emit(progress, "Loading CT study", 0.08)
        study = self.loader(dicom_dir, self.config, self.output_dir)
        study = self.preprocessor(study, self.config)
        artifacts.study = study

        self._emit(progress, "Segmenting tibia", 0.20)
        segmentation = self.segmenter.segment(study, self.output_dir)
        artifacts.segmentation = segmentation

        self._emit(progress, "Generating tetrahedral mesh", 0.35)
        mesh = self.mesh_builder.build(study, segmentation, self.output_dir)
        artifacts.mesh = mesh

        self._emit(progress, "Assigning material properties", 0.48)
        materials = self.material_mapper.map_to_mesh(study, mesh, self.output_dir)
        artifacts.materials = materials

        self._emit(progress, "Preparing brace contact model", 0.58)
        brace = self.brace_builder.prepare(mesh, self.output_dir, brace_stl=brace_stl)
        artifacts.brace = brace

        self._emit(progress, "Exporting FEBio model", 0.66)
        febio = self.febio_builder.write_model(mesh, materials, brace, self.output_dir)
        artifacts.febio = febio

        if human_in_the_loop:
            response = input("Proceed to simulation? [y/N]: ").strip().lower()
            if response not in {"y", "yes"}:
                raise RuntimeError("Simulation cancelled by user.")

        self._emit(progress, "Running structural simulation", 0.80)
        simulation = self.simulator.run(febio, study, segmentation, materials, brace, self.output_dir)
        artifacts.simulation = simulation

        self._emit(progress, "Analyzing fracture risk", 0.90)
        risk = self.analyzer.analyze(simulation, self.output_dir, brace.source)
        artifacts.risk = risk

        self._emit(progress, "Creating visualizations", 0.96)
        artifacts.visualization_paths = self.visualizer.create_outputs(simulation, risk, self.output_dir)

        self._emit(progress, "Building PDF report", 0.99)
        artifacts.report_path = self.report_builder.build(artifacts)

        summary_path = self.output_dir / "summary.json"
        summary_payload = {
            "output_dir": str(self.output_dir),
            "leg_localization": artifacts.study.metadata.get("leg_localization", {}) if artifacts.study else {},
            "segmentation": artifacts.segmentation.stats if artifacts.segmentation else {},
            "mesh": artifacts.mesh.stats if artifacts.mesh else {},
            "materials": artifacts.materials.stats if artifacts.materials else {},
            "brace": {
                "enabled": artifacts.brace.enabled if artifacts.brace else False,
                "source": artifacts.brace.source if artifacts.brace else "none",
            },
            "simulation": artifacts.simulation.summary if artifacts.simulation else {},
            "risk": artifacts.risk.summary if artifacts.risk else {},
            "safety_factor": artifacts.risk.summary.get("min_safety_factor", 0.0) if artifacts.risk else 0.0,
            "fracture_risk": artifacts.risk.summary.get("risk_category", "unknown") if artifacts.risk else "unknown",
            "estimated_years_to_failure": artifacts.risk.summary.get("years_to_failure_estimate", 0.0)
            if artifacts.risk
            else 0.0,
            "report_path": str(artifacts.report_path) if artifacts.report_path else "",
            "visualization_paths": artifacts.visualization_paths,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        self._emit(progress, "Complete", 1.0)
        return artifacts
