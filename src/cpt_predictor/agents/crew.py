from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..analysis import RiskAnalyzer
from ..brace import BraceContactBuilder
from ..fea import FEBioModelBuilder
from ..io.dicom_loader import load_ct_study
from ..materials import BoneMaterialMapper
from ..meshing import MeshBuilder
from ..models import PipelineArtifacts
from ..preprocessing import preprocess_study
from ..reporting import PDFReportBuilder
from ..segmentation import TibiaSegmenter
from ..simulator import FEBioRunner
from ..visualization import ResultVisualizer


ProgressCallback = Optional[Callable[[str, float], None]]


@dataclass
class AgentProfile:
    name: str
    role: str
    goal: str
    backstory: str
    tools: List[str] = field(default_factory=list)


def build_agent_profiles() -> List[AgentProfile]:
    return [
        AgentProfile("ProjectManagerAgent", "Project coordinator", "Coordinate the workflow and approvals.", "A biomedical systems lead focused on reproducible delivery.", ["manifest", "approval_gate"]),
        AgentProfile("CTLoaderAgent", "CT ingestion specialist", "Load DICOM CT data into a consistent HU volume.", "A DICOM engineer who handles slice ordering and metadata cleanup.", ["SimpleITK", "pydicom"]),
        AgentProfile("SegmentationAgent", "Segmentation specialist", "Segment the tibia and pseudarthrosis region.", "A medical-imaging engineer who prefers safe MONAI fallbacks and classical methods.", ["MONAI", "scikit-image"]),
        AgentProfile("MeshingAgent", "Meshing specialist", "Create a tetrahedral bone mesh suitable for FEA.", "A computational mechanics engineer who cares about solver stability.", ["PyVista", "TetGen"]),
        AgentProfile("MaterialAgent", "Material-property mapper", "Map HU to density, modulus, and strength.", "A biomechanics analyst using Bonemat-style relationships.", ["NumPy"]),
        AgentProfile("BraceContactAgent", "Brace/contact specialist", "Load or synthesize the brace and support zone.", "An orthotics modeler who keeps brace assumptions explicit.", ["STL", "PyVista"]),
        AgentProfile("FEASetupAgent", "FEBio setup specialist", "Write the FEBio model and simulation manifest.", "A finite-element engineer building open and reproducible solver decks.", ["FEBio", "XML"]),
        AgentProfile("SimulatorAgent", "Simulation specialist", "Run FEBio or the surrogate structural solver.", "A pragmatic solver operator who always has a fallback.", ["subprocess", "surrogate_solver"]),
        AgentProfile("AnalyzerAgent", "Risk-analysis specialist", "Compute fracture risk, fatigue, and recommendations.", "A risk analyst turning mechanics output into interpretable metrics.", ["NumPy", "statistics"]),
        AgentProfile("VisualizerAgent", "Visualization specialist", "Create 3D maps, dashboards, and the PDF report.", "A scientific communicator focused on usable output artifacts.", ["PyVista", "Matplotlib", "ReportLab"]),
    ]


def build_crew_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "project": config.get("project", {}).get("name", "OsteoVigil"),
        "agents": [asdict(profile) for profile in build_agent_profiles()],
        "sequence": [profile.name for profile in build_agent_profiles()],
        "human_in_the_loop": bool(config.get("orchestration", {}).get("human_in_the_loop", False)),
    }


def build_crewai_agents(llm: Any = None) -> Optional[List[Any]]:
    try:
        from crewai import Agent
    except Exception:
        return None

    return [
        Agent(
            role=profile.role,
            goal=profile.goal,
            backstory=profile.backstory,
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )
        for profile in build_agent_profiles()
    ]


def create_crewai_crew(config: Dict[str, Any], llm: Any = None) -> Optional[Any]:
    try:
        from crewai import Crew, Process, Task
    except Exception:
        return None

    agents = build_crewai_agents(llm=llm)
    if not agents:
        return None

    manifest = build_crew_manifest(config)
    tasks = []
    for index, stage in enumerate(manifest["sequence"]):
        tasks.append(
            Task(
                description=f"Execute the `{stage}` stage and hand off its artifacts to the next agent.",
                expected_output=f"Stage handoff for {stage}",
                agent=agents[index],
            )
        )
    return Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False)


class PipelineCrewOrchestrator:
    """Deterministic local multi-agent wrapper around the main pipeline."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def _emit(self, agent_name: str, stage: str, fraction: float, progress: ProgressCallback) -> None:
        message = f"{agent_name}: {stage}"
        self.pipeline.logger.info(message)
        if progress:
            progress(message, fraction)

    def _approval(self, prompt_text: str, enabled: bool) -> None:
        if not enabled:
            return
        response = input(f"{prompt_text} [y/N]: ").strip().lower()
        if response not in {"y", "yes"}:
            raise RuntimeError(f"Paused by user during {prompt_text}")

    def run(
        self,
        dicom_dir: Optional[str] = None,
        brace_stl: Optional[str] = None,
        human_in_the_loop: bool = False,
        progress: ProgressCallback = None,
    ) -> PipelineArtifacts:
        artifacts = PipelineArtifacts(output_dir=self.pipeline.output_dir)
        manifest = build_crew_manifest(self.pipeline.config)
        (self.pipeline.output_dir / "crew_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        self._emit("ProjectManagerAgent", "Initialized workflow manifest", 0.03, progress)

        self._emit("CTLoaderAgent", "Loading CT study", 0.10, progress)
        study = load_ct_study(dicom_dir, self.pipeline.config, self.pipeline.output_dir)
        study = preprocess_study(study, self.pipeline.config)
        artifacts.study = study

        self._emit("SegmentationAgent", "Segmenting tibia", 0.22, progress)
        artifacts.segmentation = TibiaSegmenter(self.pipeline.config).segment(study, self.pipeline.output_dir)

        self._emit("MeshingAgent", "Generating tetrahedral mesh", 0.36, progress)
        artifacts.mesh = MeshBuilder(self.pipeline.config).build(study, artifacts.segmentation, self.pipeline.output_dir)

        self._emit("MaterialAgent", "Assigning heterogeneous properties", 0.48, progress)
        artifacts.materials = BoneMaterialMapper(self.pipeline.config).map_to_mesh(study, artifacts.mesh, self.pipeline.output_dir)

        self._emit("BraceContactAgent", "Preparing brace model", 0.58, progress)
        artifacts.brace = BraceContactBuilder(self.pipeline.config).prepare(
            artifacts.mesh,
            self.pipeline.output_dir,
            brace_stl=brace_stl,
        )

        self._approval("Proceed to FEBio export?", human_in_the_loop)
        self._emit("FEASetupAgent", "Writing FEBio model", 0.68, progress)
        artifacts.febio = FEBioModelBuilder(self.pipeline.config).write_model(
            artifacts.mesh,
            artifacts.materials,
            artifacts.brace,
            self.pipeline.output_dir,
        )

        self._approval("Proceed to simulation?", human_in_the_loop)
        self._emit("SimulatorAgent", "Running structural simulation", 0.82, progress)
        artifacts.simulation = FEBioRunner(self.pipeline.config).run(
            artifacts.febio,
            artifacts.study,
            artifacts.segmentation,
            artifacts.materials,
            artifacts.brace,
            self.pipeline.output_dir,
        )

        self._emit("AnalyzerAgent", "Computing fracture risk", 0.91, progress)
        artifacts.risk = RiskAnalyzer(self.pipeline.config).analyze(
            artifacts.simulation,
            self.pipeline.output_dir,
            artifacts.brace.source,
        )

        self._emit("VisualizerAgent", "Generating visuals and report", 0.97, progress)
        artifacts.visualization_paths = ResultVisualizer(self.pipeline.config).create_outputs(
            artifacts.simulation,
            artifacts.risk,
            self.pipeline.output_dir,
        )
        artifacts.report_path = PDFReportBuilder(self.pipeline.config).build(artifacts)

        self._emit("ProjectManagerAgent", "Workflow complete", 1.0, progress)
        return artifacts
