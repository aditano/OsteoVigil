from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert numpy / Path / non-finite values into JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class StudyData:
    volume: np.ndarray
    hu_volume: np.ndarray
    spacing_zyx: Tuple[float, float, float]
    origin_xyz: Tuple[float, float, float]
    direction: Tuple[float, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)
    normalized_volume: Optional[np.ndarray] = None
    source_dir: Optional[Path] = None


@dataclass
class SegmentationResult:
    mask: np.ndarray
    method: str
    stats: Dict[str, Any] = field(default_factory=dict)
    mask_path: Optional[Path] = None


@dataclass
class MeshResult:
    mesh: Any
    surface: Any
    mesh_path: Path
    surface_path: Path
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialResult:
    mesh: Any
    mesh_path: Path
    materials_table: List[Dict[str, Any]]
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BraceModel:
    enabled: bool
    surface: Any
    surface_path: Optional[Path]
    source: str
    support_bounds_xyz: Optional[Tuple[float, float, float, float, float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FEBioSetup:
    feb_path: Path
    manifest_path: Path
    node_sets: Dict[str, List[int]]
    load_summary: Dict[str, Any]
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    mode: str
    mesh: Any
    mesh_path: Path
    summary: Dict[str, Any]
    log_path: Optional[Path] = None
    raw_stdout: Optional[str] = None


@dataclass
class RiskAssessment:
    summary: Dict[str, Any]
    recommendations: List[str]
    summary_path: Path


@dataclass
class PipelineArtifacts:
    output_dir: Path
    study: Optional[StudyData] = None
    segmentation: Optional[SegmentationResult] = None
    mesh: Optional[MeshResult] = None
    materials: Optional[MaterialResult] = None
    brace: Optional[BraceModel] = None
    febio: Optional[FEBioSetup] = None
    simulation: Optional[SimulationResult] = None
    risk: Optional[RiskAssessment] = None
    visualization_paths: Dict[str, str] = field(default_factory=dict)
    report_path: Optional[Path] = None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "leg_localization": self.study.metadata.get("leg_localization", {}) if self.study else {},
            "segmentation": self.segmentation.stats if self.segmentation else {},
            "mesh": self.mesh.stats if self.mesh else {},
            "materials": self.materials.stats if self.materials else {},
            "brace": {
                "enabled": self.brace.enabled if self.brace else False,
                "source": self.brace.source if self.brace else "none",
            },
            "simulation": self.simulation.summary if self.simulation else {},
            "risk": self.risk.summary if self.risk else {},
            "safety_factor": self.risk.summary.get("min_safety_factor", 0.0) if self.risk else 0.0,
            "fracture_risk": self.risk.summary.get("risk_category", "unknown") if self.risk else "unknown",
            "estimated_years_to_failure": self.risk.summary.get("years_to_failure_estimate", 0.0)
            if self.risk
            else 0.0,
            "report_path": str(self.report_path) if self.report_path else "",
            "visualization_paths": self.visualization_paths,
        }

    def write_summary(self) -> Path:
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(to_jsonable(self.to_summary()), indent=2), encoding="utf-8")
        return summary_path

