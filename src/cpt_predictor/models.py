from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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

