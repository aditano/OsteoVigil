from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODULE_NAME = "cpt_predictor.pipeline"


if importlib.util.find_spec(MODULE_NAME) is None:
    pytest.skip("pipeline module not yet implemented", allow_module_level=True)

pipeline = importlib.import_module(MODULE_NAME)


def _get_pipeline_fn():
    if hasattr(pipeline, "CPTFracturePipeline"):
        return pipeline.CPTFracturePipeline
    for name in [
        "run_pipeline",
        "execute_pipeline",
        "run_cpt_pipeline",
        "process_case",
    ]:
        if hasattr(pipeline, name):
            return getattr(pipeline, name)
    raise AttributeError("No expected pipeline entry point found")


def _make_dummy_inputs(tmp_path: Path):
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    volume = np.zeros((40, 40, 40), dtype=np.float32)
    z, y, x = np.indices(volume.shape)
    center = np.array(volume.shape) // 2
    radius = 10
    tibia = ((y - center[1]) ** 2 + (x - center[2]) ** 2) <= radius**2
    volume[tibia] = 900.0
    np.save(tmp_path / "dummy_ct.npy", volume)

    return dicom_dir, tmp_path / "dummy_ct.npy"


def test_pipeline_smoke_creates_summary_artifacts(tmp_path):
    pipeline_entry = _get_pipeline_fn()
    dicom_dir, dummy_ct = _make_dummy_inputs(tmp_path)
    output_dir = tmp_path / "outputs"

    result = None
    if inspect.isclass(pipeline_entry):
        init_kwargs = {"output_dir": str(output_dir)}
        try:
            init_kwargs.update(
                {
                    "dicom_dir": str(dicom_dir),
                    "dummy_data_path": str(dummy_ct),
                    "allow_dummy_if_missing": True,
                }
            )
            pipeline_obj = pipeline_entry(**init_kwargs)
        except TypeError:
            pipeline_obj = pipeline_entry(str(dicom_dir), str(output_dir))

        run_method = None
        for name in ["run", "execute", "process", "run_pipeline"]:
            if hasattr(pipeline_obj, name):
                run_method = getattr(pipeline_obj, name)
                break
        assert run_method is not None, "CPTFracturePipeline should expose a run method"

        try:
            result = run_method(
                dicom_dir=str(dicom_dir),
                dummy_data_path=str(dummy_ct),
                allow_dummy_if_missing=True,
            )
        except TypeError:
            result = run_method()
    else:
        run_pipeline = pipeline_entry
        kwargs = {
            "dicom_dir": str(dicom_dir),
            "output_dir": str(output_dir),
            "dummy_data_path": str(dummy_ct),
            "allow_dummy_if_missing": True,
        }

        try:
            result = run_pipeline(**kwargs)
        except TypeError:
            result = run_pipeline(str(dicom_dir), str(output_dir))

    if isinstance(result, dict):
        summary = result
    else:
        summary = {}

    possible_summary = [
        output_dir / "summary.json",
        output_dir / "simulation_summary.json",
        output_dir / "results.json",
    ]
    summary_path = next((path for path in possible_summary if path.exists()), None)

    if summary_path is not None:
        summary = json.loads(summary_path.read_text())

    assert summary
    assert any(
        key in summary
        for key in [
            "safety_factor",
            "fracture_risk",
            "risk_score",
            "estimated_years_to_failure",
        ]
    )

    produced_files = list(output_dir.glob("*"))
    assert produced_files, "pipeline should emit at least one artifact"
