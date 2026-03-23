from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from .agents.crew import PipelineCrewOrchestrator, build_crew_manifest
from .pipeline import CPTFracturePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OsteoVigil CPT fracture prediction workflow")
    parser.add_argument("--dicom-dir", default=None, help="Path to a folder of CT DICOM slices.")
    parser.add_argument("--brace-stl", default=None, help="Optional STL path for the AFO brace.")
    parser.add_argument("--output-dir", default="outputs/run", help="Where to save artifacts.")
    parser.add_argument("--config", default=None, help="Optional YAML config override file.")
    parser.add_argument("--dummy-data", action="store_true", help="Use synthetic demo data if DICOM is unavailable.")
    parser.add_argument("--use-agents", action="store_true", help="Run via the named multi-agent orchestrator.")
    parser.add_argument("--human-in-the-loop", action="store_true", help="Prompt before export and simulation.")
    parser.add_argument("--surrogate-only", action="store_true", help="Force the surrogate solver instead of FEBio.")
    parser.add_argument("--febio-exe", default=None, help="Optional explicit FEBio executable path.")
    parser.add_argument("--body-mass-kg", type=float, default=55.0, help="Patient body mass in kilograms.")
    parser.add_argument("--steps-per-day", type=int, default=6000, help="Daily activity assumption.")
    parser.add_argument("--print-manifest", action="store_true", help="Print the multi-agent manifest before running.")
    return parser


def _runtime_overrides(args: argparse.Namespace) -> dict:
    return {
        "input": {"allow_dummy_if_missing": bool(args.dummy_data)},
        "patient": {
            "body_mass_kg": float(args.body_mass_kg),
            "steps_per_day": int(args.steps_per_day),
        },
        "orchestration": {"human_in_the_loop": bool(args.human_in_the_loop)},
        "simulation": {
            "prefer_febio": not bool(args.surrogate_only),
            "febio_exe": args.febio_exe,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = CPTFracturePipeline(
        config_path=args.config,
        overrides=_runtime_overrides(args),
        output_dir=args.output_dir,
    )

    if args.print_manifest:
        print(json.dumps(build_crew_manifest(pipeline.config), indent=2))

    artifacts = pipeline.run(
        dicom_dir=args.dicom_dir,
        brace_stl=args.brace_stl,
        use_agents=args.use_agents,
        human_in_the_loop=args.human_in_the_loop,
    )

    print(
        json.dumps(
            {
                "risk_category": artifacts.risk.summary.get("risk_category") if artifacts.risk else "unknown",
                "min_safety_factor": artifacts.risk.summary.get("min_safety_factor") if artifacts.risk else 0.0,
                "years_to_failure_estimate": artifacts.risk.summary.get("years_to_failure_estimate")
                if artifacts.risk
                else 0.0,
                "report_path": str(artifacts.report_path) if artifacts.report_path else "",
                "output_dir": str(artifacts.output_dir),
            },
            indent=2,
        )
    )
    return 0


def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="OsteoVigil", layout="wide")
    st.title("OsteoVigil")
    st.caption("Upload DICOM data or point to a local CT folder, then run a CPT fracture-risk workflow.")

    with st.sidebar:
        st.header("Options")
        use_agents = st.checkbox("Use multi-agent orchestrator", value=True)
        use_dummy = st.checkbox("Allow synthetic demo fallback", value=True)
        body_mass = st.number_input("Body mass (kg)", min_value=10.0, max_value=200.0, value=55.0, step=1.0)
        steps_per_day = st.number_input("Steps per day", min_value=500, max_value=30000, value=6000, step=500)
        output_dir = st.text_input("Output directory", value="outputs/streamlit_run")

    dicom_dir_input = st.text_input("Local DICOM folder path", value="")
    dicom_files = st.file_uploader("Or upload DICOM slices", accept_multiple_files=True, type=["dcm"])
    brace_file = st.file_uploader("Optional brace STL", type=["stl"])

    if not st.button("Run Simulation", type="primary"):
        return

    dicom_dir = None
    if dicom_dir_input.strip():
        dicom_dir = dicom_dir_input.strip()
    elif dicom_files:
        upload_dir = Path(tempfile.mkdtemp(prefix="osteovigil_dicom_"))
        for file in dicom_files:
            (upload_dir / file.name).write_bytes(file.getbuffer())
        dicom_dir = str(upload_dir)

    brace_path = None
    if brace_file is not None:
        brace_dir = Path(tempfile.mkdtemp(prefix="osteovigil_brace_"))
        brace_path_obj = brace_dir / brace_file.name
        brace_path_obj.write_bytes(brace_file.getbuffer())
        brace_path = str(brace_path_obj)

    progress_bar = st.progress(0)
    status = st.empty()

    def progress(stage: str, fraction: float) -> None:
        progress_bar.progress(min(max(float(fraction), 0.0), 1.0))
        status.write(stage)

    pipeline = CPTFracturePipeline(
        overrides={
            "input": {"allow_dummy_if_missing": bool(use_dummy)},
            "patient": {"body_mass_kg": float(body_mass), "steps_per_day": int(steps_per_day)},
        },
        output_dir=output_dir,
    )

    if use_agents:
        artifacts = PipelineCrewOrchestrator(pipeline).run(
            dicom_dir=dicom_dir,
            brace_stl=brace_path,
            human_in_the_loop=False,
            progress=progress,
        )
    else:
        artifacts = pipeline.run(
            dicom_dir=dicom_dir,
            brace_stl=brace_path,
            use_agents=False,
            human_in_the_loop=False,
            progress=progress,
        )

    st.success("Simulation complete")
    if artifacts.risk:
        cols = st.columns(3)
        cols[0].metric("Risk", artifacts.risk.summary["risk_category"].title())
        cols[1].metric("Min Safety Factor", f"{artifacts.risk.summary['min_safety_factor']:.2f}")
        cols[2].metric("Years To Failure", f"{artifacts.risk.summary['years_to_failure_estimate']:.2f}")
        st.write(artifacts.risk.summary["fracture_likely_statement"])
        for recommendation in artifacts.risk.recommendations:
            st.write(f"- {recommendation}")

    for key in ("stress_map", "risk_dashboard"):
        path = artifacts.visualization_paths.get(key)
        if path and Path(path).exists():
            st.image(path, caption=key.replace("_", " ").title())

    st.json(
        {
            "output_dir": str(artifacts.output_dir),
            "report_path": str(artifacts.report_path) if artifacts.report_path else "",
            "manifest": build_crew_manifest(pipeline.config),
        }
    )
