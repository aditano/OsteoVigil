from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Sequence

warnings.filterwarnings(
    "ignore",
    message=r".*doesn't match a supported version!$",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = REPO_ROOT / "data" / "demo"
DEMO_CASES = {
    "Normal / good tibia demo": DEMO_ROOT / "normal_real_talocrural",
    "Abnormal / bad tibia demo": DEMO_ROOT / "abnormal_synthetic_cpt",
}
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "osteovigil-mpl"))


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


def _hide_streamlit_chrome() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDeployButton"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _resolve_demo_case(choice: str) -> tuple[Optional[str], Optional[str], str]:
    selected_dir = DEMO_CASES.get(choice)
    if selected_dir is None:
        return None, None, choice

    if choice.startswith("Normal"):
        dicom_dir = selected_dir
        return (str(dicom_dir) if dicom_dir.exists() else None), None, choice

    dicom_dir = selected_dir / "dicom"
    brace_path = selected_dir / "afo_proxy.stl"
    return (
        str(dicom_dir) if dicom_dir.exists() else None,
        str(brace_path) if brace_path.exists() else None,
        choice,
    )


def _load_uploads_to_tempdir(files, prefix: str) -> Optional[str]:
    if not files:
        return None

    upload_dir = Path(tempfile.mkdtemp(prefix=prefix))
    for file in files:
        (upload_dir / file.name).write_bytes(file.getbuffer())
    return str(upload_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .agents.crew import PipelineCrewOrchestrator, build_crew_manifest
    from .pipeline import CPTFracturePipeline

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
    from .agents.crew import PipelineCrewOrchestrator
    from .pipeline import CPTFracturePipeline

    st.set_page_config(page_title="OsteoVigil", layout="wide")
    _hide_streamlit_chrome()
    st.title("OsteoVigil")
    st.caption("Upload DICOM data or point to a local CT folder, then run a CPT fracture-risk workflow.")

    if "last_run" not in st.session_state:
        st.session_state.last_run = None

    with st.sidebar:
        st.header("Options")
        use_agents = st.checkbox("Enable in-depth logs", value=True)
        use_bundled_demo = st.checkbox("Use bundled demo data", value=True)
        demo_case = None
        if use_bundled_demo:
            demo_case = st.radio(
                "Bundled demo case",
                tuple(DEMO_CASES.keys()),
                index=0,
            )
        body_mass = st.number_input("Body mass (kg)", min_value=10.0, max_value=200.0, value=55.0, step=1.0)
        steps_per_day = st.number_input("Steps per day", min_value=500, max_value=30000, value=6000, step=500)
        output_dir = st.text_input("Output directory", value="outputs/streamlit_run")

    dicom_dir_input = st.text_input("Local DICOM folder path", value="", disabled=use_bundled_demo)
    dicom_files = st.file_uploader(
        "Or upload DICOM slices",
        accept_multiple_files=True,
        type=["dcm"],
        disabled=use_bundled_demo,
    )
    brace_file = st.file_uploader("Optional brace STL", type=["stl"], disabled=use_bundled_demo)

    run_clicked = st.button("Run Simulation", type="primary")
    if run_clicked:
        dicom_dir = None
        brace_path = None

        if use_bundled_demo:
            if demo_case is None:
                st.error("Select a bundled demo case.")
                return
            dicom_dir, brace_path, _ = _resolve_demo_case(demo_case)
            if not dicom_dir:
                st.error("The selected demo DICOM folder is missing.")
                return
        elif dicom_dir_input.strip():
            dicom_dir = dicom_dir_input.strip()
        elif dicom_files:
            dicom_dir = _load_uploads_to_tempdir(dicom_files, "osteovigil_dicom_")
        else:
            st.error("Select a DICOM folder, upload DICOM slices, or enable bundled demo data.")
            return

        if not use_bundled_demo and brace_file is not None:
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
                "input": {"allow_dummy_if_missing": False},
                "patient": {"body_mass_kg": float(body_mass), "steps_per_day": int(steps_per_day)},
            },
            output_dir=output_dir,
        )
        try:
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
        except Exception as exc:
            st.session_state.last_run = None
            st.error(f"Simulation failed: {exc}")
            return

        st.session_state.last_run = {
            "risk_summary": artifacts.risk.summary if artifacts.risk else {},
            "recommendations": artifacts.risk.recommendations if artifacts.risk else [],
            "visualization_paths": dict(artifacts.visualization_paths),
            "report_path": str(artifacts.report_path) if artifacts.report_path else "",
        }

    last_run = st.session_state.last_run
    if not last_run:
        return

    st.success("Simulation complete")
    risk_summary = last_run.get("risk_summary", {})
    if risk_summary:
        cols = st.columns(3)
        cols[0].metric("Risk", str(risk_summary.get("risk_category", "unknown")).title())
        cols[1].metric("Min Safety Factor", f"{float(risk_summary.get('min_safety_factor', 0.0)):.2f}")
        cols[2].metric("Years To Failure", f"{float(risk_summary.get('years_to_failure_estimate', 0.0)):.2f}")
        st.write(risk_summary.get("fracture_likely_statement", ""))
        for recommendation in last_run.get("recommendations", []):
            st.write(f"- {recommendation}")

    visualization_paths = last_run.get("visualization_paths", {})
    for key in ("stress_heatmap_2d", "stress_map", "risk_dashboard"):
        path = visualization_paths.get(key)
        if path and Path(path).exists():
            st.image(path, caption=key.replace("_", " ").title())

    report_path = last_run.get("report_path", "")
    if report_path and Path(report_path).exists():
        report_file = Path(report_path)
        st.download_button(
            "Export PDF report",
            data=report_file.read_bytes(),
            file_name=report_file.name,
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
