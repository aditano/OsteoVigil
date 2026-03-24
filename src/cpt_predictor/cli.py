from __future__ import annotations

import argparse
import json
import os
import shutil
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


def _resolve_febio_candidate(explicit_path: Optional[str] = None) -> Optional[str]:
    from .utils.febio_manager import resolve_managed_febio_executable

    if explicit_path and Path(explicit_path).exists():
        return explicit_path
    env_value = os.getenv("FEBIO_EXE")
    if env_value and Path(env_value).exists():
        return env_value
    managed = resolve_managed_febio_executable()
    if managed and managed.exists():
        return str(managed)
    for candidate in ("febio4", "febio4.exe", "febio3", "febio3.exe"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _solver_availability_status(explicit_path: Optional[str] = None) -> tuple[str, str, str]:
    detected = _resolve_febio_candidate(explicit_path)
    if detected:
        return (
            "success",
            "Solver availability: FEBio detected",
            f"FEBio executable found at {detected}. The completed-run banner below will confirm whether FEBio or the fallback was actually used.",
        )
    return (
        "warning",
        "Solver availability: fallback surrogate only",
        "No FEBio executable was detected for this session, so the built-in surrogate solver will be used.",
    )


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
    parser.add_argument(
        "--target-leg",
        choices=("auto", "left", "right"),
        default="auto",
        help="For bilateral or full-body CT scans, select which leg to analyze.",
    )
    parser.add_argument("--print-manifest", action="store_true", help="Print the multi-agent manifest before running.")
    return parser


def _runtime_overrides(args: argparse.Namespace) -> dict:
    return {
        "input": {
            "allow_dummy_if_missing": bool(args.dummy_data),
            "target_leg_side": str(getattr(args, "target_leg", "auto")),
        },
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


def _render_solver_status(level: str, title: str, detail: str) -> None:
    import streamlit as st

    renderer = getattr(st, level, st.info)
    renderer(title)
    st.caption(detail)


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


def _run_streamlit_pipeline(
    *,
    dicom_dir: str,
    brace_path: Optional[str],
    use_agents: bool,
    target_leg: str,
    body_mass: float,
    steps_per_day: int,
    output_dir: str,
):
    from .agents.crew import PipelineCrewOrchestrator
    from .pipeline import CPTFracturePipeline

    import streamlit as st

    progress_bar = st.progress(0)
    status = st.empty()

    def progress(stage: str, fraction: float) -> None:
        progress_bar.progress(min(max(float(fraction), 0.0), 1.0))
        status.write(stage)

    pipeline = CPTFracturePipeline(
        overrides={
            "input": {"allow_dummy_if_missing": False, "target_leg_side": target_leg},
            "patient": {"body_mass_kg": float(body_mass), "steps_per_day": int(steps_per_day)},
        },
        output_dir=output_dir,
    )
    if use_agents:
        return PipelineCrewOrchestrator(pipeline).run(
            dicom_dir=dicom_dir,
            brace_stl=brace_path,
            human_in_the_loop=False,
            progress=progress,
        )
    return pipeline.run(
        dicom_dir=dicom_dir,
        brace_stl=brace_path,
        use_agents=False,
        human_in_the_loop=False,
        progress=progress,
    )


def _store_streamlit_run(artifacts) -> dict:
    return {
        "risk_summary": artifacts.risk.summary if artifacts.risk else {},
        "recommendations": artifacts.risk.recommendations if artifacts.risk else [],
        "visualization_paths": dict(artifacts.visualization_paths),
        "report_path": str(artifacts.report_path) if artifacts.report_path else "",
        "study_metadata": dict(artifacts.study.metadata) if artifacts.study else {},
    }


def _resolve_streamlit_inputs(
    *,
    use_bundled_demo: bool,
    demo_case: Optional[str],
    dicom_dir_input: str,
    dicom_files,
    brace_file,
):
    dicom_dir = None
    brace_path = None

    if use_bundled_demo:
        if demo_case is None:
            raise ValueError("Select a bundled demo case.")
        dicom_dir, brace_path, _ = _resolve_demo_case(demo_case)
        if not dicom_dir:
            raise FileNotFoundError("The selected demo DICOM folder is missing.")
        return dicom_dir, brace_path

    if dicom_dir_input.strip():
        dicom_dir = dicom_dir_input.strip()
    elif dicom_files:
        dicom_dir = _load_uploads_to_tempdir(dicom_files, "osteovigil_dicom_")
    else:
        raise ValueError("Select a DICOM folder, upload DICOM slices, or enable bundled demo data.")

    if brace_file is not None:
        brace_dir = Path(tempfile.mkdtemp(prefix="osteovigil_brace_"))
        brace_path_obj = brace_dir / brace_file.name
        brace_path_obj.write_bytes(brace_file.getbuffer())
        brace_path = str(brace_path_obj)

    return dicom_dir, brace_path


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
                "simulation_mode": artifacts.risk.summary.get("simulation_mode") if artifacts.risk else "unknown",
                "leg_localization": artifacts.study.metadata.get("leg_localization", {}) if artifacts.study else {},
                "report_path": str(artifacts.report_path) if artifacts.report_path else "",
                "output_dir": str(artifacts.output_dir),
            },
            indent=2,
        )
    )
    return 0


def run_streamlit_app() -> None:
    import streamlit as st
    from .preprocessing import MultipleLegSelectionRequiredError

    # Tell the runtime layer to avoid PyVista/VTK window creation on macOS.
    os.environ.setdefault("OSTEOVIGIL_DISABLE_PYVISTA_WINDOWING", "1")
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "1")
    os.environ.setdefault("PYVISTA_USE_PANEL", "0")

    st.set_page_config(page_title="OsteoVigil", layout="wide")
    _hide_streamlit_chrome()
    st.title("OsteoVigil")
    st.caption("Upload DICOM data or point to a local CT folder, then run a CPT fracture-risk workflow.")
    availability_level, availability_title, availability_detail = _solver_availability_status()
    _render_solver_status(availability_level, availability_title, availability_detail)

    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    if "pending_leg_selection" not in st.session_state:
        st.session_state.pending_leg_selection = None

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

        _render_solver_status(availability_level, availability_title, availability_detail)

    dicom_dir_input = st.text_input("Local DICOM folder path", value="", disabled=use_bundled_demo)
    dicom_files = st.file_uploader(
        "Or upload DICOM slices",
        accept_multiple_files=True,
        type=["dcm"],
        disabled=use_bundled_demo,
    )
    brace_file = st.file_uploader("Optional brace STL", type=["stl"], disabled=use_bundled_demo)

    run_clicked = st.button("Run Simulation", type="primary")
    pending_leg_selection = st.session_state.pending_leg_selection
    selected_pending_leg = None
    continue_clicked = False
    if pending_leg_selection:
        scan_scope = str(pending_leg_selection.get("scan_scope", "multi-leg scan")).replace("_", " ")
        st.warning(f"Detected {scan_scope} data. Choose which leg to analyze to continue.")
        selected_pending_leg = st.selectbox(
            "Leg to analyze",
            ("left", "right"),
            key="pending_leg_choice",
            help="This option only appears when the uploaded scan includes more than one leg.",
        )
        continue_clicked = st.button("Analyze Selected Leg")

    if run_clicked:
        try:
            dicom_dir, brace_path = _resolve_streamlit_inputs(
                use_bundled_demo=use_bundled_demo,
                demo_case=demo_case,
                dicom_dir_input=dicom_dir_input,
                dicom_files=dicom_files,
                brace_file=brace_file,
            )
            artifacts = _run_streamlit_pipeline(
                dicom_dir=dicom_dir,
                brace_path=brace_path,
                use_agents=use_agents,
                target_leg="auto",
                body_mass=float(body_mass),
                steps_per_day=int(steps_per_day),
                output_dir=output_dir,
            )
        except MultipleLegSelectionRequiredError as exc:
            st.session_state.last_run = None
            st.session_state.pending_leg_selection = {
                "dicom_dir": dicom_dir,
                "brace_path": brace_path,
                "use_agents": bool(use_agents),
                "body_mass": float(body_mass),
                "steps_per_day": int(steps_per_day),
                "output_dir": output_dir,
                "scan_scope": exc.scan_scope,
            }
            st.rerun()
        except Exception as exc:
            st.session_state.last_run = None
            st.session_state.pending_leg_selection = None
            st.error(f"Simulation failed: {exc}")
            return
        st.session_state.pending_leg_selection = None
        st.session_state.last_run = _store_streamlit_run(artifacts)

    if continue_clicked:
        pending_leg_selection = st.session_state.pending_leg_selection
        if not pending_leg_selection:
            st.error("There is no pending multi-leg scan to continue.")
            return
        try:
            artifacts = _run_streamlit_pipeline(
                dicom_dir=str(pending_leg_selection["dicom_dir"]),
                brace_path=pending_leg_selection.get("brace_path"),
                use_agents=bool(pending_leg_selection.get("use_agents", False)),
                target_leg=str(selected_pending_leg or "left"),
                body_mass=float(pending_leg_selection.get("body_mass", body_mass)),
                steps_per_day=int(pending_leg_selection.get("steps_per_day", steps_per_day)),
                output_dir=str(pending_leg_selection.get("output_dir", output_dir)),
            )
        except Exception as exc:
            st.session_state.last_run = None
            st.error(f"Simulation failed: {exc}")
            return
        st.session_state.pending_leg_selection = None
        st.session_state.last_run = _store_streamlit_run(artifacts)

    last_run = st.session_state.last_run
    if not last_run:
        return

    st.success("Simulation complete")
    risk_summary = last_run.get("risk_summary", {})
    study_metadata = last_run.get("study_metadata", {})
    localization = study_metadata.get("leg_localization", {})
    simulation_mode = str(risk_summary.get("simulation_mode", "unknown"))
    if simulation_mode == "surrogate":
        _render_solver_status(
            "warning",
            "Solver used for this run: fallback surrogate",
            "This completed analysis used the built-in surrogate solver rather than FEBio.",
        )
    elif simulation_mode.startswith("febio"):
        _render_solver_status(
            "success",
            "Solver used for this run: FEBio",
            f"The completed analysis reported solver mode {simulation_mode}.",
        )
    else:
        _render_solver_status(
            "info",
            f"Solver used for this run: {simulation_mode}",
            "The completed analysis reported a non-standard solver mode.",
        )
    if localization.get("cropped"):
        chosen_side = str(localization.get("target_leg_side", "unknown")).title()
        scan_scope = str(localization.get("scan_scope", "bilateral scan")).replace("_", " ")
        st.info(f"Detected {scan_scope} data and cropped to the {chosen_side} leg before tibia analysis.")
    if risk_summary:
        cols = st.columns(3)
        cols[0].metric("Risk", str(risk_summary.get("risk_category", "unknown")).title())
        cols[1].metric("Min Safety Factor", f"{float(risk_summary.get('min_safety_factor', 0.0)):.2f}")
        cols[2].metric("Years To Failure", f"{float(risk_summary.get('years_to_failure_estimate', 0.0)):.2f}")
        st.write(risk_summary.get("fracture_likely_statement", ""))
        for recommendation in last_run.get("recommendations", []):
            st.write(f"- {recommendation}")

    visualization_paths = last_run.get("visualization_paths", {})
    visualization_captions = {
        "stress_heatmap_2d": "Stress Heatmaps (AP and Lateral Views)",
        "stress_map": "3D Stress Map",
        "risk_dashboard": "Risk Dashboard",
    }
    for key in ("stress_heatmap_2d", "stress_map", "risk_dashboard"):
        path = visualization_paths.get(key)
        if path and Path(path).exists():
            st.image(path, caption=visualization_captions.get(key, key.replace("_", " ").title()))

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
