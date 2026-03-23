# OsteoVigil Project Status

Last updated: 2026-03-23

## Purpose

This file is the living handoff/status document for the `OsteoVigil` project. It explains:

- what has been built
- what has been verified
- what still needs to be done
- what to run next
- where the important code lives

## Project Summary

`OsteoVigil` is an open-source Python application for congenital pseudarthrosis of the tibia (CPT) fracture-risk analysis from lower-leg CT data, with optional AFO brace support. The current codebase includes:

- DICOM CT loading
- preprocessing
- tibia segmentation
- tetrahedral mesh generation
- HU-to-material-property mapping
- brace STL import or proxy brace creation
- FEBio model export
- surrogate structural simulation fallback
- fracture-risk and fatigue estimation
- visualization output
- PDF report generation
- CLI entrypoint
- Streamlit GUI
- deterministic multi-agent orchestration with CrewAI-compatible agent definitions

## What Has Been Done

### Repository setup

- Created the project from scratch in `/Users/anthonyditano/Documents/GitHub/OsteoVigil`.
- Initialized git locally.
- Added packaging/configuration files:
  - `requirements.txt`
  - `pyproject.toml`
  - `Dockerfile`
  - `.gitignore`
  - `.dockerignore`
  - `LICENSE`
- Pushed the initial repository to GitHub:
  - Remote: `https://github.com/aditano/OsteoVigil.git`
  - Branch: `main`
  - Initial pushed commit: `2d81f97`

### Documentation

- Wrote the main project outline in `README.md`.
- Included architecture, folder structure, tech stack, assumptions, limitations, disclaimers, compute needs, outputs, and roadmap.

### Core application code

- Built the main pipeline in `src/cpt_predictor/pipeline.py`.
- Added shared data models in `src/cpt_predictor/models.py`.
- Added config loading in `src/cpt_predictor/config.py`.
- Added logging setup in `src/cpt_predictor/logging_utils.py`.

### CT loading and sample data

- Added DICOM loading with:
  - `SimpleITK` first
  - `pydicom` fallback
- Added synthetic CPT demo data generation.
- Added synthetic DICOM writing for demo/smoke workflows.
- Added two included demo cases under `data/demo/`:
  - a real public normal distal tibia/fibula/ankle DICOM series
  - a synthetic abnormal CPT-style DICOM series
- Files:
  - `src/cpt_predictor/io/dicom_loader.py`
  - `src/cpt_predictor/io/sample_data.py`
  - `src/cpt_predictor/preprocessing.py`

### Segmentation

- Added classical tibia segmentation using thresholding, morphology, connected components, and gap-bridging logic.
- Added optional MONAI/TorchScript hook if a model is provided.
- File:
  - `src/cpt_predictor/segmentation.py`

### Meshing and materials

- Added marching-cubes surface reconstruction.
- Added tetrahedral volume mesh generation using TetGen when available, with PyVista fallback.
- Added HU-to-density/modulus/strength mapping using Bonemat-style equations.
- Files:
  - `src/cpt_predictor/meshing.py`
  - `src/cpt_predictor/materials.py`

### Brace and FEA setup

- Added optional brace STL import.
- Added proxy brace generation when STL is not provided.
- Added FEBio `.feb` export with:
  - material bins
  - node sets
  - distal fixation
  - proximal loading
  - simplified brace support
- Files:
  - `src/cpt_predictor/brace.py`
  - `src/cpt_predictor/fea.py`

### Simulation, analysis, and outputs

- Added FEBio subprocess runner.
- Added surrogate solver fallback so the app still runs if FEBio is not installed.
- Added stress, strain, safety factor, fatigue-cycle, and years-to-failure estimation.
- Added visualization and PDF report generation.
- Files:
  - `src/cpt_predictor/simulator.py`
  - `src/cpt_predictor/analysis.py`
  - `src/cpt_predictor/visualization.py`
  - `src/cpt_predictor/reporting.py`

### Multi-agent orchestration

- Implemented the requested named agents:
  - `ProjectManagerAgent`
  - `CTLoaderAgent`
  - `SegmentationAgent`
  - `MeshingAgent`
  - `MaterialAgent`
  - `BraceContactAgent`
  - `FEASetupAgent`
  - `SimulatorAgent`
  - `AnalyzerAgent`
  - `VisualizerAgent`
- Added deterministic local orchestration around the real pipeline.
- Added CrewAI-compatible agent definitions and manifest generation.
- File:
  - `src/cpt_predictor/agents/crew.py`

### User interfaces

- Added CLI entrypoint:
  - `main.py`
  - `src/cpt_predictor/cli.py`
- Added Streamlit app:
  - `streamlit_app.py`
- Added desktop launchers that bootstrap `.venv` and install `requirements.txt` automatically on first run:
  - `bootstrap.py`
  - `install_febio.py`
  - `launch_osteovigil.command`
  - `launch_osteovigil.bat`
- Updated the PyQt desktop app so bundled demo mode can be toggled on, a demo case can be selected, and the manual DICOM/brace inputs are disabled while demo mode is active:
  - `desktop_app.py`
- Added a repo-local FEBio manager that can auto-download or build FEBio under `.third_party/febio/` and expose it to bootstrap and runtime resolution:
  - `src/cpt_predictor/utils/febio_manager.py`
  - `install_febio.py`
  - `tests/test_febio_manager.py`

### Tests

- Added tests for:
  - material mapping behavior
  - segmentation on synthetic data
  - pipeline smoke behavior
- Files:
  - `tests/test_materials.py`
  - `tests/test_segmentation.py`
  - `tests/test_pipeline_smoke.py`

## What Has Been Verified

### Verified successfully

- Full syntax compilation passed with:

```bash
PYTHONPYCACHEPREFIX=/tmp/osteovigil_pycache python3 -m compileall src tests main.py streamlit_app.py
```

- The repo was successfully pushed to GitHub.
- Both included demo DICOM folders load successfully through `src/cpt_predictor/io/dicom_loader.py`.
- The desktop launcher bootstrap logic is in place, but it has not yet been exercised end-to-end in this sandbox with the full dependency install.
- The repo now has a single-command bootstrap path: `python bootstrap.py`
- `bootstrap.py` now fails fast with a clear error if the caller uses Python older than 3.11.
- The bundled demo folders referenced by the desktop app exist on disk and were previously validated through the DICOM loader.
- The simulator can now resolve a repo-managed FEBio executable from `.third_party/febio/` in addition to `FEBIO_EXE` and `PATH`.
- `bootstrap.py --help` and `install_febio.py --help` both run successfully under Python 3.12.
- Syntax compilation passed for `bootstrap.py`, `install_febio.py`, `src/cpt_predictor/utils/febio_manager.py`, `src/cpt_predictor/simulator.py`, and `tests/test_febio_manager.py` under Python 3.12.

### Not yet verified

- Dependencies have not been installed in this sandbox.
- `pytest` has not been run end-to-end in a fully provisioned environment.
- The updated demo-mode controls in the desktop app have not been interactively clicked through in this sandbox.
- The Streamlit app has not been interactively exercised here.
- The full demo pipeline has not been run with all scientific dependencies installed.
- FEBio execution has not been validated here.
- The new automated FEBio installer has not been exercised end-to-end against the live GitHub network in this sandbox.
- Real patient DICOM input has not been tested here.

## Current State

The codebase is scaffolded and integrated, and the project is ready for real local runtime setup. The biggest gap is not architecture anymore; it is runtime validation with the actual dependency stack and solver.

## What Needs To Be Done Next

### Immediate next steps

1. Run `python bootstrap.py` on a machine with Python 3.11 or 3.12.
2. Confirm the bootstrap path installs pinned Python dependencies and attempts the managed FEBio install.
3. Run `pytest`.
4. Run a demo pipeline using synthetic data.
5. Run the Streamlit UI and verify the basic user workflow.

### Simulation validation

1. Validate the new managed FEBio installer against the live GitHub network on macOS, Windows, and Linux.
2. If needed, fall back to setting `FEBIO_EXE` or adding FEBio to `PATH`.
3. Run the pipeline with FEBio enabled.
4. Compare FEBio mode against surrogate mode outputs on the same synthetic case.

### Real-data validation

1. Run the pipeline on a real lower-leg CT DICOM folder.
2. Confirm spacing/orientation handling.
3. Inspect segmentation quality manually.
4. Inspect mesh quality and element counts.
5. Validate material distributions and output plausibility.
6. If available, run with a real brace STL instead of the proxy brace.

### Model quality improvements

1. Replace the classical segmentation fallback with a trained MONAI or nnU-Net model.
2. Improve brace contact modeling beyond the current simplified support assumption.
3. Improve load definition using better gait or musculoskeletal inputs.
4. Calibrate HU-to-property mapping against data or literature specific to the use case.
5. Improve failure/fatigue modeling for more realistic long-term estimates.

### Productization improvements

1. Add better progress/state persistence for long runs.
2. Add structured run manifests and more robust artifact indexing.
3. Improve error reporting in the UI.
4. Add optional export bundles for clinicians/researchers.
5. Add CI once the dependency strategy is stable.

## Known Limitations

- This is a research/engineering starter system, not a validated medical device.
- The default segmentation is classical and should not be treated as clinically validated.
- The surrogate simulation is useful as a fallback but is not equivalent to a full nonlinear FEBio solve.
- Brace contact is simplified.
- Load application is simplified.
- Fatigue prediction is only a scenario estimate.
- CPT-specific clinical accuracy still requires validation against real data and domain experts.

## Important Files

### Core docs

- `README.md`
- `PROJECT_STATUS.md`

### Configuration and packaging

- `config/default.yaml`
- `requirements.txt`
- `pyproject.toml`
- `Dockerfile`

### Main app

- `main.py`
- `streamlit_app.py`
- `src/cpt_predictor/cli.py`
- `src/cpt_predictor/pipeline.py`

### Demo data

- `data/demo/README.md`
- `data/demo/normal_real_talocrural/`
- `data/demo/abnormal_synthetic_cpt/`

### Core modules

- `src/cpt_predictor/models.py`
- `src/cpt_predictor/io/dicom_loader.py`
- `src/cpt_predictor/io/sample_data.py`
- `src/cpt_predictor/preprocessing.py`
- `src/cpt_predictor/segmentation.py`
- `src/cpt_predictor/meshing.py`
- `src/cpt_predictor/materials.py`
- `src/cpt_predictor/brace.py`
- `src/cpt_predictor/fea.py`
- `src/cpt_predictor/simulator.py`
- `src/cpt_predictor/analysis.py`
- `src/cpt_predictor/visualization.py`
- `src/cpt_predictor/reporting.py`
- `src/cpt_predictor/agents/crew.py`

### Tests

- `tests/test_materials.py`
- `tests/test_segmentation.py`
- `tests/test_pipeline_smoke.py`

## Commands To Resume Work

### Environment setup

```bash
cd /Users/anthonyditano/Documents/GitHub/OsteoVigil
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run tests

```bash
PYTHONPYCACHEPREFIX=/tmp/osteovigil_pycache pytest -q
```

### Demo run

```bash
python main.py --dummy-data --use-agents --output-dir outputs/demo_run
```

### Run UI

```bash
streamlit run streamlit_app.py
```

### Real CT run

```bash
python main.py --dicom-dir /path/to/dicom --brace-stl /path/to/brace.stl --output-dir outputs/patient_run
```

### FEBio setup

```bash
python install_febio.py
```

Manual override:

```bash
export FEBIO_EXE=/absolute/path/to/febio4
```

## Suggested Working Order

1. Get the environment installed.
2. Run tests.
3. Run the synthetic demo.
4. Verify the Streamlit interface.
5. Install and validate FEBio.
6. Test with real DICOM data.
7. Upgrade segmentation and contact realism.

## Update Rule For This File

This file should be updated after every meaningful milestone:

- architecture changes
- completed runtime validation
- dependency changes
- FEBio integration changes
- real-data test results
- major bug fixes
- roadmap changes
