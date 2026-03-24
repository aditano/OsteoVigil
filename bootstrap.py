#!/usr/bin/env python3
"""Bootstrap OsteoVigil into a local virtual environment and launch it."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


MIN_PYTHON = (3, 11)
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cpt_predictor.utils.febio_manager import ensure_febio_available

REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"
VENV_DIR = REPO_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQ_STAMP = VENV_DIR / ".osteovigil_requirements_installed"


def default_entrypoint() -> str:
    if sys.platform == "darwin":
        return "streamlit"
    return "desktop"


def configure_qt_environment(env: dict[str, str]) -> None:
    pyqt_root = None

    if os.name == "nt":
        candidate = VENV_DIR / "Lib" / "site-packages" / "PyQt6" / "Qt6"
        if candidate.exists():
            pyqt_root = candidate
    else:
        lib_dir = VENV_DIR / "lib"
        for candidate in sorted(lib_dir.glob("python*/site-packages/PyQt6/Qt6")):
            if candidate.exists():
                pyqt_root = candidate
                break

    if not pyqt_root:
        return

    plugins_dir = pyqt_root / "plugins"
    platforms_dir = plugins_dir / "platforms"
    if plugins_dir.exists():
        env.setdefault("QT_PLUGIN_PATH", str(plugins_dir))
    if platforms_dir.exists():
        env.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms_dir))
    if sys.platform == "darwin":
        env.setdefault("QT_QPA_PLATFORM", "cocoa")


def configure_streamlit_environment(env: dict[str, str]) -> None:
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("STREAMLIT_BROWSER_SERVER_ADDRESS", "localhost")
    env.setdefault("STREAMLIT_SERVER_ADDRESS", "localhost")
    env.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
    env.setdefault("STREAMLIT_SERVER_RUN_ON_SAVE", "false")


def run_command(cmd: list[str]) -> None:
    print(f"[bootstrap] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def ensure_supported_python() -> None:
    if sys.version_info >= MIN_PYTHON:
        return

    version_label = ".".join(str(part) for part in MIN_PYTHON)
    raise SystemExit(
        f"OsteoVigil requires Python {version_label}+ for bootstrap. "
        f"Current interpreter is {sys.version.split()[0]}. "
        f"Please rerun with Python {version_label} or newer."
    )


def ensure_virtualenv() -> None:
    if VENV_PYTHON.exists():
        return

    print(f"[bootstrap] Creating virtual environment with {sys.executable}...")
    run_command([sys.executable, "-m", "venv", str(VENV_DIR)])


def needs_dependency_install() -> bool:
    if not REQ_STAMP.exists():
        return True
    return REQUIREMENTS_FILE.stat().st_mtime > REQ_STAMP.stat().st_mtime


def ensure_dependencies() -> None:
    if not needs_dependency_install():
        return

    print("[bootstrap] Installing project dependencies...")
    run_command([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
    run_command([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
    REQ_STAMP.touch()


def ensure_febio(force_reinstall: bool = False) -> Optional[Path]:
    if os.getenv("OSTEOVIGIL_SKIP_FEBIO_BOOTSTRAP", "").strip().lower() in {"1", "true", "yes"}:
        print("[bootstrap] Skipping automatic FEBio install because OSTEOVIGIL_SKIP_FEBIO_BOOTSTRAP is set.")
        return None

    try:
        febio_exe = ensure_febio_available(
            python_executable=Path(sys.executable),
            repo_root=REPO_ROOT,
            force_reinstall=force_reinstall,
            prefer_release_assets=True,
            verbose=True,
        )
    except Exception as exc:
        print(f"[bootstrap] Warning: automatic FEBio setup failed: {exc}")
        print("[bootstrap] Continuing with the built-in surrogate solver fallback.")
        return None

    print(f"[bootstrap] FEBio is ready at {febio_exe}")
    return febio_exe


def launch_command(entrypoint: str, entry_args: list[str]) -> list[str]:
    if entrypoint == "desktop":
        return [str(VENV_PYTHON), str(REPO_ROOT / "desktop_app.py"), *entry_args]
    if entrypoint == "cli":
        return [str(VENV_PYTHON), str(REPO_ROOT / "main.py"), *entry_args]
    if entrypoint == "streamlit":
        return [
            str(VENV_PYTHON),
            "-m",
            "streamlit",
            "run",
            str(REPO_ROOT / "streamlit_app.py"),
            *entry_args,
        ]
    raise ValueError(f"Unsupported entrypoint: {entrypoint}")


def run_entrypoint(entrypoint: str, entry_args: list[str], launch_env: dict[str, str]) -> int:
    command = launch_command(entrypoint, entry_args)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=launch_env, check=False)
    if completed.returncode == 0:
        return 0

    if entrypoint == "desktop" and sys.platform == "darwin":
        print("[bootstrap] Desktop Qt launch failed on macOS.")
        print("[bootstrap] Falling back to the Streamlit interface instead.")
        fallback = launch_command("streamlit", [])
        return subprocess.call(fallback, cwd=REPO_ROOT, env=launch_env)

    raise subprocess.CalledProcessError(completed.returncode, command)


def parse_args() -> tuple[str, bool, list[str]]:
    entrypoint_default = default_entrypoint()
    parser = argparse.ArgumentParser(
        description="Create .venv, install dependencies, and launch OsteoVigil."
    )
    parser.add_argument(
        "--entrypoint",
        choices=("desktop", "cli", "streamlit"),
        default=entrypoint_default,
        help="Application entrypoint to launch after bootstrap completes.",
    )
    parser.add_argument(
        "--force-febio-reinstall",
        action="store_true",
        help="Force a fresh FEBio auto-install attempt before launch.",
    )
    parser.add_argument(
        "entry_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected entrypoint. Prefix with -- to separate them.",
    )
    args = parser.parse_args()

    entry_args = args.entry_args
    if entry_args[:1] == ["--"]:
        entry_args = entry_args[1:]
    return args.entrypoint, bool(args.force_febio_reinstall), entry_args


def main() -> int:
    entrypoint, force_febio_reinstall, entry_args = parse_args()
    ensure_supported_python()
    ensure_virtualenv()
    ensure_dependencies()

    if Path(sys.prefix).resolve() != VENV_DIR.resolve():
        relaunch = [str(VENV_PYTHON), str(REPO_ROOT / "bootstrap.py"), "--entrypoint", entrypoint]
        if force_febio_reinstall:
            relaunch.append("--force-febio-reinstall")
        if entry_args:
            relaunch.extend(["--", *entry_args])
        os.execv(str(VENV_PYTHON), relaunch)

    febio_exe = ensure_febio(force_reinstall=force_febio_reinstall)
    launch_env = os.environ.copy()
    if febio_exe:
        launch_env["FEBIO_EXE"] = str(febio_exe)
    configure_qt_environment(launch_env)
    configure_streamlit_environment(launch_env)

    return run_entrypoint(entrypoint, entry_args, launch_env)


if __name__ == "__main__":
    raise SystemExit(main())
