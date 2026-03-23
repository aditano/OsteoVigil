#!/usr/bin/env python3
"""Bootstrap OsteoVigil into a local virtual environment and launch it."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


MIN_PYTHON = (3, 11)
REPO_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"
VENV_DIR = REPO_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQ_STAMP = VENV_DIR / ".osteovigil_requirements_installed"


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


def parse_args() -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(
        description="Create .venv, install dependencies, and launch OsteoVigil."
    )
    parser.add_argument(
        "--entrypoint",
        choices=("desktop", "cli", "streamlit"),
        default="desktop",
        help="Application entrypoint to launch after bootstrap completes.",
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
    return args.entrypoint, entry_args


def main() -> int:
    entrypoint, entry_args = parse_args()
    ensure_supported_python()
    ensure_virtualenv()
    ensure_dependencies()

    if Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        relaunch = [str(VENV_PYTHON), str(REPO_ROOT / "bootstrap.py"), "--entrypoint", entrypoint]
        if entry_args:
            relaunch.extend(["--", *entry_args])
        os.execv(str(VENV_PYTHON), relaunch)

    subprocess.check_call(launch_command(entrypoint, entry_args), cwd=REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
