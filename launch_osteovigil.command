#!/usr/bin/env bash
# OsteoVigil macOS launcher — double-click this file to bootstrap and start the app.
# Requires macOS to allow .command files: System Settings -> Privacy & Security

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

find_python() {
    for candidate in python3.11 python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

BOOTSTRAP_PYTHON="$(find_python || true)"
if [ -z "${BOOTSTRAP_PYTHON}" ]; then
    echo "No compatible Python interpreter was found."
    echo "Please install Python 3.11 or 3.12, then rerun this launcher."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating local virtual environment with ${BOOTSTRAP_PYTHON}..."
    "${BOOTSTRAP_PYTHON}" -m venv .venv
fi

source ".venv/bin/activate"

REQ_STAMP=".venv/.osteovigil_requirements_installed"
if [ ! -f "${REQ_STAMP}" ] || [ "requirements.txt" -nt "${REQ_STAMP}" ]; then
    echo "Installing OsteoVigil dependencies..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    touch "${REQ_STAMP}"
fi

echo "Starting OsteoVigil..."
python desktop_app.py
