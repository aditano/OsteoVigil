#!/usr/bin/env bash
# OsteoVigil macOS launcher — double-click this file to start the app.
# Requires macOS to allow .command files: System Settings → Privacy & Security
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Install PyQt6 if missing (non-fatal — will fail gracefully at import)
python3 -c "import PyQt6" 2>/dev/null || {
    echo "Installing PyQt6..."
    pip3 install "PyQt6>=6.6.0" || echo "Warning: Could not install PyQt6. Please run: pip3 install 'PyQt6>=6.6.0'"
}

echo "Starting OsteoVigil..."
python3 desktop_app.py
