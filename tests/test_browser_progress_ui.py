"""The strength page shows run progress and an honest CPU/GPU badge."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_progress_bar_and_compute_badge_are_in_the_page() -> None:
    page = (REPO_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    main = (REPO_ROOT / "web" / "src" / "main.ts").read_text(encoding="utf-8")
    solver = (REPO_ROOT / "web" / "src" / "browserSolver.ts").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert 'id="progress-track"' in page
    assert 'id="progress-percent"' in page
    assert 'id="compute-badge"' in page
    assert 'id="engine-note"' in page
    assert "The solver runs in this browser. The DICOM stays on this computer." in main
    assert "GPU unavailable · CPU only" in main or "formatComputeBadge" in main
    assert 'export const SOLVER_THREAD = "main"' in solver
    assert "Decoding DICOM…" in solver
    assert "Loading NumPy…" in solver
    assert "Pyodide/WASM" in readme
    assert "GPU unavailable · CPU only" in readme
    assert "2D canvas" in readme
