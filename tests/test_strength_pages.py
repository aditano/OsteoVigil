"""Static GitHub Pages build: the solver runs in the browser."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from cpt_predictor.io.dicom_export import write_dicom_series
from cpt_predictor.reference_leg import synthetic_tibfib_volume


REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPO_ROOT / "web"
SITE_ROOT = Path("/tmp/osteovigil-pages-root")
PAGE_DIR = SITE_ROOT / "OsteoVigil"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_pages_site() -> None:
    if PAGE_DIR.is_symlink():
        PAGE_DIR.unlink()
    SITE_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["npm", "exec", "--", "tsc", "--noEmit"], cwd=WEB_ROOT)
    subprocess.check_call(
        ["npm", "exec", "--", "vite", "build", "--base", "/OsteoVigil/", "--outDir", "/tmp/osteovigil-pages-dist"],
        cwd=WEB_ROOT,
    )
    os.symlink("/tmp/osteovigil-pages-dist", PAGE_DIR)


def _canvas_is_not_flat(page, canvas_id: str) -> bool:
    return bool(
        page.evaluate(
            """(id) => {
              const canvas = document.getElementById(id);
              const context = canvas.getContext("2d");
              const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
              const colors = new Set();
              for (let index = 0; index < data.length; index += 16) {
                colors.add(`${data[index]},${data[index + 1]},${data[index + 2]}`);
                if (colors.size > 2) {
                  return true;
                }
              }
              return false;
            }""",
            canvas_id,
        )
    )


@pytest.mark.skipif(shutil.which("npx") is None, reason="Node is required to build the Pages site.")
def test_pages_build_analyzes_a_dicom_in_the_browser(tmp_path: Path) -> None:
    playwright_sync = pytest.importorskip("playwright.sync_api")
    _build_pages_site()
    hu, spacing = synthetic_tibfib_volume(40, 2.5)
    folder = write_dicom_series(hu, spacing, tmp_path / "series", patient_weight_kg=70.0)
    slices = [str(path) for path in sorted(folder.glob("*.dcm"))]
    port = _free_port()
    process = subprocess.Popen(
        ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(SITE_ROOT)],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with playwright_sync.sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel="chrome",
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            page = browser.new_page()
            page.on("pageerror", lambda error: print(f"PAGEERROR {error}"))
            page.goto(f"http://127.0.0.1:{port}/OsteoVigil/", wait_until="networkidle")
            page.set_input_files("#files", slices)
            page.click("#run")
            page.wait_for_selector("#results:not([hidden])", timeout=240_000)
            assert "CT" in page.locator("#modality").inner_text()
            percent = page.locator("#percent").inner_text()
            assert any(character.isdigit() for character in percent)
            assert _canvas_is_not_flat(page, "ap")
            assert _canvas_is_not_flat(page, "lateral")
            body = page.locator("body").inner_text().lower()
            assert "research biomechanical estimate" in body
            assert "the dicom stays on this computer" in body
            assert "surrogate" not in body
            assert "demo mode" not in body
            browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
