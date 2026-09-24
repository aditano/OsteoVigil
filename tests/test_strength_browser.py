"""Browser check: upload a tiny DICOM and read the strength page."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cpt_predictor.io.dicom_export import write_dicom_series
from cpt_predictor.reference_leg import synthetic_tibfib_volume


REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIST = REPO_ROOT / "web" / "dist" / "index.html"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(port: int, process: subprocess.Popen[bytes]) -> None:
    import urllib.request

    deadline = time.time() + 30
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Website process exited with {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("The strength site did not open /api/health.")


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


@pytest.mark.skipif(not WEB_DIST.is_file(), reason="Build the website with npm run build in web/.")
def test_browser_upload_shows_modality_percent_and_both_views(tmp_path: Path) -> None:
    playwright_sync = pytest.importorskip("playwright.sync_api")
    hu, spacing = synthetic_tibfib_volume(40, 2.5)
    folder = write_dicom_series(hu, spacing, tmp_path / "series", patient_weight_kg=70.0)
    slices = [str(path) for path in sorted(folder.glob("*.dcm"))]
    port = _free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "cpt_predictor.api:app",
            "--app-dir",
            str(REPO_ROOT / "src"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(port, process)
        with playwright_sync.sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel="chrome",
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--enable-unsafe-webgpu"],
            )
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/", wait_until="domcontentloaded")
            page.set_input_files("#files", slices)
            page.click("#run")
            page.wait_for_selector("#results:not([hidden])", timeout=120_000)
            assert "CT" in page.locator("#modality").inner_text()
            percent = page.locator("#percent").inner_text()
            assert any(character.isdigit() for character in percent)
            assert _canvas_is_not_flat(page, "ap")
            assert _canvas_is_not_flat(page, "lateral")
            renderer = page.locator("#renderer").get_attribute("data-renderer")
            assert renderer in {"cpu", "webgpu"}
            body = page.locator("body").inner_text().lower()
            assert "research biomechanical estimate" in body
            assert "not a diagnosis" in body
            assert "surrogate" not in body
            assert "demo mode" not in body
            browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
