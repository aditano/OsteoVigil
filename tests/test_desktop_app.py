from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

import desktop_app


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def test_desktop_window_starts_with_bundled_demo_ready(qapp):
    window = desktop_app.MainWindow()
    ready, code, message = window._current_input_state()
    assert ready is True
    assert code is None
    assert window.run_btn.isEnabled()
    assert "bundled demo" in message.lower()
    window.close()


def test_desktop_run_button_disabled_without_manual_dicom(qapp):
    window = desktop_app.MainWindow()
    window.demo_check.setChecked(False)
    window.dicom_edit.setText("")
    ready, code, message = window._current_input_state()
    assert ready is False
    assert code == desktop_app.GUI_ERROR_CODES["missing_manual_dicom"]
    assert not window.run_btn.isEnabled()
    window.close()
