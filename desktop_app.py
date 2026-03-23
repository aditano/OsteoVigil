"""
OsteoVigil Desktop Application
CPT Fracture Prediction System — PyQt6 GUI
"""

import sys
import os
import json
import shutil
import traceback

# Ensure src directory is on the path before any cpt_predictor imports
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QGroupBox, QProgressBar, QTextEdit, QScrollArea,
    QFileDialog, QMessageBox, QFrame, QSizePolicy, QSpacerItem,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QPixmap, QColor, QPalette

DEMO_CASES = {
    "normal_real_talocrural": {
        "label": "Normal distal tibia/fibula CT (real public demo)",
        "description": "Uses the bundled real distal tibia/fibula/ankle DICOM series. No brace STL is included.",
        "dicom_dir": os.path.join(_REPO_ROOT, "data", "demo", "normal_real_talocrural"),
        "brace_stl": None,
    },
    "abnormal_synthetic_cpt": {
        "label": "Abnormal CPT-like CT (synthetic demo)",
        "description": "Uses the bundled abnormal CPT-style demo DICOM series and its included AFO proxy STL.",
        "dicom_dir": os.path.join(_REPO_ROOT, "data", "demo", "abnormal_synthetic_cpt", "dicom"),
        "brace_stl": os.path.join(_REPO_ROOT, "data", "demo", "abnormal_synthetic_cpt", "afo_proxy.stl"),
    },
}

# ---------------------------------------------------------------------------
# Color / theme constants
# ---------------------------------------------------------------------------
BG_DEEP    = "#0d1117"
BG_CARD    = "#161b22"
BG_INPUT   = "#21262d"
BORDER     = "#30363d"
ACCENT     = "#00d4aa"
TEXT_MAIN  = "#e6edf3"
TEXT_MUTED = "#8b949e"
RISK_COLORS = {
    "high":     "#f85149",
    "elevated": "#d29922",
    "moderate": "#388bfd",
    "lower":    "#3fb950",
}

STYLESHEET = f"""
/* ── Global ─────────────────────────────────────────────── */
QWidget {{
    background-color: {BG_DEEP};
    color: {TEXT_MAIN};
    font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}}

/* ── Main window scroll area ─────────────────────────────── */
QScrollArea {{
    border: none;
    background-color: {BG_DEEP};
}}

QScrollBar:vertical {{
    background: {BG_CARD};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ── Group boxes ─────────────────────────────────────────── */
QGroupBox {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
    font-size: 13px;
    color: {TEXT_MAIN};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: -1px;
    padding: 0 6px;
    background-color: {BG_CARD};
    color: {ACCENT};
    font-size: 13px;
    font-weight: 700;
}}

/* ── Line edits ──────────────────────────────────────────── */
QLineEdit {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {TEXT_MAIN};
    font-size: 13px;
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}
QLineEdit::placeholder {{
    color: {TEXT_MUTED};
}}
QLineEdit:disabled {{
    color: {TEXT_MUTED};
}}

/* ── Combo boxes ─────────────────────────────────────────── */
QComboBox {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {TEXT_MAIN};
    font-size: 13px;
}}
QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox:disabled {{
    color: {TEXT_MUTED};
}}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    color: {TEXT_MAIN};
    selection-background-color: {ACCENT};
    selection-color: #0d1117;
}}

/* ── Spin boxes ──────────────────────────────────────────── */
QDoubleSpinBox, QSpinBox {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {TEXT_MAIN};
    font-size: 13px;
    min-width: 90px;
}}
QDoubleSpinBox:focus, QSpinBox:focus {{
    border-color: {ACCENT};
}}
QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    background-color: {BORDER};
    border: none;
    border-radius: 3px;
    width: 18px;
}}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
    background-color: {TEXT_MUTED};
}}

/* ── Check boxes ─────────────────────────────────────────── */
QCheckBox {{
    color: {TEXT_MAIN};
    font-size: 13px;
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background-color: {BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

/* ── Buttons ─────────────────────────────────────────────── */
QPushButton {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 14px;
    color: {TEXT_MAIN};
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {BORDER};
    border-color: {TEXT_MUTED};
}}
QPushButton:pressed {{
    background-color: {BG_DEEP};
}}
QPushButton:disabled {{
    color: {TEXT_MUTED};
    background-color: {BG_INPUT};
    border-color: {BORDER};
}}

QPushButton#primary {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    color: #0d1117;
    font-weight: 700;
    font-size: 15px;
    padding: 10px 28px;
    border-radius: 8px;
}}
QPushButton#primary:hover {{
    background-color: #00bfa0;
    border-color: #00bfa0;
}}
QPushButton#primary:disabled {{
    background-color: #1a3d38;
    border-color: #1a3d38;
    color: {TEXT_MUTED};
}}

QPushButton#export {{
    background-color: {BG_CARD};
    border: 1px solid {ACCENT};
    color: {ACCENT};
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
}}
QPushButton#export:hover {{
    background-color: #0a2520;
}}

/* ── Progress bar ────────────────────────────────────────── */
QProgressBar {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    height: 10px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 5px;
}}

/* ── Text edit (log) ─────────────────────────────────────── */
QTextEdit {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    color: {TEXT_MAIN};
    font-family: "Cascadia Code", "Consolas", "Fira Code", "Courier New", monospace;
    font-size: 12px;
    padding: 8px;
}}

/* ── Labels ──────────────────────────────────────────────── */
QLabel#header_title {{
    color: {ACCENT};
    font-size: 28px;
    font-weight: 800;
}}
QLabel#header_subtitle {{
    color: {TEXT_MUTED};
    font-size: 13px;
}}
QLabel#section_title {{
    color: {TEXT_MAIN};
    font-size: 15px;
    font-weight: 700;
}}
QLabel#muted {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QLabel#metric_value {{
    color: {ACCENT};
    font-size: 22px;
    font-weight: 700;
}}
QLabel#metric_label {{
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 600;
}}
QLabel#stage {{
    color: {TEXT_MUTED};
    font-size: 12px;
    font-style: italic;
}}

/* ── Frames / cards ──────────────────────────────────────── */
QFrame#card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QFrame#divider {{
    background-color: {BORDER};
    max-height: 1px;
    min-height: 1px;
}}
"""


# ---------------------------------------------------------------------------
# Pipeline worker thread
# ---------------------------------------------------------------------------
class PipelineWorker(QThread):
    progress_updated = pyqtSignal(str, float)   # (stage_label, 0-100)
    pipeline_finished = pyqtSignal(object)       # artifacts dict
    pipeline_error = pyqtSignal(str)             # traceback string

    def __init__(
        self,
        dicom_dir: str | None,
        brace_stl: str | None,
        output_dir: str,
        body_mass: float,
        steps_per_day: int,
        use_dummy: bool,
        use_agents: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.dicom_dir = dicom_dir
        self.brace_stl = brace_stl
        self.output_dir = output_dir
        self.body_mass = body_mass
        self.steps_per_day = steps_per_day
        self.use_dummy = use_dummy
        self.use_agents = use_agents

    def _progress_callback(self, stage: str, pct: float):
        self.progress_updated.emit(stage, pct)

    def run(self):
        try:
            # Import inside run() to avoid startup import errors if deps are missing
            from cpt_predictor.pipeline import CPTFracturePipeline

            pipeline = CPTFracturePipeline(
                output_dir=self.output_dir,
                overrides={
                    "input": {"allow_dummy_if_missing": self.use_dummy},
                    "patient": {
                        "body_mass_kg": self.body_mass,
                        "steps_per_day": self.steps_per_day,
                    },
                },
            )

            if self.use_agents:
                from cpt_predictor.agents.crew import PipelineCrewOrchestrator
                artifacts = PipelineCrewOrchestrator(pipeline).run(
                    dicom_dir=self.dicom_dir,
                    brace_stl=self.brace_stl,
                    human_in_the_loop=False,
                    progress=self._progress_callback,
                )
            else:
                artifacts = pipeline.run(
                    dicom_dir=self.dicom_dir,
                    brace_stl=self.brace_stl,
                    use_agents=False,
                    human_in_the_loop=False,
                    progress=self._progress_callback,
                )

            self.pipeline_finished.emit(artifacts)

        except Exception:
            self.pipeline_error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Metric card widget
# ---------------------------------------------------------------------------
class MetricCard(QFrame):
    def __init__(self, label: str, value: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        lbl = QLabel(label.upper())
        lbl.setObjectName("metric_label")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

        val_row = QHBoxLayout()
        val_row.setContentsMargins(0, 0, 0, 0)
        val_row.setSpacing(4)
        val_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        val_lbl = QLabel(value)
        val_lbl.setObjectName("metric_value")
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val_row.addWidget(val_lbl)

        if unit:
            unit_lbl = QLabel(unit)
            unit_lbl.setObjectName("muted")
            unit_lbl.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
            val_row.addWidget(unit_lbl)

        layout.addLayout(val_row)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OsteoVigil — CPT Fracture Prediction")
        self.setMinimumSize(860, 700)
        self.resize(980, 820)

        self._worker: PipelineWorker | None = None
        self._artifacts: dict = {}
        self._manual_dicom_text = ""
        self._manual_brace_text = ""
        self._demo_mode_active = False

        # ── Outer scroll area so the whole window is scrollable ──────────
        outer_scroll = QScrollArea()
        outer_scroll.setWidgetResizable(True)
        outer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCentralWidget(outer_scroll)

        container = QWidget()
        outer_scroll.setWidget(container)

        root = QVBoxLayout(container)
        root.setContentsMargins(32, 28, 32, 32)
        root.setSpacing(20)

        # ── Header ───────────────────────────────────────────────────────
        root.addWidget(self._build_header())
        root.addWidget(self._build_divider())

        # ── Input section ────────────────────────────────────────────────
        root.addWidget(self._build_inputs())

        # ── Run button ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 4)
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setObjectName("primary")
        self.run_btn.setFixedHeight(46)
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addStretch()
        btn_row.addWidget(self.run_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # ── Progress section (hidden initially) ──────────────────────────
        self.progress_frame = self._build_progress()
        self.progress_frame.setVisible(False)
        root.addWidget(self.progress_frame)

        # ── Results section (hidden initially) ───────────────────────────
        self.results_frame = self._build_results_placeholder()
        self.results_frame.setVisible(False)
        root.addWidget(self.results_frame)

        root.addStretch()

    # ── Header ──────────────────────────────────────────────────────────
    def _build_header(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(14)

        # Accent icon block (text-based so it renders on all platforms)
        icon_lbl = QLabel("OV")
        icon_lbl.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        icon_lbl.setFixedSize(52, 52)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet(
            f"background-color: {ACCENT}; color: #0d1117; border-radius: 10px; font-weight: 900;"
        )
        h.addWidget(icon_lbl)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)

        title = QLabel("OsteoVigil")
        title.setObjectName("header_title")
        title_col.addWidget(title)

        sub = QLabel("Computational Periprosthetic Tibial Fracture Prediction System")
        sub.setObjectName("header_subtitle")
        title_col.addWidget(sub)

        h.addLayout(title_col)
        h.addStretch()

        version_lbl = QLabel("v1.0.0")
        version_lbl.setObjectName("muted")
        version_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(version_lbl)

        return w

    def _build_divider(self) -> QFrame:
        d = QFrame()
        d.setObjectName("divider")
        d.setFrameShape(QFrame.Shape.HLine)
        return d

    # ── Inputs ──────────────────────────────────────────────────────────
    def _build_inputs(self) -> QGroupBox:
        box = QGroupBox("Patient & Scan Parameters")
        layout = QVBoxLayout(box)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 20, 16, 16)

        # Input source
        layout.addWidget(self._field_label("Input Source", "Use your own scan or one of the bundled demo cases"))
        source_row = QHBoxLayout()
        source_row.setSpacing(12)
        self.demo_check = QCheckBox("Use bundled demo case")
        self.demo_check.setChecked(True)
        self.demo_check.toggled.connect(self._sync_demo_inputs)
        self.demo_combo = QComboBox()
        for key, case in DEMO_CASES.items():
            self.demo_combo.addItem(case["label"], key)
        self.demo_combo.currentIndexChanged.connect(self._sync_demo_inputs)
        source_row.addWidget(self.demo_check)
        source_row.addWidget(self.demo_combo, 1)
        layout.addLayout(source_row)

        self.demo_info = QLabel()
        self.demo_info.setObjectName("muted")
        self.demo_info.setWordWrap(True)
        layout.addWidget(self.demo_info)

        # DICOM folder
        layout.addWidget(self._field_label("DICOM Folder", "CT scan DICOM series directory"))
        dicom_row = QHBoxLayout()
        self.dicom_edit = QLineEdit()
        self.dicom_edit.setPlaceholderText("Select DICOM folder…")
        self.dicom_browse = QPushButton("Browse…")
        self.dicom_browse.setFixedWidth(90)
        self.dicom_browse.clicked.connect(self._browse_dicom)
        dicom_row.addWidget(self.dicom_edit)
        dicom_row.addWidget(self.dicom_browse)
        layout.addLayout(dicom_row)

        # Brace STL (optional)
        layout.addWidget(self._field_label("Brace STL File", "Optional — tibial brace geometry (.stl)"))
        brace_row = QHBoxLayout()
        self.brace_edit = QLineEdit()
        self.brace_edit.setPlaceholderText("Optional brace geometry file…")
        self.brace_browse = QPushButton("Browse…")
        self.brace_browse.setFixedWidth(90)
        self.brace_browse.clicked.connect(self._browse_brace)
        brace_row.addWidget(self.brace_edit)
        brace_row.addWidget(self.brace_browse)
        layout.addLayout(brace_row)

        # Output directory
        layout.addWidget(self._field_label("Output Directory", "Where results will be saved"))
        out_row = QHBoxLayout()
        self.out_edit = QLineEdit()
        default_out = os.path.join(os.path.expanduser("~"), "OsteoVigil_Results")
        self.out_edit.setText(default_out)
        out_browse = QPushButton("Browse…")
        out_browse.setFixedWidth(90)
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.out_edit)
        out_row.addWidget(out_browse)
        layout.addLayout(out_row)

        # Numeric params row
        num_row = QHBoxLayout()
        num_row.setSpacing(24)

        mass_col = QVBoxLayout()
        mass_col.setSpacing(4)
        mass_col.addWidget(self._field_label("Body Mass", "Patient weight in kg"))
        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(10.0, 200.0)
        self.mass_spin.setValue(55.0)
        self.mass_spin.setSingleStep(1.0)
        self.mass_spin.setSuffix(" kg")
        mass_col.addWidget(self.mass_spin)
        num_row.addLayout(mass_col)

        steps_col = QVBoxLayout()
        steps_col.setSpacing(4)
        steps_col.addWidget(self._field_label("Steps per Day", "Daily activity level"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(500, 30000)
        self.steps_spin.setValue(6000)
        self.steps_spin.setSingleStep(500)
        self.steps_spin.setSuffix(" steps")
        steps_col.addWidget(self.steps_spin)
        num_row.addLayout(steps_col)

        num_row.addStretch()
        layout.addLayout(num_row)

        # Checkboxes
        check_row = QHBoxLayout()
        check_row.setSpacing(28)
        self.agent_check = QCheckBox("Enable in-depth logs")
        self.agent_check.setChecked(True)
        self.agent_check.setToolTip(
            "Shows more detailed stage-by-stage progress labels and writes the workflow manifest to the output folder."
        )
        check_row.addWidget(self.agent_check)
        check_row.addStretch()
        layout.addLayout(check_row)

        self._sync_demo_inputs()
        return box

    def _field_label(self, title: str, hint: str = "") -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        lbl = QLabel(title)
        lbl.setObjectName("section_title")
        lbl.setFont(QFont("Arial", 12, QFont.Weight.DemiBold))
        h.addWidget(lbl)

        if hint:
            hint_lbl = QLabel(f"— {hint}")
            hint_lbl.setObjectName("muted")
            h.addWidget(hint_lbl)

        h.addStretch()
        return w

    # ── Browse callbacks ─────────────────────────────────────────────────
    def _browse_dicom(self):
        path = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if path:
            self.dicom_edit.setText(path)

    def _browse_brace(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Brace STL File", "", "STL Files (*.stl);;All Files (*)")
        if path:
            self.brace_edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.out_edit.setText(path)

    def _selected_demo_case(self) -> dict:
        key = self.demo_combo.currentData()
        return DEMO_CASES[key]

    def _sync_demo_inputs(self, *_args):
        use_demo = self.demo_check.isChecked()

        if use_demo and not self._demo_mode_active:
            self._manual_dicom_text = self.dicom_edit.text()
            self._manual_brace_text = self.brace_edit.text()

        self.demo_combo.setEnabled(use_demo)

        if use_demo:
            demo_case = self._selected_demo_case()
            brace_stl = demo_case["brace_stl"]
            self.dicom_edit.setText(demo_case["dicom_dir"])
            self.dicom_edit.setToolTip(demo_case["dicom_dir"])
            self.brace_edit.setText(brace_stl or "")
            self.brace_edit.setToolTip(brace_stl or "No brace STL included for this demo case.")
            if brace_stl:
                self.brace_edit.setPlaceholderText("Bundled brace STL for the selected demo case")
            else:
                self.brace_edit.setPlaceholderText("No brace STL included for this demo case")
            self.demo_info.setText(demo_case["description"])
        else:
            self.dicom_edit.setText(self._manual_dicom_text)
            self.dicom_edit.setToolTip("")
            self.brace_edit.setText(self._manual_brace_text)
            self.brace_edit.setToolTip("")
            self.brace_edit.setPlaceholderText("Optional brace geometry file…")
            self.demo_info.setText("Select your own DICOM folder and optional brace STL file.")

        for widget in (self.dicom_edit, self.dicom_browse, self.brace_edit, self.brace_browse):
            widget.setEnabled(not use_demo)

        self._demo_mode_active = use_demo

    # ── Progress ─────────────────────────────────────────────────────────
    def _build_progress(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        prog_header = QLabel("Analysis in Progress")
        prog_header.setObjectName("section_title")
        layout.addWidget(prog_header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        layout.addWidget(self.progress_bar)

        self.stage_label = QLabel("Initializing…")
        self.stage_label.setObjectName("stage")
        layout.addWidget(self.stage_label)

        return frame

    # ── Results placeholder (replaced after run) ─────────────────────────
    def _build_results_placeholder(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Content is populated dynamically in _populate_results()
        return frame

    # ── Run logic ────────────────────────────────────────────────────────
    def _on_run(self):
        output_dir = self.out_edit.text().strip() or os.path.join(os.path.expanduser("~"), "OsteoVigil_Results")
        os.makedirs(output_dir, exist_ok=True)

        use_demo = self.demo_check.isChecked()
        if use_demo:
            demo_case = self._selected_demo_case()
            dicom_dir = demo_case["dicom_dir"]
            brace_stl = demo_case["brace_stl"]
            if not os.path.isdir(dicom_dir):
                QMessageBox.warning(
                    self,
                    "Demo Data Missing",
                    f"The selected demo DICOM folder was not found:\n\n{dicom_dir}",
                )
                return
            if brace_stl and not os.path.isfile(brace_stl):
                QMessageBox.warning(
                    self,
                    "Demo Brace Missing",
                    f"The selected demo brace STL was not found:\n\n{brace_stl}",
                )
                return
        else:
            dicom_dir = self.dicom_edit.text().strip() or None
            brace_stl = self.brace_edit.text().strip() or None

        use_agents = self.agent_check.isChecked()

        if not use_demo and not dicom_dir:
            QMessageBox.warning(
                self,
                "No DICOM Folder",
                "Please select a DICOM folder or enable 'Use bundled demo case'.",
            )
            return

        self.run_btn.setEnabled(False)
        self.results_frame.setVisible(False)
        self.progress_frame.setVisible(True)
        self.progress_bar.setValue(0)
        self.stage_label.setText("Starting pipeline…")

        self._worker = PipelineWorker(
            dicom_dir=dicom_dir,
            brace_stl=brace_stl,
            output_dir=output_dir,
            body_mass=self.mass_spin.value(),
            steps_per_day=self.steps_spin.value(),
            use_dummy=False,
            use_agents=use_agents,
        )
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.pipeline_finished.connect(self._on_finished)
        self._worker.pipeline_error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, stage: str, fraction: float):
        self.stage_label.setText(stage)
        # Pipeline emits 0.0–1.0; progress bar expects 0–100
        self.progress_bar.setValue(int(min(max(fraction, 0.0), 1.0) * 100))

    def _on_finished(self, artifacts: dict):
        self._artifacts = artifacts
        self.progress_bar.setValue(100)
        self.stage_label.setText("Complete!")
        self._worker = None
        self.run_btn.setEnabled(True)
        self._populate_results(artifacts)
        self.results_frame.setVisible(True)

    def _on_error(self, tb: str):
        self._worker = None
        self.run_btn.setEnabled(True)
        self.progress_frame.setVisible(False)
        QMessageBox.critical(
            self,
            "Pipeline Error",
            f"An error occurred during the analysis:\n\n{tb}",
        )

    # ── Results population ───────────────────────────────────────────────
    def _populate_results(self, artifacts):
        """Populate the results panel from a PipelineArtifacts dataclass."""
        # Clear previous content from the frame's layout
        old_layout = self.results_frame.layout()
        if old_layout:
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            layout = old_layout
        else:
            layout = QVBoxLayout(self.results_frame)

        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(20)

        # ── Unpack PipelineArtifacts fields ──────────────────────────────
        risk = getattr(artifacts, "risk", None)
        summary = risk.summary if risk else {}
        recommendations = risk.recommendations if risk else []

        simulation = getattr(artifacts, "simulation", None)
        vis_paths = getattr(artifacts, "visualization_paths", {}) or {}
        report_path = getattr(artifacts, "report_path", None)
        output_dir = str(getattr(artifacts, "output_dir", self.out_edit.text().strip()))

        risk_category = summary.get("risk_category", "unknown")
        max_stress = summary.get("max_von_mises_mpa", "—")
        min_sf = summary.get("min_safety_factor", "—")
        years_fail = summary.get("years_to_failure_estimate", "—")

        # Resolve image paths — prefer vis_paths dict, fall back to output_dir
        stress_path = vis_paths.get("stress_map") or os.path.join(output_dir, "stress_map.png")
        dashboard_path = vis_paths.get("risk_dashboard") or os.path.join(output_dir, "risk_dashboard.png")
        pdf_path = str(report_path) if report_path else os.path.join(output_dir, "cpt_fracture_report.pdf")

        def _fmt(v) -> str:
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v) if v != "—" else "—"

        # ── Section title ────────────────────────────────────────────────
        res_title = QLabel("Analysis Results")
        res_title.setObjectName("section_title")
        res_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(res_title)
        layout.addWidget(self._build_divider())

        # ── Risk badge ───────────────────────────────────────────────────
        risk_color = RISK_COLORS.get(risk_category.lower(), TEXT_MUTED)
        badge_row = QHBoxLayout()
        badge_lbl = QLabel(f"  {risk_category.upper()} RISK  ")
        badge_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge_lbl.setFixedHeight(42)
        badge_lbl.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        badge_lbl.setStyleSheet(
            f"background-color: {risk_color}22; color: {risk_color}; "
            f"border: 2px solid {risk_color}; border-radius: 8px; padding: 0 24px;"
        )
        badge_row.addStretch()
        badge_row.addWidget(badge_lbl)
        badge_row.addStretch()
        layout.addLayout(badge_row)

        # Fracture statement beneath badge
        statement = summary.get("fracture_likely_statement", "")
        if statement:
            stmt_lbl = QLabel(statement)
            stmt_lbl.setObjectName("muted")
            stmt_lbl.setWordWrap(True)
            stmt_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(stmt_lbl)

        # ── Metric cards ─────────────────────────────────────────────────
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        cards_row.addWidget(MetricCard("Max Von Mises Stress", _fmt(max_stress), "MPa"))
        cards_row.addWidget(MetricCard("Min Safety Factor", _fmt(min_sf), ""))
        cards_row.addWidget(MetricCard("Years to Failure", _fmt(years_fail), "yrs"))
        layout.addLayout(cards_row)

        # ── Full summary block ───────────────────────────────────────────
        if summary:
            sum_box = QGroupBox("Risk Summary")
            sum_layout = QVBoxLayout(sum_box)
            sum_layout.setContentsMargins(12, 16, 12, 12)
            # Render each key-value pair; skip the long statement (shown above)
            lines = []
            for k, v in summary.items():
                if k == "fracture_likely_statement":
                    continue
                lines.append(f"  {k.replace('_', ' ').title()}:   {v}")
            sum_text = QLabel("\n".join(lines))
            sum_text.setObjectName("muted")
            sum_text.setWordWrap(True)
            sum_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            sum_layout.addWidget(sum_text)
            layout.addWidget(sum_box)

        # ── Output images ────────────────────────────────────────────────
        for img_path, caption in [
            (stress_path, "Stress Map"),
            (dashboard_path, "Risk Dashboard"),
        ]:
            if img_path and os.path.isfile(str(img_path)):
                img_box = QGroupBox(caption)
                img_layout = QVBoxLayout(img_box)
                img_layout.setContentsMargins(8, 12, 8, 8)
                pix = QPixmap(str(img_path))
                if not pix.isNull():
                    img_lbl = QLabel()
                    scaled = pix.scaledToWidth(
                        min(860, pix.width()),
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    img_lbl.setPixmap(scaled)
                    img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    img_layout.addWidget(img_lbl)
                else:
                    img_layout.addWidget(QLabel(f"(Could not load image: {img_path})"))
                layout.addWidget(img_box)

        # ── Recommendations ──────────────────────────────────────────────
        if recommendations:
            rec_box = QGroupBox("Clinical Recommendations")
            rec_layout = QVBoxLayout(rec_box)
            rec_layout.setContentsMargins(12, 16, 12, 12)
            for rec in recommendations:
                rec_lbl = QLabel(f"  \u2022  {rec}")
                rec_lbl.setWordWrap(True)
                rec_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                rec_layout.addWidget(rec_lbl)
            layout.addWidget(rec_box)

        # ── Export buttons ───────────────────────────────────────────────
        export_box = QGroupBox("Export Results")
        export_layout = QHBoxLayout(export_box)
        export_layout.setContentsMargins(12, 16, 12, 12)
        export_layout.setSpacing(12)

        export_pdf_btn = QPushButton("Export PDF Report")
        export_pdf_btn.setObjectName("export")
        export_pdf_btn.clicked.connect(
            lambda: self._export_file(pdf_path, "PDF Report (*.pdf)", "cpt_fracture_report.pdf")
        )

        export_stress_btn = QPushButton("Export Stress Map")
        export_stress_btn.setObjectName("export")
        export_stress_btn.clicked.connect(
            lambda: self._export_file(stress_path, "PNG Image (*.png)", "stress_map.png")
        )

        export_dash_btn = QPushButton("Export Risk Dashboard")
        export_dash_btn.setObjectName("export")
        export_dash_btn.clicked.connect(
            lambda: self._export_file(dashboard_path, "PNG Image (*.png)", "risk_dashboard.png")
        )

        export_layout.addWidget(export_pdf_btn)
        export_layout.addWidget(export_stress_btn)
        export_layout.addWidget(export_dash_btn)
        export_layout.addStretch()
        layout.addWidget(export_box)

        # ── JSON simulation log ──────────────────────────────────────────
        log_box = QGroupBox("Simulation Summary (JSON)")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(8, 12, 8, 8)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setMaximumHeight(380)

        sim_summary = simulation.summary if simulation else {}
        try:
            pretty = json.dumps(
                {**sim_summary, "risk_summary": summary, "output_dir": output_dir},
                indent=2,
                default=str,
            )
        except Exception:
            pretty = str(sim_summary)
        self.log_text.setPlainText(pretty)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_box)

    # ── Export helper ────────────────────────────────────────────────────
    def _export_file(self, src_path: str, file_filter: str, default_name: str):
        if not src_path or not os.path.isfile(str(src_path)):
            QMessageBox.warning(self, "File Not Found", f"Source file not found:\n{src_path}")
            return

        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save File",
            os.path.join(os.path.expanduser("~"), default_name),
            file_filter,
        )
        if dest:
            try:
                shutil.copy(str(src_path), dest)
            except Exception as exc:
                QMessageBox.critical(self, "Export Failed", str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("OsteoVigil")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
