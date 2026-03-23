from __future__ import annotations

from pathlib import Path
from typing import Dict

from .models import PipelineArtifacts


class PDFReportBuilder:
    def __init__(self, config: Dict):
        self.config = config

    def build(self, artifacts: PipelineArtifacts) -> Path:
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        except Exception:
            fallback = artifacts.output_dir / "cpt_fracture_report.txt"
            fallback.write_text("Report generation requires reportlab.", encoding="utf-8")
            return fallback

        report_path = artifacts.output_dir / "cpt_fracture_report.pdf"
        doc = SimpleDocTemplate(str(report_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        risk = artifacts.risk.summary if artifacts.risk else {}
        patient = self.config["patient"]

        story.append(Paragraph("CPT Fracture Prediction Report", styles["Title"]))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(f"Patient age: {patient['age_years']} years", styles["BodyText"]))
        story.append(Paragraph(f"Diagnosis: {patient['diagnosis']}", styles["BodyText"]))
        story.append(Paragraph(f"Simulation mode: {risk.get('simulation_mode', 'unknown')}", styles["BodyText"]))
        story.append(Spacer(1, 0.15 * inch))

        table_data = [
            ["Metric", "Value"],
            ["Risk category", str(risk.get("risk_category", "unknown")).title()],
            ["Peak von Mises stress (MPa)", f"{risk.get('max_von_mises_mpa', 0.0):.2f}"],
            ["Minimum safety factor", f"{risk.get('min_safety_factor', 0.0):.2f}"],
            ["Estimated years to failure", f"{risk.get('years_to_failure_estimate', 0.0):.2f}"],
            ["Governing gait phase", str(risk.get("governing_phase", "unknown"))],
        ]
        table = Table(table_data, colWidths=[2.6 * inch, 2.8 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

        if artifacts.risk:
            story.append(Paragraph(artifacts.risk.summary["fracture_likely_statement"], styles["BodyText"]))
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph("Recommendations", styles["Heading2"]))
            for recommendation in artifacts.risk.recommendations:
                story.append(Paragraph(f"- {recommendation}", styles["BodyText"]))

        for image_key in ("stress_map", "risk_dashboard"):
            image_path = artifacts.visualization_paths.get(image_key)
            if image_path and Path(image_path).exists():
                story.append(Spacer(1, 0.15 * inch))
                story.append(Image(image_path, width=5.5 * inch, height=3.2 * inch))

        if self.config["reporting"].get("include_disclaimer", True):
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph("Disclaimer", styles["Heading2"]))
            story.append(
                Paragraph(
                    "This report is for research and educational use only and is not a validated medical-device output.",
                    styles["BodyText"],
                )
            )

        doc.build(story)
        return report_path

