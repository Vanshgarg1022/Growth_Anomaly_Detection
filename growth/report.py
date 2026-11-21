import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import io
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_trajectory_plot_png(df_visits: pd.DataFrame) -> io.BytesIO:
    """Generates a 2x2 matplotlib figure of z-scores and returns PNG bytes."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
    indicators = [
        ("waz", "Weight-for-Age (WAZ)", axes[0, 0]),
        ("haz", "Height-for-Age (HAZ)", axes[0, 1]),
        ("bmiz", "BMI-for-Age (BMIZ)", axes[1, 0]),
        ("hcz", "Head Circ-for-Age (HCZ)", axes[1, 1]),
    ]

    for col, title, ax in indicators:
        ax.plot(df_visits["age_months"], df_visits[col], marker="o", linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(2.0, color="orange", linestyle=":", alpha=0.7)
        ax.axhline(-2.0, color="orange", linestyle=":", alpha=0.7)
        ax.axhline(3.0, color="red", linestyle=":", alpha=0.7)
        ax.axhline(-3.0, color="red", linestyle=":", alpha=0.7)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    img_buf.seek(0)
    return img_buf


def generate_patient_pdf_report(res: dict) -> bytes:
    """
    Generates a complete PDF report byte stream for a child's growth analysis.
    Handles both normal analysis and 'Insufficient longitudinal history' cases.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E3A8A"),
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=5,
    )
    normal_style = styles["Normal"]

    story = []

    # Title & Header
    patient_code = res.get("patient_code", "Unknown")
    story.append(Paragraph(f"Paediatric Growth Longitudinal Report", title_style))
    story.append(Paragraph(f"Patient Code: <b>{patient_code}</b> | Total Visits: <b>{res.get('n_visits', 0)}</b>", normal_style))
    story.append(Spacer(1, 10))

    n_vis = res.get("n_visits", 0)

    def _row_table(rows):
        t = Table(rows, colWidths=[2.5 * inch, 4.5 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#E2E8F0")),
            ("TEXTCOLOR", (0, 0), (1, 0), colors.HexColor("#0F172A")),
            ("FONTNAME", (0, 0), (1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    rules = ", ".join(res.get("fired_rules", [])) or "None"
    rf_score = res.get("rf_prob")

    if n_vis < 3:
        story.append(Paragraph("Assessment Status", heading_style))
        story.append(Paragraph(
            "<b>INSUFFICIENT LONGITUDINAL HISTORY</b><br/>"
            "At least 3 valid visits are required to compute trajectory features and the model score. "
            "The conventional WHO cut-off rule below still applies.", normal_style))
        story.append(Spacer(1, 6))
        story.append(_row_table([
            ["Metric", "Value"],
            ["Overall status", res.get("risk_status", "Insufficient history")],
            ["WHO cut-off rule", res.get("baseline_status", "N/A")],
            ["Rules fired", rules],
        ]))
    else:
        story.append(Paragraph("Executive Risk Summary", heading_style))
        story.append(_row_table([
            ["Metric", "Value"],
            ["Overall status", res.get("risk_status", "N/A")],
            ["Alert raised by", res.get("alert_source", "N/A")],
            ["WHO cut-off rule", f"{res.get('baseline_status', 'N/A')} (rules: {rules})"],
            ["Trajectory model score", f"{rf_score:.4f}" if rf_score is not None else "N/A (model unavailable)"],
            ["Model version", res.get("model_version", "N/A")],
        ]))
        story.append(Spacer(1, 10))

        # Contributing Features / Explanation
        story.append(Paragraph("Trajectory Drivers & Explanations", heading_style))
        story.append(Paragraph(str(res.get("explanation", "N/A")).replace("&", "&amp;").replace("<", "&lt;"), normal_style))
        story.append(Spacer(1, 10))

        # Trajectory Plots
        df_v = res.get("df_visits")
        if df_v is not None and not df_v.empty:
            story.append(Paragraph("Longitudinal z-Score Trajectories", heading_style))
            plot_png = generate_trajectory_plot_png(df_v)
            story.append(Image(plot_png, width=6.5 * inch, height=4.0 * inch))
            story.append(Spacer(1, 10))

    # Disclaimer Footer
    story.append(Spacer(1, 15))
    disclaimer_text = (
        "<i>Clinical Disclaimer: This document is a decision-support prototype output based on WHO Child Growth Standards (0-5 years). "
        "The trajectory model was trained on synthetic data and is not clinically validated. "
        "It does not provide a definitive medical diagnosis. All findings require review by a qualified paediatric clinician.</i>"
    )
    story.append(Paragraph(disclaimer_text, normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()