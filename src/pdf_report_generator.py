"""Portable Unicode report using the DejaVu fonts shipped with matplotlib."""

from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.report_generator import Section


def _register_fonts() -> None:
    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in [("ReportSans", "DejaVuSans.ttf"), ("ReportSansBold", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))


def build_pdf_report(sections: list[Section], output_path: Path, figures: Path) -> Path:
    _register_fonts()
    body = ParagraphStyle(
        "Body",
        fontName="ReportSans",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8,
        alignment=TA_LEFT,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=body,
        fontName="ReportSansBold",
        fontSize=17,
        leading=23,
        spaceAfter=14,
        keepWithNext=True,
    )
    cell = ParagraphStyle("Cell", parent=body, fontSize=7.3, leading=10, spaceAfter=0)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=(21 * cm, 29.7 * cm),
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Нефть и рубль: устойчивость связи",
        author="Бойко София; Шумакова Екатерина",
    )
    story = []
    for number, section in enumerate(sections):
        if number:
            story.append(PageBreak())
        story.append(Paragraph(escape(section.title), heading))
        for paragraph in section.paragraphs:
            story.append(Paragraph(escape(paragraph), body))
        for frame in section.tables:
            data = [[Paragraph(escape(str(c)), cell) for c in frame.columns]]
            for row in frame.itertuples(index=False, name=None):
                data.append(
                    [
                        Paragraph(
                            escape(f"{v:.4f}" if isinstance(v, float) and not pd.isna(v) else str(v)),
                            cell,
                        )
                        for v in row
                    ]
                )
            widths = [doc.width / len(frame.columns)] * len(frame.columns)
            if len(frame.columns) > 2:
                widths = [doc.width * 0.34] + [doc.width * 0.66 / (len(frame.columns) - 1)] * (
                    len(frame.columns) - 1
                )
            table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EFF8")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#CBD5E1")),
                    ]
                )
            )
            story.extend([table, Spacer(1, 10)])
        if section.image:
            image = Image(str(figures / section.image))
            ratio = min(doc.width / image.imageWidth, 8.4 * cm / image.imageHeight)
            image.drawWidth = image.imageWidth * ratio
            image.drawHeight = image.imageHeight * ratio
            story.append(image)

    def footer(canvas, document):
        canvas.setFont("ReportSans", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(1.7 * cm, 0.9 * cm, "Нефть и рубль | воспроизводимое исследование")
        canvas.drawRightString(19.3 * cm, 0.9 * cm, str(document.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output_path
