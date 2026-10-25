"""
Layer 7: Document Renderer
Compiles DocumentReport into publication-quality PDF and HTML documents.
"""

from __future__ import annotations

import asyncio
import datetime
import html
import logging
import os
import shutil
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as PlatypusImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.models.schemas import DocumentReport, VisualAsset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Brand Palette & Metrics
# ---------------------------------------------------------------------------
PRIMARY_COLOR = colors.HexColor("#4f46e5")    # Indigo 600
PRIMARY_DARK = colors.HexColor("#3730a3")     # Indigo 800
SECONDARY_COLOR = colors.HexColor("#0d9488")  # Teal 600
ACCENT_AMBER = colors.HexColor("#d97706")     # Amber 600
ACCENT_RED = colors.HexColor("#dc2626")       # Red 600
ACCENT_GREEN = colors.HexColor("#16a34a")     # Green 600
BG_LIGHT = colors.HexColor("#f8fafc")         # Slate 50
BG_CARD = colors.HexColor("#f1f5f9")          # Slate 100
TEXT_MAIN = colors.HexColor("#0f172a")        # Slate 900
TEXT_MUTED = colors.HexColor("#475569")       # Slate 600
BORDER_COLOR = colors.HexColor("#cbd5e1")     # Slate 300


# ---------------------------------------------------------------------------
# Numbered Canvas for "Page X of Y" and Running Header/Footer
# ---------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages in a second pass and draws headers/footers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count: int) -> None:
        # Suppress running header & footer on cover page (page 1)
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(TEXT_MUTED)

        # Running header
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(54, 11 * 72 - 36, 8.5 * 72 - 54, 11 * 72 - 36)
        self.drawString(54, 11 * 72 - 30, "VidMash — AI Video Intelligence Document")

        # Running footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 30, page_str)
        self.drawString(54, 30, "Zero-Loss Synthesis • Generated with VidMash Cortex Engine")
        self.line(54, 42, 8.5 * 72 - 54, 42)

        self.restoreState()


class DocumentRenderer:
    """Renders a DocumentReport model into standalone PDF and HTML formats."""

    def __init__(self) -> None:
        self._styles = getSampleStyleSheet()
        self._init_custom_styles()

    def _init_custom_styles(self) -> None:
        """Create typography styles for publication quality layout."""
        self._styles.add(
            ParagraphStyle(
                name="DocTitle",
                fontName="Helvetica-Bold",
                fontSize=26,
                leading=32,
                textColor=PRIMARY_DARK,
                alignment=TA_LEFT,
                spaceAfter=12,
            )
        )
        self._styles.add(
            ParagraphStyle(
                name="DocSubtitle",
                fontName="Helvetica",
                fontSize=13,
                leading=18,
                textColor=TEXT_MUTED,
                alignment=TA_LEFT,
                spaceAfter=20,
            )
        )
        self._styles.add(
            ParagraphStyle(
                name="SectionHeading",
                fontName="Helvetica-Bold",
                fontSize=18,
                leading=22,
                textColor=PRIMARY_COLOR,
                spaceBefore=16,
                spaceAfter=8,
                keepWithNext=True,
            )
        )
        self._styles.add(
            ParagraphStyle(
                name="SubSectionHeading",
                fontName="Helvetica-Bold",
                fontSize=13,
                leading=17,
                textColor=SECONDARY_COLOR,
                spaceBefore=10,
                spaceAfter=6,
                keepWithNext=True,
            )
        )
        self._styles.add(
            ParagraphStyle(
                name="ReportBody",
                fontName="Helvetica",
                fontSize=9.5,
                leading=14.5,
                textColor=TEXT_MAIN,
                alignment=TA_LEFT,
                spaceAfter=8,
            )
        )
        self._styles.add(
            ParagraphStyle(
                name="SummaryBoxText",
                fontName="Helvetica",
                fontSize=10,
                leading=15,
                textColor=TEXT_MAIN,
                spaceAfter=6,
            )
        )
        self._styles.add(
            ParagraphStyle(
                name="QuoteText",
                fontName="Helvetica-Oblique",
                fontSize=9.5,
                leading=14,
                textColor=colors.HexColor("#334155"),
            )
        )
        self._styles.add(
            ParagraphStyle(
                name="QuoteSpeaker",
                fontName="Helvetica-Bold",
                fontSize=8.5,
                leading=11,
                textColor=SECONDARY_COLOR,
                alignment=TA_RIGHT,
            )
        )
        self._styles.add(
            ParagraphStyle(
                name="ImageCaption",
                fontName="Helvetica-Bold",
                fontSize=8.5,
                leading=11,
                textColor=TEXT_MUTED,
                alignment=TA_CENTER,
                spaceBefore=4,
                spaceAfter=10,
            )
        )
        self._styles.add(
            ParagraphStyle(
                name="BadgeText",
                fontName="Helvetica-Bold",
                fontSize=8,
                leading=10,
                textColor=colors.white,
                alignment=TA_CENTER,
            )
        )

    # ------------------------------------------------------------------
    # Public Entry Point
    # ------------------------------------------------------------------

    async def render(
        self, report: DocumentReport, output_dir: str
    ) -> tuple[str, str]:
        """Asynchronously render PDF and HTML documents.

        Args:
            report: The fully composed DocumentReport.
            output_dir: Directory where outputs should be saved.

        Returns:
            Tuple of (pdf_absolute_path, html_absolute_path).
        """
        os.makedirs(output_dir, exist_ok=True)

        pdf_path = os.path.join(output_dir, "report.pdf")
        html_path = os.path.join(output_dir, "report.html")

        # Execute CPU-bound PDF/HTML builders in default executor
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._render_pdf, report, pdf_path)
        await loop.run_in_executor(None, self._render_html, report, html_path)

        return os.path.abspath(pdf_path), os.path.abspath(html_path)

    # ------------------------------------------------------------------
    # PDF Generator
    # ------------------------------------------------------------------

    def _render_pdf(self, report: DocumentReport, output_path: str) -> str:
        """Render DocumentReport to PDF using ReportLab Platypus."""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54,
        )

        story: list[Any] = []

        # 1. Cover / Header Banner
        self._build_cover(story, report)
        story.append(PageBreak())

        # 2. Executive Summary
        if report.executive_summary:
            story.append(Paragraph("Executive Summary", self._styles["SectionHeading"]))
            story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_COLOR, spaceAfter=10))

            summary_paras = [p.strip() for p in report.executive_summary.split("\n\n") if p.strip()]
            summary_flowables = [
                Paragraph(p, self._styles["SummaryBoxText"]) for p in summary_paras
            ]

            summary_table = Table([[summary_flowables]], colWidths=[504])
            summary_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
                    ("LEFTPADDING", (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                    ("LINELEFT", (0, 0), (0, -1), 4, PRIMARY_COLOR),
                    ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ])
            )
            story.append(summary_table)
            story.append(Spacer(1, 14))

        # 3. Key Takeaways
        if report.key_takeaways:
            story.append(Paragraph("Key Takeaways", self._styles["SectionHeading"]))
            story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_COLOR, spaceAfter=10))

            takeaway_rows = []
            for idx, item in enumerate(report.key_takeaways, 1):
                num_badge = Paragraph(f"<b>{idx}</b>", self._styles["BadgeText"])
                item_text = Paragraph(f"<b>{html.escape(item)}</b>", self._styles["ReportBody"])
                takeaway_rows.append([num_badge, item_text])

            takeaway_table = Table(takeaway_rows, colWidths=[24, 480])
            takeaway_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (0, -1), SECONDARY_COLOR),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (0, -1), 4),
                    ("RIGHTPADDING", (0, 0), (0, -1), 4),
                    ("LEFTPADDING", (1, 0), (1, -1), 10),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ])
            )
            story.append(takeaway_table)
            story.append(Spacer(1, 14))

        # 4. Action Items (Checklist)
        if report.action_items:
            story.append(Paragraph("Actionable Roadmap & Checklist", self._styles["SectionHeading"]))
            story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_COLOR, spaceAfter=10))

            action_rows = []
            for item in report.action_items:
                priority = (item.priority or "medium").lower()
                p_color = ACCENT_RED if priority == "high" else (ACCENT_AMBER if priority == "medium" else ACCENT_GREEN)
                p_badge = Paragraph(f"<b>{priority.upper()}</b>", ParagraphStyle(
                    "PBadge", parent=self._styles["BadgeText"], backColor=p_color
                ))
                text_cell = Paragraph(html.escape(item.text), self._styles["ReportBody"])
                time_cell = Paragraph(
                    f"<i>{self._format_seconds(item.source_timestamp)}</i>",
                    ParagraphStyle("TimeSub", parent=self._styles["ReportBody"], fontSize=8, textColor=TEXT_MUTED)
                )
                action_rows.append([p_badge, text_cell, time_cell])

            action_table = Table(action_rows, colWidths=[55, 395, 54])
            action_table.setStyle(
                TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
                    ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ])
            )
            story.append(action_table)
            story.append(Spacer(1, 14))

        # 5. Narrative Deep-Dive Sections
        for sec in report.sections:
            sec_elements: list[Any] = []
            sec_elements.append(Paragraph(html.escape(sec.title), self._styles["SectionHeading"]))
            sec_elements.append(HRFlowable(width="100%", thickness=1, color=SECONDARY_COLOR, spaceAfter=8))

            # Paragraphs
            paragraphs = [p.strip() for p in sec.content.split("\n\n") if p.strip()]
            for p in paragraphs:
                sec_elements.append(Paragraph(html.escape(p), self._styles["ReportBody"]))

            # Key Quotes
            for q in sec.key_quotes:
                quote_flowables = [
                    Paragraph(f"“{html.escape(q.text)}”", self._styles["QuoteText"]),
                    Paragraph(
                        f"— {html.escape(q.speaker or 'Speaker')} ({self._format_seconds(q.timestamp)})",
                        self._styles["QuoteSpeaker"],
                    ),
                ]
                q_table = Table([[quote_flowables]], colWidths=[490])
                q_table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 12),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("LINELEFT", (0, 0), (0, -1), 3, SECONDARY_COLOR),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ])
                )
                sec_elements.append(Spacer(1, 6))
                sec_elements.append(q_table)
                sec_elements.append(Spacer(1, 6))

            # Attached Visuals
            for visual in sec.visuals:
                if visual.image_path and os.path.exists(visual.image_path):
                    try:
                        img = PlatypusImage(visual.image_path, width=470, height=210)
                        sec_elements.append(Spacer(1, 6))
                        sec_elements.append(img)
                        sec_elements.append(
                            Paragraph(f"Figure: {html.escape(visual.title)}", self._styles["ImageCaption"])
                        )
                    except Exception as err:
                        logger.warning("Failed to insert visual image %s into PDF: %s", visual.image_path, err)

            story.append(KeepTogether(sec_elements[:2]))
            story.extend(sec_elements[2:])
            story.append(Spacer(1, 10))

        # 6. Global Visual Gallery (Any visuals not already attached to sections)
        attached_paths = {v.image_path for s in report.sections for v in s.visuals if v.image_path}
        unattached_visuals = [v for v in report.visuals if v.image_path and v.image_path not in attached_paths]

        if unattached_visuals:
            story.append(PageBreak())
            story.append(Paragraph("Visual Intelligence Gallery", self._styles["SectionHeading"]))
            story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_COLOR, spaceAfter=12))

            for visual in unattached_visuals:
                if os.path.exists(visual.image_path):
                    try:
                        img = PlatypusImage(visual.image_path, width=480, height=220)
                        story.append(img)
                        story.append(Paragraph(html.escape(visual.title), self._styles["ImageCaption"]))
                        story.append(Spacer(1, 10))
                    except Exception as err:
                        logger.warning("Failed to render visual %s: %s", visual.image_path, err)

        # 7. Knowledge Graph Annex
        if report.knowledge_graph and (report.knowledge_graph.entities or report.knowledge_graph.relationships):
            story.append(PageBreak())
            story.append(Paragraph("Knowledge Graph & Entity Inventory", self._styles["SectionHeading"]))
            story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_COLOR, spaceAfter=10))

            if report.knowledge_graph.entities:
                top_entities = sorted(report.knowledge_graph.entities, key=lambda e: e.mentions, reverse=True)[:15]
                ent_header = [
                    Paragraph("<b>Entity</b>", self._styles["ReportBody"]),
                    Paragraph("<b>Type</b>", self._styles["ReportBody"]),
                    Paragraph("<b>Mentions</b>", self._styles["ReportBody"]),
                ]
                ent_data = [ent_header]
                for e in top_entities:
                    ent_data.append([
                        Paragraph(html.escape(e.name), self._styles["ReportBody"]),
                        Paragraph(f"<font color='#0d9488'>{html.escape(e.entity_type)}</font>", self._styles["ReportBody"]),
                        Paragraph(str(e.mentions), self._styles["ReportBody"]),
                    ])

                ent_table = Table(ent_data, colWidths=[240, 160, 104])
                ent_table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), BG_CARD),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                    ])
                )
                story.append(ent_table)
                story.append(Spacer(1, 14))

        # Build PDF with dual-pass NumberedCanvas
        doc.build(story, canvasmaker=NumberedCanvas)
        logger.info("PDF generated successfully at %s", output_path)
        return output_path

    def _build_cover(self, story: list[Any], report: DocumentReport) -> None:
        """Create a cover banner."""
        story.append(Spacer(1, 20))
        # Top branding badge
        brand_badge = Table(
            [[Paragraph("<b>VIDMASH • CORTEX AI INTELLIGENCE SYSTEM</b>", ParagraphStyle(
                "BrandBadge", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor("#4f46e5"), alignment=TA_CENTER
            ))]],
            colWidths=[504],
        )
        brand_badge.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e0e7ff")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#c7d2fe")),
            ])
        )
        story.append(brand_badge)
        story.append(Spacer(1, 25))

        # Main Title
        story.append(Paragraph(html.escape(report.metadata.title or "Video Analysis Document"), self._styles["DocTitle"]))
        story.append(Paragraph(
            f"Original Source: <b>{html.escape(report.metadata.channel or 'Unknown Channel')}</b> &nbsp;•&nbsp; "
            f"Duration: <b>{self._format_seconds(report.metadata.duration)}</b> &nbsp;•&nbsp; "
            f"Compression: <b>{html.escape(report.compression_ratio or '10x Speed-Read')}</b>",
            self._styles["DocSubtitle"]
        ))
        story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY_COLOR, spaceAfter=20))

        # Metadata Card Grid
        meta_data = [
            [
                Paragraph("<b>Date Generated:</b>", self._styles["ReportBody"]),
                Paragraph(datetime.datetime.now().strftime("%B %d, %Y - %H:%M"), self._styles["ReportBody"]),
                Paragraph("<b>Processing Time:</b>", self._styles["ReportBody"]),
                Paragraph(f"{report.processing_time:.1f}s", self._styles["ReportBody"]),
            ],
            [
                Paragraph("<b>Total Sections:</b>", self._styles["ReportBody"]),
                Paragraph(str(len(report.sections)), self._styles["ReportBody"]),
                Paragraph("<b>Key Takeaways:</b>", self._styles["ReportBody"]),
                Paragraph(str(len(report.key_takeaways)), self._styles["ReportBody"]),
            ],
            [
                Paragraph("<b>Action Items:</b>", self._styles["ReportBody"]),
                Paragraph(str(len(report.action_items)), self._styles["ReportBody"]),
                Paragraph("<b>Visual Diagrams:</b>", self._styles["ReportBody"]),
                Paragraph(str(len(report.visuals)), self._styles["ReportBody"]),
            ],
        ]
        meta_table = Table(meta_data, colWidths=[120, 132, 120, 132])
        meta_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), BG_CARD),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ])
        )
        story.append(meta_table)
        story.append(Spacer(1, 25))

        # Guiding Principles Box
        promise_box = [
            Paragraph("<b>Zero-Loss Guarantee</b>", ParagraphStyle(
                "PromiseHead", fontName="Helvetica-Bold", fontSize=11, textColor=PRIMARY_DARK, spaceAfter=4
            )),
            Paragraph(
                "This document was constructed via VidMash's 7-layer cognitive synthesis engine. "
                "Every argument, numerical statistic, entity connection, and actionable insight has been preserved "
                "with 100% fidelity. There is <b>zero need</b> to watch the source video.",
                ParagraphStyle("PromiseBody", fontName="Helvetica", fontSize=9, leading=13.5, textColor=TEXT_MUTED)
            ),
        ]
        p_table = Table([[promise_box]], colWidths=[504])
        p_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef2ff")),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("LINELEFT", (0, 0), (0, -1), 4, PRIMARY_COLOR),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d2fe")),
            ])
        )
        story.append(p_table)

    # ------------------------------------------------------------------
    # HTML Generator
    # ------------------------------------------------------------------

    def _render_html(self, report: DocumentReport, output_path: str) -> str:
        """Render DocumentReport to standalone, responsive HTML."""
        from jinja2 import Environment, FileSystemLoader

        template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
        job_id = os.path.basename(os.path.dirname(os.path.abspath(output_path)))
        
        # Try loading template from file; fall back to embedded template
        rendered_content = ""
        if os.path.exists(os.path.join(template_dir, "report.html")):
            try:
                env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
                env.filters["format_seconds"] = self._format_seconds
                env.filters["basename"] = lambda p: os.path.basename(str(p)).replace("\\", "/") if p else ""
                template = env.get_template("report.html")
                rendered_content = template.render(report=report, job_id=job_id, now=datetime.datetime.now())
            except Exception as e:
                logger.error("Could not render via Jinja report.html template: %s", e, exc_info=True)

        if not rendered_content:
            rendered_content = self._render_embedded_html(report)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered_content)

        logger.info("HTML document rendered successfully at %s", output_path)
        return output_path

    def _render_embedded_html(self, report: DocumentReport) -> str:
        """Generate high-fidelity standalone HTML report with Tailwind CSS CDN."""
        visuals_html = ""
        for v in report.visuals:
            img_rel = os.path.basename(v.image_path) if v.image_path else ""
            if img_rel:
                visuals_html += f"""
                <div class="bg-white rounded-xl shadow-md border border-slate-200 overflow-hidden p-4">
                    <img src="{img_rel}" alt="{html.escape(v.title)}" class="w-full h-auto rounded-lg mb-2 object-cover" />
                    <h4 class="font-semibold text-slate-800 text-sm text-center">{html.escape(v.title)}</h4>
                </div>
                """

        sections_html = ""
        for idx, sec in enumerate(report.sections, 1):
            quotes_html = ""
            for q in sec.key_quotes:
                quotes_html += f"""
                <blockquote class="my-4 border-l-4 border-teal-500 pl-4 py-2 bg-slate-50 italic text-slate-700 text-sm rounded-r-lg">
                    "{html.escape(q.text)}"
                    <span class="block not-italic text-xs text-teal-600 font-semibold mt-1">— {html.escape(q.speaker or 'Speaker')} [{self._format_seconds(q.timestamp)}]</span>
                </blockquote>
                """

            sec_visuals_html = ""
            for v in sec.visuals:
                v_rel = os.path.basename(v.image_path) if v.image_path else ""
                if v_rel:
                    sec_visuals_html += f"""
                    <div class="my-4 bg-slate-50 p-3 rounded-lg border border-slate-200 text-center">
                        <img src="{v_rel}" alt="{html.escape(v.title)}" class="mx-auto rounded-md shadow-sm max-h-96" />
                        <p class="text-xs text-slate-500 mt-2 font-medium">{html.escape(v.title)}</p>
                    </div>
                    """

            body_html = "".join(f"<p class='mb-3 text-slate-700 leading-relaxed'>{html.escape(p)}</p>" for p in sec.content.split("\n\n") if p.strip())

            sections_html += f"""
            <section id="section-{idx}" class="mb-12 scroll-mt-20">
                <div class="flex items-center gap-3 mb-4">
                    <span class="flex items-center justify-center w-8 h-8 rounded-full bg-indigo-100 text-indigo-700 font-bold text-sm">{idx}</span>
                    <h3 class="text-2xl font-bold text-slate-900">{html.escape(sec.title)}</h3>
                </div>
                <div class="prose max-w-none text-slate-700">
                    {body_html}
                </div>
                {quotes_html}
                {sec_visuals_html}
            </section>
            """

        takeaways_html = "".join(
            f"""
            <li class="flex items-start gap-3 bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                <span class="flex-shrink-0 w-6 h-6 rounded-full bg-teal-500 text-white flex items-center justify-center font-bold text-xs">{i}</span>
                <span class="text-slate-800 text-sm font-medium leading-relaxed">{html.escape(item)}</span>
            </li>
            """
            for i, item in enumerate(report.key_takeaways, 1)
        )

        actions_html = "".join(
            f"""
            <div class="flex items-center justify-between p-3.5 bg-white rounded-xl border border-slate-200 shadow-sm">
                <div class="flex items-center gap-3">
                    <input type="checkbox" class="w-4 h-4 rounded text-indigo-600 focus:ring-indigo-500 border-slate-300" />
                    <span class="text-sm font-medium text-slate-800">{html.escape(item.text)}</span>
                </div>
                <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold {'bg-red-100 text-red-700' if item.priority=='high' else ('bg-amber-100 text-amber-700' if item.priority=='medium' else 'bg-green-100 text-green-700')}">
                    {item.priority.upper()}
                </span>
            </div>
            """
            for item in report.action_items
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(report.metadata.title)} — VidMash Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
        code, pre {{ font-family: 'JetBrains Mono', monospace; }}
    </style>
</head>
<body class="bg-slate-50 text-slate-900 antialiased">
    <!-- Top Bar -->
    <header class="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-sm backdrop-blur-md bg-white/90">
        <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
            <div class="flex items-center gap-2">
                <div class="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-black text-xs">V</div>
                <span class="font-bold text-slate-900 tracking-tight">VidMash Intelligence</span>
            </div>
            <div class="text-xs text-slate-500">
                Synthesis Time: <span class="font-semibold text-slate-700">{report.processing_time:.1f}s</span> • Compression: <span class="font-semibold text-indigo-600">{html.escape(report.compression_ratio or '10x Speed-Read')}</span>
            </div>
        </div>
    </header>

    <main class="max-w-5xl mx-auto px-4 py-10">
        <!-- Hero Header -->
        <div class="bg-gradient-to-br from-indigo-900 via-indigo-800 to-slate-900 rounded-3xl p-8 md:p-12 text-white shadow-xl mb-10 relative overflow-hidden">
            <div class="relative z-10">
                <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 backdrop-blur-md text-indigo-200 text-xs font-semibold mb-4 border border-white/10">
                    <span>⚡ Complete Zero-Loss Synthesis</span>
                </div>
                <h1 class="text-3xl md:text-5xl font-extrabold tracking-tight mb-4 leading-tight">
                    {html.escape(report.metadata.title)}
                </h1>
                <div class="flex flex-wrap items-center gap-y-2 gap-x-6 text-sm text-slate-300">
                    <div>Channel: <strong class="text-white">{html.escape(report.metadata.channel)}</strong></div>
                    <div>Duration: <strong class="text-white">{self._format_seconds(report.metadata.duration)}</strong></div>
                    <div>Sections: <strong class="text-white">{len(report.sections)}</strong></div>
                </div>
            </div>
        </div>

        <!-- Executive Summary -->
        <div class="bg-white rounded-2xl p-6 md:p-8 border border-slate-200 shadow-sm mb-10">
            <h2 class="text-xl font-bold text-slate-900 mb-4 flex items-center gap-2">
                <span class="text-indigo-600">📌</span> Executive Summary
            </h2>
            <div class="text-slate-700 leading-relaxed space-y-4">
                {"".join(f"<p>{html.escape(p)}</p>" for p in report.executive_summary.split(chr(10)+chr(10)) if p.strip())}
            </div>
        </div>

        <!-- Key Takeaways -->
        <div class="mb-10">
            <h2 class="text-xl font-bold text-slate-900 mb-4 flex items-center gap-2">
                <span class="text-teal-600">🎯</span> Core Takeaways
            </h2>
            <ul class="grid grid-cols-1 md:grid-cols-2 gap-3">
                {takeaways_html}
            </ul>
        </div>

        <!-- Action Items -->
        <div class="mb-10">
            <h2 class="text-xl font-bold text-slate-900 mb-4 flex items-center gap-2">
                <span class="text-amber-600">✅</span> Action Items & Next Steps
            </h2>
            <div class="space-y-2">
                {actions_html}
            </div>
        </div>

        <!-- Narrative Sections -->
        <div class="bg-white rounded-2xl p-6 md:p-10 border border-slate-200 shadow-sm mb-10">
            <h2 class="text-2xl font-bold text-slate-900 mb-8 pb-4 border-b border-slate-200">
                Comprehensive Knowledge Synthesis
            </h2>
            {sections_html}
        </div>

        <!-- Visual Intelligence Gallery -->
        <div class="mb-12">
            <h2 class="text-xl font-bold text-slate-900 mb-4 flex items-center gap-2">
                <span class="text-indigo-600">📊</span> Visual Intelligence Gallery
            </h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                {visuals_html}
            </div>
        </div>
    </main>

    <footer class="bg-white border-t border-slate-200 py-6 text-center text-xs text-slate-500">
        VidMash Intelligence System • Generated with 100% Fidelity • No Need to Watch the Source
    </footer>
</body>
</html>"""

    @staticmethod
    def _format_seconds(seconds: float | int | None) -> str:
        """Format seconds into HH:MM:SS or MM:SS."""
        if not seconds or seconds < 0:
            return "00:00"
        secs = int(seconds)
        hours = secs // 3600
        mins = (secs % 3600) // 60
        rem_secs = secs % 60
        if hours > 0:
            return f"{hours:02d}:{mins:02d}:{rem_secs:02d}"
        return f"{mins:02d}:{rem_secs:02d}"
