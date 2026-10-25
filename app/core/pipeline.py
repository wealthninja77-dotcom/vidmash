"""
VidMash Pipeline Orchestrator.

Coordinates all 7 layers of the Cortex Engine to transform a video URL
into a comprehensive, beautifully formatted document.

Pipeline Layers:
    1. Extraction   — Download metadata and transcript
    2. Segmentation — Split transcript into semantic sections
    3. Density      — Analyse information density via TF-IDF / NLP
    4. Knowledge    — Build an entity-relation knowledge graph
    5. Visuals      — Generate charts, mind-maps, and diagrams
    6. Narrative    — Compose the written report via AI
    7. Renderer     — Produce final PDF and HTML documents
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Callable, Coroutine, Optional

from app.config import get_settings
from app.core.layer1_extraction import ExtractionCortex
from app.core.layer2_segmentation import SegmentationEngine
from app.core.layer3_density import DensityAnalyzer
from app.core.layer4_knowledge import KnowledgeGraphBuilder
from app.core.layer5_visuals import VisualEngine
from app.core.layer6_narrative import NarrativeArchitect
from app.core.layer7_renderer import DocumentRenderer
from app.models.schemas import (
    DocumentReport,
    KnowledgeGraph,
    ProcessingStage,
    SemanticSegment,
    DensityScore,
    VideoMetadata,
)
from app.services.ai_service import AIService

logger = logging.getLogger("vidmash.pipeline")

# Type alias for the optional progress callback
ProgressCallback = Optional[
    Callable[[ProcessingStage, float, str], Coroutine[Any, Any, None]]
]
MetadataCallback = Optional[Callable[[VideoMetadata], None]]


class VidMashPipeline:
    """Orchestrates the 7-layer Cortex Engine pipeline.

    Each layer is executed sequentially. A *progress_callback* function
    (if supplied) is invoked at every stage transition so that the caller
    can update the user-facing job status in real-time.
    """

    def __init__(self) -> None:
        """Initialise pipeline layer instances and shared services."""
        self._settings = get_settings()
        self._ai_service = AIService()
        self._extraction = ExtractionCortex()
        self._segmentation = SegmentationEngine()
        self._density = DensityAnalyzer()
        self._knowledge = KnowledgeGraphBuilder()
        self._visuals = VisualEngine()
        self._narrative = NarrativeArchitect()
        self._renderer = DocumentRenderer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        url: str,
        output_dir: str,
        progress_callback: ProgressCallback = None,
        on_metadata: MetadataCallback = None,
    ) -> tuple[str, str]:
        """Execute the full 7-layer pipeline.

        Args:
            url: YouTube video URL to process.
            output_dir: Directory to write generated artefacts to.
            progress_callback: Optional async callable receiving
                ``(stage, progress_pct, message)`` updates.

        Returns:
            A ``(pdf_path, html_path)`` tuple of absolute paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        start = time.perf_counter()
        layer_timings: dict[str, float] = {}

        async def _report(
            stage: ProcessingStage,
            progress: float,
            message: str,
        ) -> None:
            if progress_callback:
                try:
                    await progress_callback(stage, progress, message)
                except Exception as cb_err:
                    logger.debug("Progress callback error: %s", cb_err)

        # ---- Layer 1: Extraction (0-15%) ---------------------------------
        t0 = time.perf_counter()
        async def _layer1_progress(pct: float, msg: str) -> None:
            await _report(ProcessingStage.EXTRACTING, pct, msg)

        try:
            metadata, transcript_chunks = await self._extraction.extract(
                url, progress_callback=_layer1_progress
            )
            if on_metadata:
                try:
                    on_metadata(metadata)
                except Exception as meta_err:
                    logger.debug("on_metadata error: %s", meta_err)
            logger.info(
                "Layer 1 complete – %d transcript chunks, title: %s",
                len(transcript_chunks),
                metadata.title,
            )
        except Exception as exc:
            logger.error("Layer 1 failed: %s", exc)
            raise RuntimeError(f"Extraction failed: {exc}") from exc
        layer_timings["extraction"] = time.perf_counter() - t0
        await _report(ProcessingStage.EXTRACTING, 15.0, "Extraction complete")

        # ---- Layer 2: Segmentation (15-25%) ------------------------------
        t0 = time.perf_counter()
        await _report(ProcessingStage.SEGMENTING, 18.0, "Detecting semantic topic boundaries...")
        try:
            segments = await self._segmentation.segment(transcript_chunks, metadata)
            logger.info("Layer 2 complete – %d segments", len(segments))
        except Exception as exc:
            logger.error("Layer 2 failed: %s", exc)
            raise RuntimeError(f"Segmentation failed: {exc}") from exc
        layer_timings["segmentation"] = time.perf_counter() - t0
        await _report(ProcessingStage.SEGMENTING, 25.0, "Segmentation complete")

        # ---- Layer 3: Density Analysis (25-40%) --------------------------
        t0 = time.perf_counter()
        await _report(
            ProcessingStage.ANALYZING, 28.0, "Executing Semantic Density Algorithm..."
        )
        try:
            scored_tuples = await self._density.analyze(segments)
            logger.info("Layer 3 complete – %d density scores", len(scored_tuples))
        except Exception as exc:
            logger.warning("Layer 3 failed (non-critical): %s", exc)
            scored_tuples = [(s, DensityScore(segment_id=s.id, ids_score=0.5)) for s in segments]
        layer_timings["density"] = time.perf_counter() - t0
        await _report(
            ProcessingStage.ANALYZING, 40.0, "Density analysis complete"
        )

        # ---- Layer 4: Knowledge Graph (40-55%) ---------------------------
        t0 = time.perf_counter()
        await _report(
            ProcessingStage.BUILDING_GRAPH, 42.0, "Building multi-entity knowledge graph..."
        )
        try:
            knowledge_graph = await self._knowledge.build(scored_tuples, self._ai_service)
            logger.info(
                "Layer 4 complete – %d entities, %d relations",
                len(knowledge_graph.entities),
                len(knowledge_graph.relationships),
            )
        except Exception as exc:
            logger.warning("Layer 4 failed (non-critical): %s", exc)
            knowledge_graph = KnowledgeGraph()
        layer_timings["knowledge"] = time.perf_counter() - t0
        await _report(
            ProcessingStage.BUILDING_GRAPH, 55.0, "Knowledge graph built"
        )

        # Build normalized segment dicts for Layer 5 & 6
        segment_dicts = [self._normalize_segment(st) for st in scored_tuples]

        # ---- Layer 5: Visuals (55-70%) -----------------------------------
        t0 = time.perf_counter()
        await _report(
            ProcessingStage.GENERATING_VISUALS, 58.0, "Generating diagrams, charts, and mind maps..."
        )
        try:
            visual_assets = await self._visuals.generate(
                knowledge_graph, segment_dicts, metadata, output_dir
            )
            logger.info("Layer 5 complete – %d visuals", len(visual_assets))
        except Exception as exc:
            logger.warning("Layer 5 failed (non-critical): %s", exc)
            visual_assets = []
        layer_timings["visuals"] = time.perf_counter() - t0
        await _report(
            ProcessingStage.GENERATING_VISUALS, 70.0, "Visuals generated"
        )

        # ---- Layer 6: Narrative (70-85%) ---------------------------------
        t0 = time.perf_counter()
        await _report(
            ProcessingStage.WRITING_NARRATIVE, 72.0, "Synthesising zero-loss narrative report..."
        )
        try:
            report = await self._narrative.compose(
                scored_segments=segment_dicts,
                knowledge_graph=knowledge_graph,
                visuals=visual_assets,
                metadata=metadata,
                ai_service=self._ai_service,
            )
            logger.info(
                "Layer 6 complete – %d sections, %d takeaways",
                len(report.sections),
                len(report.key_takeaways),
            )
        except Exception as exc:
            logger.error("Layer 6 failed: %s", exc)
            raise RuntimeError(f"Narrative composition failed: {exc}") from exc
        layer_timings["narrative"] = time.perf_counter() - t0
        await _report(
            ProcessingStage.WRITING_NARRATIVE, 85.0, "Narrative composed"
        )

        # ---- Layer 7: Renderer (85-100%) ---------------------------------
        t0 = time.perf_counter()
        await _report(ProcessingStage.RENDERING, 88.0, "Rendering publication-quality PDF & HTML...")

        total_duration = time.perf_counter() - start
        report.processing_time = total_duration

        # Format compression ratio
        video_minutes = metadata.duration / 60 if metadata.duration else 0
        total_words = sum(len(s.text.split()) for s in segments)
        read_minutes = max(1.0, total_words / 250)
        report.compression_ratio = (
            f"{video_minutes:.0f}m video → {read_minutes:.0f}m read"
            if video_minutes > 0
            else "10x Speed-Read"
        )

        try:
            pdf_path, html_path = await self._renderer.render(report, output_dir)
            logger.info("Layer 7 complete – PDF: %s, HTML: %s", pdf_path, html_path)
        except Exception as exc:
            logger.error("Layer 7 failed: %s", exc)
            raise RuntimeError(f"Rendering failed: {exc}") from exc

        # Save structured report JSON and Markdown for API & exports
        report_json_path = os.path.join(output_dir, "report.json")
        try:
            with open(report_json_path, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))
        except Exception as j_err:
            logger.debug("Could not write report.json: %s", j_err)

        report_md_path = os.path.join(output_dir, "report.md")
        try:
            md_content = self._generate_markdown(report)
            with open(report_md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
        except Exception as m_err:
            logger.debug("Could not write report.md: %s", m_err)

        layer_timings["renderer"] = time.perf_counter() - t0

        await _report(ProcessingStage.COMPLETED, 100.0, "Processing complete! Document ready.")

        logger.info(
            "Pipeline finished in %.1fs – layer timings: %s",
            total_duration,
            layer_timings,
        )

        return pdf_path, html_path

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_segment(item: Any) -> dict[str, Any]:
        """Convert a (SemanticSegment, DensityScore) tuple or dict into a standard dict."""
        if isinstance(item, tuple) and len(item) == 2:
            seg, ds = item
            return {
                "id": getattr(seg, "id", 0),
                "text": getattr(seg, "text", ""),
                "start_time": getattr(seg, "start_time", 0.0),
                "start": getattr(seg, "start_time", 0.0),
                "end_time": getattr(seg, "end_time", 0.0),
                "end": getattr(seg, "end_time", 0.0),
                "topic": getattr(seg, "topic", ""),
                "segment_type": getattr(seg, "segment_type", "core_argument"),
                "word_count": getattr(seg, "word_count", 0),
                "ids_score": getattr(ds, "ids_score", 0.5),
                "score": getattr(ds, "ids_score", 0.5),
                "is_high_value": getattr(ds, "is_high_value", False),
                "evidence_strength": getattr(ds, "evidence_strength", 0.0),
                "novelty_score": getattr(ds, "novelty_score", 0.0),
                "entity_richness": getattr(ds, "entity_richness", 0.0),
                "statistical_density": getattr(ds, "statistical_density", 0.0),
            }
        elif isinstance(item, dict):
            return item
        else:
            return {
                "text": getattr(item, "text", str(item)),
                "start_time": getattr(item, "start_time", 0.0),
                "start": getattr(item, "start_time", 0.0),
                "end_time": getattr(item, "end_time", 0.0),
                "end": getattr(item, "end_time", 0.0),
                "topic": getattr(item, "topic", ""),
                "segment_type": getattr(item, "segment_type", "core_argument"),
                "ids_score": getattr(item, "ids_score", 0.5),
                "score": getattr(item, "ids_score", 0.5),
                "is_high_value": getattr(item, "is_high_value", False),
            }


    @staticmethod
    def _generate_markdown(report: DocumentReport) -> str:
        """Render DocumentReport into an academic research paper markdown dossier."""
        lines = [
            f"# {report.metadata.title}",
            "",
            f"**Author / Channel:** {report.metadata.channel}  ",
            f"**Original Duration:** {int(report.metadata.duration // 60)}m {int(report.metadata.duration % 60)}s  ",
            f"**Synthesis Engine:** VidMash Autonomous 7-Layer Cortex Engine  ",
            f"**Classification:** Primary Technical Research Dossier  ",
            f"**Compression Factor:** {report.compression_ratio}  ",
            "",
            "---",
            "",
            "## Abstract",
            "",
            report.executive_summary or "No executive summary available.",
            "",
            "## Core Empirical Takeaways",
            "",
        ]
        for idx, t in enumerate(report.key_takeaways, 1):
            lines.append(f"{idx}. {t}")

        lines.extend(["", "## Technical Deep Dive & Analysis", ""])
        for sec in report.sections:
            start_m, start_s = divmod(int(sec.timestamp_start), 60)
            end_m, end_s = divmod(int(sec.timestamp_end), 60)
            lines.append(f"### §{sec.order}.0 {sec.title} `[{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}]`")
            lines.append("")
            lines.append(sec.content)
            lines.append("")

        if report.action_items:
            lines.extend(["## Actionable Implementation Matrix", ""])
            lines.append("| Priority | Action Item | Source |")
            lines.append("| :--- | :--- | :--- |")
            for item in report.action_items:
                p_badge = f"**{item.priority.upper()}**"
                lines.append(f"| {p_badge} | {item.text} | Technical Findings |")
            lines.append("")

        if report.knowledge_graph.entities:
            lines.extend(["## Knowledge Graph & Entity Ledger", ""])
            top_ents = [f"`{e.name}` ({e.entity_type})" for e in report.knowledge_graph.entities[:15]]
            lines.append("Key entities identified: " + ", ".join(top_ents))
            lines.append("")

        lines.extend([
            "---",
            "",
            "*Synthesized autonomously by VidMash — Watch Less. Know More.*"
        ])
        return "\n".join(lines)
