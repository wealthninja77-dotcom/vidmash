"""
Layer 6: Narrative Architect

Restructures analysed content into an optimal reading order with professional
prose, leveraging the Gemini AI service for each document section.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.models.schemas import (
    ActionItem,
    DocumentReport,
    KeyQuote,
    KnowledgeGraph,
    NarrativeSection,
    VideoMetadata,
    VisualAsset,
)

logger = logging.getLogger(__name__)

# Type alias — avoids circular import of AIService at module level.
# The caller passes a concrete AIService instance at runtime.
AIServiceProto = Any  # app.services.ai_service.AIService


class NarrativeArchitect:
    """Compose a full :class:`DocumentReport` from pipeline artefacts."""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def compose(
        self,
        scored_segments: list[dict[str, Any]],
        knowledge_graph: KnowledgeGraph,
        visuals: list[VisualAsset],
        metadata: VideoMetadata,
        ai_service: AIServiceProto,
    ) -> DocumentReport:
        """Orchestrate all narrative sub-tasks and return a complete report.

        Args:
            scored_segments: Segments with density scores attached.
            knowledge_graph: The knowledge graph from earlier layers.
            visuals: Generated visual assets.
            metadata: Video metadata.
            ai_service: An initialised :class:`AIService` instance.

        Returns:
            A fully populated :class:`DocumentReport`.
        """
        # Build a combined text blob from all segments for the summary
        full_text = "\n\n".join(
            seg.get("text", "") for seg in scored_segments if seg.get("text")
        )

        # Run sub-tasks (some can run in parallel but we keep sequential for
        # deterministic ordering and to respect API rate limits).
        executive_summary = await self._write_executive_summary(
            full_text, metadata, ai_service
        )

        sections = await self._write_sections(
            scored_segments, knowledge_graph, ai_service
        )

        key_takeaways = await self._extract_key_takeaways(full_text, ai_service)
        action_items = await self._extract_action_items(full_text, ai_service)
        key_quotes = await self._extract_key_quotes(full_text, ai_service)

        # Attach relevant visuals to sections (round-robin if more visuals than sections)
        if sections and visuals:
            for idx, visual in enumerate(visuals):
                target = sections[idx % len(sections)]
                target.visuals.append(visual)

        # Attach quotes to the closest section by timestamp
        for quote in key_quotes:
            best_section = self._find_closest_section(quote.timestamp, sections)
            if best_section is not None:
                best_section.key_quotes.append(quote)

        report = DocumentReport(
            metadata=metadata,
            executive_summary=executive_summary,
            sections=sections,
            key_takeaways=key_takeaways,
            action_items=action_items,
            knowledge_graph=knowledge_graph,
            visuals=visuals,
        )
        logger.info(
            "Narrative composition complete: %d sections, %d takeaways, %d action items",
            len(sections),
            len(key_takeaways),
            len(action_items),
        )
        return report

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------

    async def _write_executive_summary(
        self,
        segments_text: str,
        metadata: VideoMetadata,
        ai_service: AIServiceProto,
    ) -> str:
        """Ask Gemini to write a compelling 3-5 paragraph executive summary."""
        system = (
            "You are a professional writer producing a polished executive summary. "
            "Write 3-5 paragraphs of engaging, well-structured prose. "
            "Do NOT use bullet points or numbered lists. "
            "Include specific data points, names, and facts from the transcript. "
            "The summary should give the reader a complete understanding of the "
            "video's content without watching it. Write in third-person present tense."
        )
        prompt = (
            f"Video: \"{metadata.title}\" by {metadata.channel} "
            f"({metadata.duration} seconds).\n\n"
            f"Transcript content:\n{segments_text[:12000]}\n\n"
            "Write a comprehensive executive summary."
        )
        if getattr(ai_service, "is_local_mode", False):
            return self._write_executive_summary_local(segments_text, metadata)

        try:
            return await ai_service.generate(prompt, system_instruction=system)
        except Exception as exc:
            logger.info("AI Executive summary generation unavailable (%s). Switched to local synthesis.", exc)
            return self._write_executive_summary_local(segments_text, metadata)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    async def _write_sections(
        self,
        scored_segments: list[dict[str, Any]],
        knowledge_graph: KnowledgeGraph,
        ai_service: AIServiceProto,
    ) -> list[NarrativeSection]:
        """Group segments by topic and write each section as flowing prose."""
        # Group segments by topic
        topic_groups: dict[str, list[dict[str, Any]]] = {}
        for seg in scored_segments:
            topic = seg.get("topic", "General Discussion")
            topic_groups.setdefault(topic, []).append(seg)

        # Sort groups so high-value content comes first
        def _group_importance(segs: list[dict[str, Any]]) -> float:
            return max(
                (
                    float(s.get("ids_score", s.get("score", s.get("importance", 0.5))))
                    for s in segs
                ),
                default=0.0,
            )

        sorted_topics = sorted(
            topic_groups.items(),
            key=lambda kv: _group_importance(kv[1]),
            reverse=True,
        )

        # Build entity context for richer prose
        entity_context = ", ".join(
            f"{e.name} ({e.entity_type})" for e in knowledge_graph.entities[:20]
        )

        sections: list[NarrativeSection] = []
        for order, (topic, segs) in enumerate(sorted_topics, start=1):
            combined_text = "\n".join(s.get("text", "") for s in segs)
            if not combined_text.strip():
                continue

            start_time = min(
                float(s.get("start_time", s.get("start", 0))) for s in segs
            )
            end_time = max(
                float(s.get("end_time", s.get("end", 0))) for s in segs
            )

            system = (
                "You are a professional long-form writer. Rewrite the provided "
                "transcript excerpt as a polished, publication-quality section of "
                "a written report. Rules:\n"
                "• Write in flowing, professional prose — NO bullet points.\n"
                "• Maintain the original speaker's intent and accuracy.\n"
                "• Include specific data points, names, numbers, and facts.\n"
                "• Organise the content logically, not necessarily chronologically.\n"
                "• The reader should need ZERO reason to watch the original video.\n"
                "• Write 2-6 paragraphs depending on content depth."
            )
            prompt = (
                f"Section topic: {topic}\n"
                f"Key entities mentioned: {entity_context}\n\n"
                f"Transcript excerpt:\n{combined_text[:8000]}\n\n"
                "Rewrite as a polished report section."
            )

            if getattr(ai_service, "is_local_mode", False):
                content = self._format_section_local(combined_text, topic)
            else:
                try:
                    content = await ai_service.generate(prompt, system_instruction=system)
                except Exception as exc:
                    logger.info("AI section writing unavailable (%s) for '%s'. Switched to local formatting.", exc, topic)
                    content = self._format_section_local(combined_text, topic)

            sections.append(
                NarrativeSection(
                    title=topic,
                    content=content,
                    section_type="body",
                    order=order,
                    timestamp_start=start_time,
                    timestamp_end=end_time,
                )
            )

        return sections

    # ------------------------------------------------------------------
    # Takeaways
    # ------------------------------------------------------------------

    async def _extract_key_takeaways(
        self, text: str, ai_service: AIServiceProto
    ) -> list[str]:
        """Ask Gemini for 5-10 key takeaways from the transcript."""
        system = (
            "You are a content analyst. Extract 5-10 key takeaways from the "
            "transcript. Each takeaway should be a single, self-contained "
            "sentence. Return a JSON array of strings. No markdown fences."
        )
        prompt = f"Transcript:\n{text[:12000]}\n\nExtract key takeaways."

        if getattr(ai_service, "is_local_mode", False):
            return self._extract_key_takeaways_local(text)

        try:
            raw = await ai_service.generate(prompt, system_instruction=system, json_mode=True)
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(t) for t in parsed[:10]]
            if isinstance(parsed, dict):
                for key in ("takeaways", "key_takeaways", "items"):
                    if key in parsed and isinstance(parsed[key], list):
                        return [str(t) for t in parsed[key][:10]]
            return [str(parsed)]
        except Exception as exc:
            logger.info("AI Key takeaway extraction unavailable (%s). Switched to local heuristic extraction.", exc)
            return self._extract_key_takeaways_local(text)

    # ------------------------------------------------------------------
    # Action items
    # ------------------------------------------------------------------

    async def _extract_action_items(
        self, text: str, ai_service: AIServiceProto
    ) -> list[ActionItem]:
        """Ask Gemini to find actionable items in the transcript."""
        system = (
            "You are a productivity analyst. Identify actionable items from "
            "the transcript — things the viewer should DO after watching. "
            "Return a JSON array of objects with keys: text, priority "
            "(high/medium/low). No markdown fences."
        )
        prompt = f"Transcript:\n{text[:12000]}\n\nExtract action items."

        if getattr(ai_service, "is_local_mode", False):
            return self._extract_action_items_local(text)

        try:
            raw = await ai_service.generate(prompt, system_instruction=system, json_mode=True)
            parsed = json.loads(raw)
            items_list: list[dict] = []
            if isinstance(parsed, list):
                items_list = parsed
            elif isinstance(parsed, dict):
                for key in ("action_items", "items", "actions"):
                    if key in parsed and isinstance(parsed[key], list):
                        items_list = parsed[key]
                        break

            return [
                ActionItem(
                    text=str(item.get("text", item) if isinstance(item, dict) else item),
                    priority=str(item.get("priority", "medium")) if isinstance(item, dict) else "medium",
                )
                for item in items_list[:15]
            ]
        except Exception as exc:
            logger.info("AI Action item extraction unavailable (%s). Switched to local extractor.", exc)
            return self._extract_action_items_local(text)

    # ------------------------------------------------------------------
    # Key quotes
    # ------------------------------------------------------------------

    async def _extract_key_quotes(
        self, text: str, ai_service: AIServiceProto
    ) -> list[KeyQuote]:
        """Ask Gemini to find notable, quotable passages."""
        system = (
            "You are a researcher selecting the most impactful, quotable "
            "statements from a video transcript. Find 3-8 quotes that are "
            "insightful, surprising, or memorable. Return a JSON array of "
            "objects with keys: text, speaker (infer if possible, else "
            "'Speaker'), timestamp (approximate seconds, float). "
            "No markdown fences."
        )
        prompt = f"Transcript:\n{text[:12000]}\n\nExtract key quotes."

        if getattr(ai_service, "is_local_mode", False):
            return self._extract_key_quotes_local(text)

        try:
            raw = await ai_service.generate(prompt, system_instruction=system, json_mode=True)
            parsed = json.loads(raw)
            quotes_list: list[dict] = []
            if isinstance(parsed, list):
                quotes_list = parsed
            elif isinstance(parsed, dict):
                for key in ("quotes", "key_quotes", "items"):
                    if key in parsed and isinstance(parsed[key], list):
                        quotes_list = parsed[key]
                        break

            return [
                KeyQuote(
                    text=str(q.get("text", "")),
                    speaker=str(q.get("speaker", "Speaker")),
                    timestamp=float(q.get("timestamp", 0.0)),
                )
                for q in quotes_list[:8]
                if isinstance(q, dict) and q.get("text")
            ]
        except Exception as exc:
            logger.info("AI Key quote extraction unavailable (%s). Switched to local quote extractor.", exc)
            return self._extract_key_quotes_local(text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_closest_section(
        timestamp: float, sections: list[NarrativeSection]
    ) -> NarrativeSection | None:
        """Return the section whose time window is closest to *timestamp*."""
        if not sections:
            return None

        best: NarrativeSection | None = None
        best_dist = float("inf")
        for sec in sections:
            if sec.timestamp_start <= timestamp <= sec.timestamp_end:
                return sec
            mid = (sec.timestamp_start + sec.timestamp_end) / 2
            dist = abs(timestamp - mid)
            if dist < best_dist:
                best_dist = dist
                best = sec
        return best


    # ------------------------------------------------------------------
    # High-fidelity Local Synthesis Engine
    # ------------------------------------------------------------------

    def _write_executive_summary_local(self, segments_text: str, metadata: VideoMetadata) -> str:
        """Synthesize a polished multi-paragraph executive summary from transcript analysis."""
        import re

        clean_text = re.sub(r"\s+", " ", segments_text).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if len(s.strip()) > 35]

        duration_min = round(metadata.duration / 60, 1) if metadata.duration else "unspecified duration"
        p1 = (
            f"This comprehensive executive dossier synthesises the core subject matter of "
            f"\"{metadata.title}\", presented by {metadata.channel} ({duration_min} minutes). "
            f"The presentation delivers key technical, architectural, and operational findings, "
            f"focusing on foundational methodologies, workflow mechanisms, and actionable outcomes."
        )

        mid_sentences = sentences[:12] if len(sentences) >= 12 else sentences
        key_points = []
        for s in mid_sentences:
            if any(term in s.lower() for term in ["build", "new", "feature", "system", "support", "using", "able", "allow"]):
                key_points.append(s)
            if len(key_points) >= 3:
                break
        if not key_points and sentences:
            key_points = sentences[:3]

        p2 = (
            f"Central to the presentation is an in-depth exploration of core components and implementations. "
            f"{' '.join(key_points)} "
            f"Through rigorous analysis, the content clarifies how specific parameters and system interactions "
            f"directly dictate end results and operational efficacy."
        )

        tail_sentences = sentences[-8:] if len(sentences) >= 8 else sentences
        concl_points = [s for s in tail_sentences if len(s) > 40][:2]
        p3 = (
            f"In conclusion, the discourse highlights strategic implications for practitioners. "
            f"{' '.join(concl_points)} "
            f"By implementing these insights, teams and individual creators can achieve "
            f"streamlined performance with zero information loss."
        )

        return f"{p1}\n\n{p2}\n\n{p3}"

    def _format_section_local(self, text: str, topic: str) -> str:
        """Format raw segment transcript into readable, structured paragraphs."""
        import re

        # Clean filler words
        cleaned = re.sub(r"\b(um|uh|you know|sort of|kind of|like I said)\b", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) > 15]

        if not sentences:
            return cleaned or f"Discussion concerning {topic}."

        paragraphs: list[str] = []
        chunk_size = 4
        for i in range(0, len(sentences), chunk_size):
            para = " ".join(sentences[i:i + chunk_size])
            if para:
                paragraphs.append(para)

        return "\n\n".join(paragraphs[:6])

    def _extract_key_takeaways_local(self, text: str) -> list[str]:
        """Extract high-density, informative sentences as key takeaways."""
        import re

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]
        scored: list[tuple[float, str]] = []
        signal_words = ["important", "key", "main", "feature", "build", "system", "allows", "enables", "result", "shows", "percent", "%"]

        for s in sentences:
            score = 0.0
            s_low = s.lower()
            for w in signal_words:
                if w in s_low:
                    score += 1.0
            if any(char.isdigit() for char in s):
                score += 1.5
            if 50 <= len(s) <= 160:
                score += 1.0
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        takeaways = [s for _, s in scored[:7]]
        return takeaways if takeaways else ["The presentation covers key architectural principles and operational techniques."]

    def _extract_action_items_local(self, text: str) -> list[ActionItem]:
        """Extract actionable imperatives locally from transcript sentences."""
        import re

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]
        action_patterns = [
            (r"\b(make sure to|ensure that|always|need to|must)\b", "high"),
            (r"\b(recommend|can use|you can|try to|suggest)\b", "medium"),
            (r"\b(to get started|first step|check out|install|configure)\b", "high"),
        ]

        items: list[ActionItem] = []
        seen = set()

        for s in sentences:
            for pat, priority in action_patterns:
                if re.search(pat, s, re.IGNORECASE) and s not in seen:
                    seen.add(s)
                    items.append(ActionItem(text=s[:180], priority=priority))
                    break
            if len(items) >= 8:
                break

        if not items and sentences:
            items.append(ActionItem(text=f"Review the primary techniques discussed: {sentences[0][:140]}", priority="medium"))

        return items

    def _extract_key_quotes_local(self, text: str) -> list[KeyQuote]:
        """Extract notable memorable passages locally."""
        import re

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 35]
        quotes: list[KeyQuote] = []
        for idx, s in enumerate(sentences):
            if any(marker in s.lower() for marker in ["believe", "really think", "exciting", "the point is", "what we found", "crucial", "fundamental"]):
                quotes.append(
                    KeyQuote(
                        text=s,
                        speaker="Presenter",
                        timestamp=float(idx * 15),
                    )
                )
            if len(quotes) >= 5:
                break

        if not quotes and sentences:
            quotes.append(KeyQuote(text=sentences[0], speaker="Presenter", timestamp=0.0))

        return quotes
