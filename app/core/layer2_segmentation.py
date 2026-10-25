"""
Layer 2: Temporal Segmentation Engine
Splits transcript into semantic segments based on topic coherence.

Uses TF-IDF vectorisation with cosine-similarity boundary detection to
identify natural topic shifts, then classifies each segment by its
rhetorical purpose (introduction, example, definition, etc.).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.schemas import (
    SemanticSegment,
    SegmentType,
    TranscriptChunk,
    VideoMetadata,
)
from app.services.ai_service import AIService

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Keyword patterns for rhetorical-purpose classification
# -------------------------------------------------------------------

_INTRO_KEYWORDS: list[str] = [
    "welcome",
    "today we",
    "in this video",
    "what's up",
    "hey everyone",
    "hi everyone",
    "hello everyone",
    "in today's video",
    "let's talk about",
    "we're going to",
    "i'm going to show",
    "what we'll cover",
    "let's get started",
    "thank you for joining",
    "thanks for tuning in",
]

_CONCLUSION_KEYWORDS: list[str] = [
    "in summary",
    "to wrap up",
    "key takeaway",
    "to summarize",
    "in conclusion",
    "let's recap",
    "wrapping up",
    "that's it for",
    "thanks for watching",
    "thank you for watching",
    "see you in the next",
    "until next time",
    "final thoughts",
    "to sum up",
    "the bottom line",
]

_DEFINITION_KEYWORDS: list[str] = [
    "is defined as",
    "means that",
    "refers to",
    "can be described as",
    "what is",
    "definition of",
    "in other words",
    "put simply",
    "essentially means",
    "which means",
    "the concept of",
    "technically speaking",
]

_STATISTIC_KEYWORDS: list[str] = [
    "percent",
    "percentage",
    "data shows",
    "according to",
    "research shows",
    "studies show",
    "statistics",
    "on average",
    "the number",
    "increased by",
    "decreased by",
    "growth rate",
    "survey",
    "the results",
    "measured",
]

_EXAMPLE_KEYWORDS: list[str] = [
    "for example",
    "for instance",
    "such as",
    "let me show",
    "here's an example",
    "to illustrate",
    "take for example",
    "consider this",
    "let's say",
    "imagine",
    "suppose",
    "case study",
    "a real world example",
    "like when",
    "picture this",
]

_ACTIONABLE_KEYWORDS: list[str] = [
    "you should",
    "make sure",
    "step 1",
    "step 2",
    "step 3",
    "do this",
    "here's what you",
    "try this",
    "i recommend",
    "my advice",
    "pro tip",
    "best practice",
    "first thing",
    "next step",
    "action item",
    "don't forget to",
    "always remember to",
    "you need to",
    "go ahead and",
]

_TANGENT_KEYWORDS: list[str] = [
    "by the way",
    "side note",
    "off topic",
    "quick tangent",
    "speaking of which",
    "on a related note",
    "funny story",
    "random thought",
    "before i forget",
    "quick aside",
    "not related but",
]


class SegmentationEngine:
    """Splits a flat transcript into semantically coherent segments.

    The engine:
    1. Merges tiny transcript chunks into paragraph-sized windows.
    2. Detects topic boundaries using TF-IDF cosine similarity.
    3. Classifies each resulting segment by rhetorical purpose.
    4. Optionally labels segment topics via Gemini AI.
    """

    # Tunables
    WINDOW_SIZE: int = 5  # chunks per paragraph merge
    SIMILARITY_THRESHOLD: float = 0.30  # boundary if similarity drops below
    MIN_SEGMENT_WORDS: int = 20  # minimum words per segment

    def __init__(self) -> None:
        self._ai_service = AIService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def segment(
        self,
        chunks: list[TranscriptChunk],
        metadata: VideoMetadata,
    ) -> list[SemanticSegment]:
        """Segment transcript chunks into semantic segments.

        Args:
            chunks: Raw transcript chunks from Layer 1.
            metadata: Video metadata (used for duration context).

        Returns:
            Ordered list of SemanticSegment instances.
        """
        if not chunks:
            logger.warning("No transcript chunks provided for segmentation.")
            return []

        logger.info("Segmenting %d transcript chunks …", len(chunks))

        # Step 1 — merge tiny chunks into coherent paragraphs
        paragraphs = self._merge_chunks_to_paragraphs(chunks, self.WINDOW_SIZE)
        logger.debug("Merged into %d paragraphs.", len(paragraphs))

        if len(paragraphs) <= 1:
            # Only one paragraph → single segment
            seg = SemanticSegment(
                id=0,
                text=paragraphs[0]["text"] if paragraphs else "",
                start_time=paragraphs[0]["start"] if paragraphs else 0.0,
                end_time=paragraphs[0]["end"] if paragraphs else 0.0,
                word_count=len(paragraphs[0]["text"].split()) if paragraphs else 0,
                segment_type=SegmentType.CORE_ARGUMENT,
            )
            return [seg]

        # Step 2 — detect topic boundaries via TF-IDF cosine similarity
        boundary_indices = self._detect_topic_boundaries(paragraphs)
        logger.debug("Detected %d boundaries.", len(boundary_indices))

        # Step 3 — build segments from boundary slices
        total_duration = metadata.duration or (chunks[-1].end if chunks else 1.0)
        segments = self._build_segments(paragraphs, boundary_indices, total_duration)
        logger.info("Built %d raw segments.", len(segments))

        # Step 4 — label topics via AI (best-effort)
        try:
            segments = await self._label_topics(segments, self._ai_service)
        except Exception as exc:
            logger.warning("AI topic labelling failed: %s — using defaults.", exc)

        return segments

    # ------------------------------------------------------------------
    # Step 1: Merge chunks → paragraphs
    # ------------------------------------------------------------------

    def _merge_chunks_to_paragraphs(
        self,
        chunks: list[TranscriptChunk],
        window_size: int = 5,
    ) -> list[dict[str, Any]]:
        """Merge small transcript chunks into paragraph-sized windows.

        Uses a sliding window of *window_size* chunks (no overlap) and
        concatenates their text.

        Args:
            chunks: The raw transcript chunks.
            window_size: Number of chunks to merge per paragraph.

        Returns:
            List of dicts with keys ``text``, ``start``, ``end``.
        """
        paragraphs: list[dict[str, Any]] = []
        i = 0
        while i < len(chunks):
            window = chunks[i : i + window_size]
            text = " ".join(ch.text for ch in window).strip()
            start_time = window[0].start
            end_time = window[-1].end

            if text:
                paragraphs.append(
                    {
                        "text": text,
                        "start": start_time,
                        "end": end_time,
                    }
                )
            i += window_size

        return paragraphs

    # ------------------------------------------------------------------
    # Step 2: TF-IDF boundary detection
    # ------------------------------------------------------------------

    def _detect_topic_boundaries(self, paragraphs: list[dict[str, Any]]) -> list[int]:
        """Identify topic-shift boundaries using TF-IDF cosine similarity.

        Compares each consecutive pair of paragraphs.  When the cosine
        similarity between two adjacent paragraphs drops below
        ``SIMILARITY_THRESHOLD``, the gap is marked as a boundary.

        Uses a sliding comparison window of 3 paragraphs on each side to
        smooth noisy spikes.

        Args:
            paragraphs: The merged paragraphs.

        Returns:
            Sorted list of paragraph indices where boundaries occur
            (the index is the *start* of the new segment).
        """
        texts = [p["text"] for p in paragraphs]

        # Build TF-IDF matrix
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            min_df=1,
            max_df=0.95,
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # Edge case: all documents are empty / stop-words only
            logger.warning("TF-IDF vectorisation failed — no boundaries detected.")
            return []

        boundaries: list[int] = []
        compare_window = 3  # look-back / look-ahead for smoothing

        for i in range(1, len(paragraphs)):
            # Build aggregated vectors for left and right windows
            left_start = max(0, i - compare_window)
            right_end = min(len(paragraphs), i + compare_window)

            left_vec = tfidf_matrix[left_start:i].mean(axis=0)
            right_vec = tfidf_matrix[i:right_end].mean(axis=0)

            # Convert to 2-D arrays for sklearn
            left_arr = np.asarray(left_vec)
            right_arr = np.asarray(right_vec)

            if left_arr.ndim == 1:
                left_arr = left_arr.reshape(1, -1)
            if right_arr.ndim == 1:
                right_arr = right_arr.reshape(1, -1)

            sim = cosine_similarity(left_arr, right_arr)[0, 0]

            if sim < self.SIMILARITY_THRESHOLD:
                boundaries.append(i)

        return sorted(set(boundaries))

    # ------------------------------------------------------------------
    # Step 3: Build segment objects
    # ------------------------------------------------------------------

    def _build_segments(
        self,
        paragraphs: list[dict[str, Any]],
        boundaries: list[int],
        total_duration: float,
    ) -> list[SemanticSegment]:
        """Create SemanticSegment objects from paragraphs and boundaries.

        Args:
            paragraphs: Merged paragraphs.
            boundaries: Boundary indices.
            total_duration: Total video duration in seconds.

        Returns:
            List of classified SemanticSegment instances.
        """
        # Build split points: [0, b1, b2, ..., len(paragraphs)]
        splits = [0] + boundaries + [len(paragraphs)]
        splits = sorted(set(splits))

        segments: list[SemanticSegment] = []

        for seg_idx, (start_split, end_split) in enumerate(
            zip(splits[:-1], splits[1:])
        ):
            group = paragraphs[start_split:end_split]
            if not group:
                continue

            text = " ".join(p["text"] for p in group).strip()
            if not text:
                continue

            start_time = group[0]["start"]
            end_time = group[-1]["end"]
            word_count = len(text.split())

            # Skip very small segments — merge into previous
            if word_count < self.MIN_SEGMENT_WORDS and segments:
                prev = segments[-1]
                merged_text = prev.text + " " + text
                segments[-1] = prev.model_copy(
                    update={
                        "text": merged_text,
                        "end_time": end_time,
                        "word_count": len(merged_text.split()),
                    }
                )
                continue

            position_ratio = (
                (start_time / total_duration) if total_duration > 0 else 0.0
            )
            seg_type = self._classify_segment_type(text, position_ratio)

            segments.append(
                SemanticSegment(
                    id=seg_idx,
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                    word_count=word_count,
                    segment_type=seg_type,
                )
            )

        # Re-index ids sequentially after potential merges
        for idx, seg in enumerate(segments):
            seg.id = idx

        return segments

    # ------------------------------------------------------------------
    # Segment type classification
    # ------------------------------------------------------------------

    def _classify_segment_type(self, text: str, position_ratio: float) -> SegmentType:
        """Classify a segment's rhetorical purpose using keyword heuristics.

        The classification combines keyword presence with positional context
        (segments near the start are more likely introductions, near the end
        more likely conclusions).

        Args:
            text: The segment's full text.
            position_ratio: 0.0–1.0 indicating position in the video.

        Returns:
            The best-matching SegmentType.
        """
        text_lower = text.lower()

        # Score each type
        scores: dict[SegmentType, float] = {
            SegmentType.INTRODUCTION: 0.0,
            SegmentType.CONCLUSION: 0.0,
            SegmentType.DEFINITION: 0.0,
            SegmentType.STATISTIC: 0.0,
            SegmentType.EXAMPLE: 0.0,
            SegmentType.ACTIONABLE: 0.0,
            SegmentType.TANGENT: 0.0,
            SegmentType.CORE_ARGUMENT: 0.0,
        }

        def _count_hits(keywords: list[str]) -> int:
            return sum(1 for kw in keywords if kw in text_lower)

        # Keyword hits
        scores[SegmentType.INTRODUCTION] += _count_hits(_INTRO_KEYWORDS)
        scores[SegmentType.CONCLUSION] += _count_hits(_CONCLUSION_KEYWORDS)
        scores[SegmentType.DEFINITION] += _count_hits(_DEFINITION_KEYWORDS)
        scores[SegmentType.STATISTIC] += _count_hits(_STATISTIC_KEYWORDS)
        scores[SegmentType.EXAMPLE] += _count_hits(_EXAMPLE_KEYWORDS)
        scores[SegmentType.ACTIONABLE] += _count_hits(_ACTIONABLE_KEYWORDS)
        scores[SegmentType.TANGENT] += _count_hits(_TANGENT_KEYWORDS)

        # Positional boosts
        if position_ratio < 0.10:
            scores[SegmentType.INTRODUCTION] += 2.0
        elif position_ratio > 0.85:
            scores[SegmentType.CONCLUSION] += 2.0

        # Statistical density boost: count numeric tokens
        num_count = len(re.findall(r"\b\d[\d,.]*%?\b", text))
        if num_count >= 3:
            scores[SegmentType.STATISTIC] += 1.5

        # Find the winner (needs at least 1 keyword hit or positional boost)
        best_type = max(scores, key=lambda t: scores[t])
        if scores[best_type] > 0:
            return best_type

        return SegmentType.CORE_ARGUMENT

    # ------------------------------------------------------------------
    # AI topic labelling
    # ------------------------------------------------------------------

    async def _label_topics(
        self,
        segments: list[SemanticSegment],
        ai_service: AIService,
    ) -> list[SemanticSegment]:
        """Use Gemini to assign a concise topic label to each segment.

        Sends all segment texts in a single prompt to reduce API calls
        and maintain cross-segment context.

        Args:
            segments: The segments to label.
            ai_service: AIService instance for Gemini calls.

        Returns:
            The same list with ``topic`` fields populated.
        """
        if not segments:
            return segments

        # Build a numbered preview of each segment (first 200 chars)
        segment_previews: list[str] = []
        for seg in segments:
            preview = seg.text[:200].replace("\n", " ")
            segment_previews.append(f"[Segment {seg.id}]: {preview}")

        prompt = (
            "You are an expert content analyst. Below are numbered segments from "
            "a video transcript. For each segment, provide a short topic label "
            "(3–8 words). Respond ONLY with a JSON array of objects with keys "
            '"id" (int) and "topic" (string). No markdown fences.\n\n'
            + "\n".join(segment_previews)
        )

        try:
            raw_response = await ai_service.generate(prompt)
            topic_labels = self._parse_topic_labels(raw_response, len(segments))

            for seg in segments:
                if seg.id in topic_labels:
                    seg.topic = topic_labels[seg.id]

        except Exception as exc:
            logger.warning("Topic labelling AI call failed: %s", exc)
            # Assign generic topic labels as fallback
            for seg in segments:
                seg.topic = f"Section {seg.id + 1}"

        return segments

    @staticmethod
    def _parse_topic_labels(raw: str, expected_count: int) -> dict[int, str]:
        """Parse the JSON array of topic labels from the AI response.

        Args:
            raw: The raw AI response string.
            expected_count: Number of segments expected.

        Returns:
            Mapping of segment id → topic label.
        """
        import json

        labels: dict[int, str] = {}

        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            lines = [
                ln
                for ln in lines
                if not ln.strip().startswith("```")
            ]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        seg_id = item.get("id")
                        topic = item.get("topic", "")
                        if seg_id is not None and isinstance(topic, str):
                            labels[int(seg_id)] = topic.strip()
        except json.JSONDecodeError:
            logger.warning("Failed to parse topic labels JSON: %s", cleaned[:200])
            # Try regex fallback
            pattern = r'"id"\s*:\s*(\d+)\s*,\s*"topic"\s*:\s*"([^"]+)"'
            for match in re.finditer(pattern, raw):
                seg_id_str, topic = match.groups()
                labels[int(seg_id_str)] = topic.strip()

        return labels
