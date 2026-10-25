"""
Layer 3: Semantic Density Analyzer
Computes Information Density Score (IDS) for each segment.
The secret sauce of VidMash.

The IDS formula:
    IDS = (unique_concepts × evidence_strength × novelty) /
          (word_count × repetition_factor)

All component scores are normalised to [0, 1].  A final boost is applied
from linguistic-importance markers.  Segments with an IDS above
``median + 0.5 × std`` are flagged as HIGH VALUE.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.schemas import DensityScore, SemanticSegment

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Linguistic marker dictionaries
# -------------------------------------------------------------------

_HIGH_IMPORTANCE_MARKERS: list[str] = [
    "crucial",
    "important",
    "key",
    "essential",
    "critical",
    "must",
    "never",
    "always",
    "fundamental",
    "vital",
    "mandatory",
    "required",
    "absolutely",
]

_MEDIUM_IMPORTANCE_MARKERS: list[str] = [
    "significant",
    "notable",
    "remember",
    "consider",
    "noteworthy",
    "worth noting",
    "keep in mind",
    "pay attention",
    "don't forget",
    "particularly",
]

_FILLER_WORDS: list[str] = [
    "um",
    "uh",
    "like",
    "you know",
    "basically",
    "actually",
    "literally",
    "sort of",
    "kind of",
    "i mean",
    "right",
    "okay so",
    "so yeah",
    "i guess",
    "whatever",
]

_EVIDENCE_MARKERS: list[str] = [
    "study",
    "studies",
    "research",
    "paper",
    "published",
    "journal",
    "according to",
    "data shows",
    "experiment",
    "evidence",
    "findings",
    "results show",
    "survey",
    "analysis",
    "statistic",
    "citation",
    "professor",
    "dr.",
    "university",
    "institute",
    "report",
    "found that",
    "demonstrated",
    "concluded",
    "peer-reviewed",
]


class DensityAnalyzer:
    """Scores each semantic segment by Information Density.

    High-value segments contain unique concepts, cite evidence, add
    novelty over prior segments, and avoid filler/repetition.
    """

    # TF-IDF concept threshold: only terms with TF-IDF weight above this
    # count as "unique concepts"
    CONCEPT_THRESHOLD: float = 0.05

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix: Any = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        segments: list[SemanticSegment],
    ) -> list[tuple[SemanticSegment, DensityScore]]:
        """Compute IDS for every segment and mark high-value ones.

        Args:
            segments: The semantic segments from Layer 2.

        Returns:
            List of (segment, density_score) tuples, preserving order.
        """
        if not segments:
            return []

        logger.info("Analyzing density for %d segments …", len(segments))

        # Build a single TF-IDF matrix across all segments
        texts = [seg.text for seg in segments]
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10_000,
            min_df=1,
            max_df=0.95,
        )
        try:
            self._tfidf_matrix = self._vectorizer.fit_transform(texts)
            self._feature_names = list(self._vectorizer.get_feature_names_out())
        except ValueError:
            logger.warning("TF-IDF vectorisation failed — falling back to defaults.")
            self._tfidf_matrix = None
            self._feature_names = []

        # Score each segment
        scored: list[tuple[SemanticSegment, DensityScore]] = []
        for idx, seg in enumerate(segments):
            score = self._compute_ids(seg, segments, idx)
            scored.append((seg, score))

        # Mark high-value segments: IDS > median + 0.5 * std
        all_ids = [s.ids_score for _, s in scored]
        if all_ids:
            median = float(np.median(all_ids))
            std = float(np.std(all_ids))
            threshold = median + 0.5 * std
            for _, ds in scored:
                ds.is_high_value = ds.ids_score >= threshold

        high_count = sum(1 for _, ds in scored if ds.is_high_value)
        logger.info(
            "Density analysis complete: %d/%d segments marked HIGH VALUE.",
            high_count,
            len(scored),
        )

        return scored

    # ------------------------------------------------------------------
    # Core IDS computation
    # ------------------------------------------------------------------

    def _compute_ids(
        self,
        segment: SemanticSegment,
        all_segments: list[SemanticSegment],
        segment_idx: int,
    ) -> DensityScore:
        """Compute the Information Density Score for one segment.

        IDS = (unique_concepts × evidence_strength × novelty)
              / (normalised_word_count × repetition_factor)
        boosted by linguistic_importance.

        All individual scores are clamped to [0, 1].

        Args:
            segment: The segment to score.
            all_segments: All segments (for repetition/novelty context).
            segment_idx: Index of the segment in the list.

        Returns:
            A fully populated DensityScore.
        """
        text = segment.text
        word_count = segment.word_count or len(text.split()) or 1

        # --- Component scores ---
        linguistic = self._linguistic_importance(text)
        stat_density = self._statistical_density(text)
        novelty = self._novelty_score(segment_idx)
        entity_rich = self._entity_richness(text)
        repetition = self._repetition_factor(segment_idx, all_segments)
        evidence = self._evidence_strength(text)

        # Unique concepts from TF-IDF
        unique_concepts = self._count_unique_concepts(segment_idx)

        # --- Normalise word_count to [0, 1] (inverse: shorter = denser) ---
        # Use a log scale so very long segments aren't unfairly penalised
        max_expected_words = 1000.0
        norm_word_count = min(
            1.0,
            (math.log1p(word_count) / math.log1p(max_expected_words)),
        )
        # Invert: shorter segments are proportionally denser
        norm_word_count = max(norm_word_count, 0.05)  # avoid division by zero

        # Normalise unique_concepts to [0, 1]
        norm_concepts = min(1.0, unique_concepts / max(50.0, unique_concepts + 1))

        # Clamp all scores
        evidence = min(1.0, max(0.0, evidence))
        novelty = min(1.0, max(0.0, novelty))
        repetition = max(0.1, min(1.0, repetition))  # at least 0.1

        # --- IDS formula ---
        numerator = (norm_concepts + 0.01) * (evidence + 0.01) * (novelty + 0.01)
        denominator = norm_word_count * repetition
        raw_ids = numerator / max(denominator, 0.001)

        # Apply linguistic-importance boost (multiplicative)
        boosted_ids = raw_ids * (1.0 + linguistic * 0.5)

        # Apply stat_density + entity_richness as additive bonus
        boosted_ids += stat_density * 0.1 + entity_rich * 0.1

        # Normalise final score to roughly [0, 1] using sigmoid-like squash
        final_ids = 1.0 / (1.0 + math.exp(-3.0 * (boosted_ids - 0.3)))

        return DensityScore(
            segment_id=segment.id,
            ids_score=round(final_ids, 4),
            unique_concepts=unique_concepts,
            evidence_strength=round(evidence, 4),
            novelty_score=round(novelty, 4),
            repetition_factor=round(repetition, 4),
            linguistic_importance=round(linguistic, 4),
            statistical_density=round(stat_density, 4),
            entity_richness=round(entity_rich, 4),
            is_high_value=False,  # set later in aggregate pass
        )

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    def _linguistic_importance(self, text: str) -> float:
        """Score 0–1 based on importance / filler-word markers.

        High-importance markers boost the score; filler words reduce it.

        Args:
            text: The segment text.

        Returns:
            Score in [0, 1].
        """
        text_lower = text.lower()
        word_count = max(len(text.split()), 1)

        high_hits = sum(1 for m in _HIGH_IMPORTANCE_MARKERS if m in text_lower)
        medium_hits = sum(1 for m in _MEDIUM_IMPORTANCE_MARKERS if m in text_lower)
        filler_hits = sum(1 for f in _FILLER_WORDS if f in text_lower)

        positive = high_hits * 1.0 + medium_hits * 0.5
        negative = filler_hits * 0.3

        # Normalise by word count (importance per word)
        raw = (positive - negative) / (word_count ** 0.5)

        # Clamp to [0, 1]
        return min(1.0, max(0.0, raw))

    def _statistical_density(self, text: str) -> float:
        """Score 0–1 based on presence of numbers, percentages, dates.

        Args:
            text: The segment text.

        Returns:
            Score in [0, 1].
        """
        word_count = max(len(text.split()), 1)

        # Count various numeric patterns
        percentages = len(re.findall(r"\b\d+(?:\.\d+)?%", text))
        dollar_amounts = len(re.findall(r"\$\d[\d,]*(?:\.\d{1,2})?", text))
        plain_numbers = len(re.findall(r"\b\d{2,}\b", text))  # 2+ digit numbers
        date_patterns = len(
            re.findall(
                r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2})",
                text,
                re.IGNORECASE,
            )
        )

        total = percentages * 2 + dollar_amounts * 2 + plain_numbers + date_patterns
        density = total / (word_count ** 0.5)

        return min(1.0, density / 3.0)

    def _novelty_score(self, segment_idx: int) -> float:
        """Measure how much NEW information this segment adds.

        Computes the cosine distance between this segment and the centroid
        of all prior segments.  Higher distance → more novelty.

        Args:
            segment_idx: Index of the segment.

        Returns:
            Novelty score in [0, 1].
        """
        if self._tfidf_matrix is None or segment_idx == 0:
            return 0.5  # First segment is "average" novelty

        # Build centroid of all prior segments
        prior_matrix = self._tfidf_matrix[:segment_idx]
        centroid = prior_matrix.mean(axis=0)
        current = self._tfidf_matrix[segment_idx]

        centroid_arr = np.asarray(centroid)
        current_arr = np.asarray(current.todense()) if hasattr(current, "todense") else np.asarray(current)

        if centroid_arr.ndim == 1:
            centroid_arr = centroid_arr.reshape(1, -1)
        if current_arr.ndim == 1:
            current_arr = current_arr.reshape(1, -1)

        sim = cosine_similarity(current_arr, centroid_arr)[0, 0]

        # Convert similarity to novelty (distance)
        novelty = 1.0 - sim
        return min(1.0, max(0.0, novelty))

    def _entity_richness(self, text: str) -> float:
        """Count named-entity-like tokens using regex heuristics.

        Looks for: capitalised multi-word names, numbers, quoted strings.

        Args:
            text: The segment text.

        Returns:
            Score in [0, 1].
        """
        word_count = max(len(text.split()), 1)

        # Capitalised word sequences (2+ words) — likely proper nouns
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)

        # Single capitalised words not at sentence start
        # (sentences detected by ". X" pattern)
        mid_caps = re.findall(r"(?<!\. )(?<!\.\s)\b[A-Z][a-z]{2,}\b", text)
        # Filter out common sentence-starting patterns
        mid_caps = [w for w in mid_caps if w not in ("The", "This", "That", "And", "But", "For", "With")]

        # Quoted strings
        quoted = re.findall(r'"[^"]{3,}"', text)

        # Large numbers / dollar amounts / percentages
        numbers = re.findall(r"\b\d[\d,.]+\b", text)

        total_entities = len(proper_nouns) + len(mid_caps) * 0.5 + len(quoted) + len(numbers) * 0.3
        richness = total_entities / (word_count ** 0.5)

        return min(1.0, richness / 2.0)

    def _repetition_factor(
        self,
        segment_idx: int,
        segments: list[SemanticSegment],
    ) -> float:
        """Measure how much this segment repeats content from others.

        High average cosine similarity to other segments → high repetition.

        Args:
            segment_idx: Index of the current segment.
            segments: All segments.

        Returns:
            Repetition factor in [0.1, 1.0].  Lower = less repetition.
        """
        if self._tfidf_matrix is None or len(segments) <= 1:
            return 0.5

        current = self._tfidf_matrix[segment_idx]
        current_arr = np.asarray(
            current.todense() if hasattr(current, "todense") else current
        )
        if current_arr.ndim == 1:
            current_arr = current_arr.reshape(1, -1)

        # Compare against all *other* segments
        similarities: list[float] = []
        for i in range(len(segments)):
            if i == segment_idx:
                continue
            other = self._tfidf_matrix[i]
            other_arr = np.asarray(
                other.todense() if hasattr(other, "todense") else other
            )
            if other_arr.ndim == 1:
                other_arr = other_arr.reshape(1, -1)
            sim = cosine_similarity(current_arr, other_arr)[0, 0]
            similarities.append(sim)

        if not similarities:
            return 0.5

        avg_sim = float(np.mean(similarities))

        # Map: high similarity → high repetition factor
        # 0.0 similarity → 0.1 (very unique), 1.0 similarity → 1.0 (total repeat)
        return max(0.1, avg_sim)

    def _evidence_strength(self, text: str) -> float:
        """Score 0–1 based on presence of evidence markers.

        Looks for citations, studies, data references, specific names,
        examples with concrete data.

        Args:
            text: The segment text.

        Returns:
            Score in [0, 1].
        """
        text_lower = text.lower()
        word_count = max(len(text.split()), 1)

        marker_hits = sum(1 for m in _EVIDENCE_MARKERS if m in text_lower)

        # Bonus for specific name mentions (e.g. "Dr. Smith", "Harvard")
        proper_names = len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)\b", text))

        # Bonus for quoted material (direct quotes are strong evidence)
        quotes = len(re.findall(r'"[^"]{10,}"', text))

        total = marker_hits + proper_names * 0.5 + quotes * 1.5
        strength = total / (word_count ** 0.5)

        return min(1.0, strength / 2.0)

    def _count_unique_concepts(self, segment_idx: int) -> int:
        """Count TF-IDF terms above the concept threshold for a segment.

        Args:
            segment_idx: Index of the segment.

        Returns:
            Number of unique concepts (terms with weight > threshold).
        """
        if self._tfidf_matrix is None:
            return 0

        row = self._tfidf_matrix[segment_idx]
        row_arr = np.asarray(
            row.todense() if hasattr(row, "todense") else row
        ).flatten()

        unique = int(np.sum(row_arr > self.CONCEPT_THRESHOLD))
        return unique
