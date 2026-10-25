"""
Layer 4: Knowledge Graph Builder
Extracts entities, relationships, and causal chains from high-value segments.

Combines fast regex-based entity extraction with Gemini AI for deeper
semantic understanding.  Produces a ``KnowledgeGraph`` with entities,
relationships, causal chains, and a topic hierarchy.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any

from app.models.schemas import (
    CausalChain,
    DensityScore,
    Entity,
    KnowledgeGraph,
    Relationship,
    SemanticSegment,
)
from app.services.ai_service import AIService

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Builds a knowledge graph from scored semantic segments.

    Prioritises high-value segments for AI-powered extraction but
    includes regex-based entities from all segments for coverage.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build(
        self,
        scored_segments: list[tuple[SemanticSegment, DensityScore]],
        ai_service: AIService,
    ) -> KnowledgeGraph:
        """Build a complete knowledge graph from scored segments.

        Args:
            scored_segments: Output from Layer 3 (segment, density) pairs.
            ai_service: AIService for Gemini calls.

        Returns:
            Populated KnowledgeGraph.
        """
        if not scored_segments:
            return KnowledgeGraph()

        logger.info(
            "Building knowledge graph from %d segments …", len(scored_segments)
        )

        # ----------------------------------------------------------
        # 1. Extract entities from ALL segments (regex — fast)
        # ----------------------------------------------------------
        all_entities: list[Entity] = []
        for seg, _ in scored_segments:
            regex_entities = self._extract_entities_regex(seg.text)
            all_entities.extend(regex_entities)

        # ----------------------------------------------------------
        # 2. AI entity extraction from HIGH-VALUE segments only
        # ----------------------------------------------------------
        high_value = [
            (seg, ds) for seg, ds in scored_segments if ds.is_high_value
        ]
        # If nothing is high value, take the top-3 by IDS
        if not high_value:
            sorted_segs = sorted(
                scored_segments, key=lambda x: x[1].ids_score, reverse=True
            )
            high_value = sorted_segs[:3]

        combined_high_text = "\n\n".join(seg.text for seg, _ in high_value)

        try:
            ai_entities = await self._extract_entities_ai(
                combined_high_text, ai_service
            )
            all_entities.extend(ai_entities)
        except Exception as exc:
            logger.warning("AI entity extraction failed: %s", exc)

        # Deduplicate entities by name (case-insensitive)
        entities = self._deduplicate_entities(all_entities)
        logger.info("Total unique entities: %d", len(entities))

        # ----------------------------------------------------------
        # 3. Find relationships (AI)
        # ----------------------------------------------------------
        relationships: list[Relationship] = []
        if entities:
            try:
                relationships = await self._find_relationships(
                    entities, combined_high_text, ai_service
                )
            except Exception as exc:
                logger.warning("Relationship extraction failed: %s", exc)

        logger.info("Relationships found: %d", len(relationships))

        # ----------------------------------------------------------
        # 4. Detect causal chains (AI)
        # ----------------------------------------------------------
        causal_chains: list[CausalChain] = []
        try:
            causal_chains = await self._detect_causal_chains(
                combined_high_text, ai_service
            )
        except Exception as exc:
            logger.warning("Causal chain detection failed: %s", exc)

        logger.info("Causal chains found: %d", len(causal_chains))

        # ----------------------------------------------------------
        # 5. Build topic hierarchy
        # ----------------------------------------------------------
        topic_hierarchy = self._build_topic_hierarchy(scored_segments)

        return KnowledgeGraph(
            entities=entities,
            relationships=relationships,
            causal_chains=causal_chains,
            topic_hierarchy=topic_hierarchy,
        )

    # ------------------------------------------------------------------
    # Regex-based entity extraction
    # ------------------------------------------------------------------

    def _extract_entities_regex(self, text: str) -> list[Entity]:
        """Extract entities using regex patterns.

        Categories:
        - PERSON: Capitalised two-word sequences (e.g. "John Smith")
        - NUMBER: Digits, percentages, dollar amounts
        - DATE: Date patterns
        - CONCEPT: Repeated capitalised words or multi-word phrases

        Args:
            text: The text to extract entities from.

        Returns:
            List of Entity instances.
        """
        entities: list[Entity] = []

        # --- PERSON: Capitalised two-word sequences ---
        person_pattern = r"\b([A-Z][a-z]{1,20})\s+([A-Z][a-z]{1,20})\b"
        # Common non-person words to filter
        _non_person = {
            "The", "This", "That", "These", "Those", "Here", "There",
            "What", "When", "Where", "Which", "While", "With",
            "After", "Before", "During", "Through", "Between",
            "About", "Above", "Below", "Under", "Over",
            "First", "Second", "Third", "Last", "Next",
            "Each", "Every", "Some", "Many", "Most",
            "Does", "Will", "Would", "Could", "Should",
            "Have", "Has", "Had", "Been", "Being",
            "Very", "More", "Much", "Also", "Even",
            "Just", "Still", "Only", "Already", "Often",
        }
        for match in re.finditer(person_pattern, text):
            first, second = match.group(1), match.group(2)
            if first not in _non_person and second not in _non_person:
                name = f"{first} {second}"
                entities.append(
                    Entity(
                        name=name,
                        entity_type="PERSON",
                        mentions=1,
                        context=text[max(0, match.start() - 30) : match.end() + 30],
                    )
                )

        # --- NUMBER: percentages, dollar amounts, large numbers ---
        pct_pattern = r"\b(\d+(?:\.\d+)?%)\b"
        for match in re.finditer(pct_pattern, text):
            entities.append(
                Entity(
                    name=match.group(1),
                    entity_type="NUMBER",
                    mentions=1,
                    context=text[max(0, match.start() - 20) : match.end() + 20],
                )
            )

        dollar_pattern = r"(\$\d[\d,]*(?:\.\d{1,2})?)"
        for match in re.finditer(dollar_pattern, text):
            entities.append(
                Entity(
                    name=match.group(1),
                    entity_type="NUMBER",
                    mentions=1,
                    context=text[max(0, match.start() - 20) : match.end() + 20],
                )
            )

        large_number_pattern = r"\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?)\b"
        for match in re.finditer(large_number_pattern, text):
            entities.append(
                Entity(
                    name=match.group(1),
                    entity_type="NUMBER",
                    mentions=1,
                    context=text[max(0, match.start() - 20) : match.end() + 20],
                )
            )

        # --- DATE: various date patterns ---
        date_patterns = [
            r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
            r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",
            (
                r"\b((?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?)\b"
            ),
            (
                r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r"\s+\d{1,2}(?:,?\s+\d{4})?)\b"
            ),
        ]
        for dp in date_patterns:
            for match in re.finditer(dp, text, re.IGNORECASE):
                entities.append(
                    Entity(
                        name=match.group(1),
                        entity_type="DATE",
                        mentions=1,
                        context=text[max(0, match.start() - 20) : match.end() + 20],
                    )
                )

        # --- CONCEPT: Capitalised phrases (3+ words, repeated or notable) ---
        concept_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,})\b"
        for match in re.finditer(concept_pattern, text):
            name = match.group(1)
            # Filter out sentence starts
            if match.start() > 0 and text[match.start() - 1] in ".!?\n":
                continue
            entities.append(
                Entity(
                    name=name,
                    entity_type="CONCEPT",
                    mentions=1,
                    context=text[max(0, match.start() - 30) : match.end() + 30],
                )
            )

        # Also detect all-caps acronyms (2-6 chars)
        acronym_pattern = r"\b([A-Z]{2,6})\b"
        _common_acronyms_to_skip = {"TV", "US", "UK", "OK", "AM", "PM", "ID", "IT", "IF", "OR", "SO", "NO"}
        for match in re.finditer(acronym_pattern, text):
            acr = match.group(1)
            if acr not in _common_acronyms_to_skip:
                entities.append(
                    Entity(
                        name=acr,
                        entity_type="CONCEPT",
                        mentions=1,
                        context=text[max(0, match.start() - 20) : match.end() + 20],
                    )
                )

        return entities

    # ------------------------------------------------------------------
    # AI-powered entity extraction
    # ------------------------------------------------------------------

    async def _extract_entities_ai(
        self, text: str, ai_service: AIService
    ) -> list[Entity]:
        """Use Gemini to extract structured entities.

        Args:
            text: Combined high-value segment text.
            ai_service: AIService instance.

        Returns:
            List of Entity instances from AI extraction.
        """
        # Truncate to avoid token limits
        truncated = text[:8000]

        prompt = (
            "You are a knowledge extraction expert. Analyze the following "
            "transcript text and extract all important entities.\n\n"
            "For each entity, identify:\n"
            '- "name": the entity name\n'
            '- "entity_type": one of PERSON, ORGANIZATION, CONCEPT, PRODUCT, '
            "TECHNOLOGY, LOCATION, EVENT, METRIC, THEORY\n"
            '- "mentions": estimated number of times this entity appears '
            "or is referenced\n"
            '- "context": a brief phrase describing the entity\'s role in '
            "the content\n\n"
            "Return ONLY a JSON array of entity objects. "
            "No markdown fences, no explanation.\n\n"
            f"TRANSCRIPT:\n{truncated}"
        )

        raw = await ai_service.generate(prompt)
        return self._parse_entities_json(raw)

    @staticmethod
    def _parse_entities_json(raw: str) -> list[Entity]:
        """Parse JSON entity list from AI response.

        Args:
            raw: Raw AI response string.

        Returns:
            List of Entity instances.
        """
        cleaned = raw.strip()

        # Strip markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines)

        entities: list[Entity] = []
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        entities.append(
                            Entity(
                                name=item["name"],
                                entity_type=item.get("entity_type", "CONCEPT"),
                                mentions=int(item.get("mentions", 1)),
                                context=item.get("context", ""),
                            )
                        )
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI entities JSON: %.200s", cleaned)
            # Regex fallback for partial JSON
            pattern = r'"name"\s*:\s*"([^"]+)".*?"entity_type"\s*:\s*"([^"]*)"'
            for match in re.finditer(pattern, raw, re.DOTALL):
                name, etype = match.groups()
                entities.append(
                    Entity(name=name, entity_type=etype or "CONCEPT")
                )

        return entities

    # ------------------------------------------------------------------
    # Relationship extraction
    # ------------------------------------------------------------------

    async def _find_relationships(
        self,
        entities: list[Entity],
        text: str,
        ai_service: AIService,
    ) -> list[Relationship]:
        """Use Gemini to identify relationships between entities.

        Args:
            entities: Deduplicated entity list.
            text: Combined high-value text.
            ai_service: AIService instance.

        Returns:
            List of Relationship instances.
        """
        # Provide entity names to constrain the AI
        entity_names = [e.name for e in entities[:50]]  # cap at 50
        truncated = text[:6000]

        prompt = (
            "You are a knowledge graph expert. Given the following entities "
            "and transcript text, identify the relationships between entities.\n\n"
            f"ENTITIES: {json.dumps(entity_names)}\n\n"
            f"TRANSCRIPT:\n{truncated}\n\n"
            "For each relationship, provide:\n"
            '- "source": the source entity name (must be from the list)\n'
            '- "target": the target entity name (must be from the list)\n'
            '- "relation_type": a verb phrase describing the relationship '
            '(e.g. "causes", "enables", "is part of", "competes with", '
            '"created by", "leads to", "depends on")\n'
            '- "strength": a float 0.0–1.0 indicating confidence\n\n'
            "Return ONLY a JSON array of relationship objects. "
            "No markdown fences, no explanation. "
            "Return at most 30 relationships."
        )

        raw = await ai_service.generate(prompt)
        return self._parse_relationships_json(raw, entity_names)

    @staticmethod
    def _parse_relationships_json(
        raw: str, valid_entities: list[str]
    ) -> list[Relationship]:
        """Parse JSON relationships from AI response.

        Args:
            raw: Raw AI response.
            valid_entities: List of known entity names for validation.

        Returns:
            List of Relationship instances.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines)

        relationships: list[Relationship] = []
        valid_set = {e.lower() for e in valid_entities}

        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    source = item.get("source", "")
                    target = item.get("target", "")
                    rel_type = item.get("relation_type", "related to")
                    strength = float(item.get("strength", 0.5))

                    # Validate that source and target are known entities
                    if (
                        source
                        and target
                        and source.lower() in valid_set
                        and target.lower() in valid_set
                    ):
                        relationships.append(
                            Relationship(
                                source=source,
                                target=target,
                                relation_type=rel_type,
                                strength=min(1.0, max(0.0, strength)),
                            )
                        )
        except json.JSONDecodeError:
            logger.warning("Failed to parse relationships JSON: %.200s", cleaned)
            # Regex fallback
            pattern = (
                r'"source"\s*:\s*"([^"]+)".*?'
                r'"target"\s*:\s*"([^"]+)".*?'
                r'"relation_type"\s*:\s*"([^"]*)"'
            )
            for match in re.finditer(pattern, raw, re.DOTALL):
                src, tgt, rtype = match.groups()
                if src.lower() in valid_set and tgt.lower() in valid_set:
                    relationships.append(
                        Relationship(source=src, target=tgt, relation_type=rtype)
                    )

        return relationships

    # ------------------------------------------------------------------
    # Causal chain detection
    # ------------------------------------------------------------------

    async def _detect_causal_chains(
        self, text: str, ai_service: AIService
    ) -> list[CausalChain]:
        """Use Gemini to find cause-effect chains in the content.

        Args:
            text: Combined high-value segment text.
            ai_service: AIService instance.

        Returns:
            List of CausalChain instances.
        """
        truncated = text[:6000]

        prompt = (
            "You are a causal reasoning expert. Analyze the following "
            "transcript text and identify all cause-and-effect chains.\n\n"
            "A causal chain is a sequence of events/facts where one leads "
            "to the next. For example:\n"
            '"High interest rates" → "Reduced borrowing" → '
            '"Lower housing demand" → "Price decrease"\n\n'
            f"TRANSCRIPT:\n{truncated}\n\n"
            "For each causal chain, provide:\n"
            '- "steps": an ordered array of short phrases, each step leading '
            "to the next\n"
            '- "description": a one-sentence summary of the overall chain\n\n'
            "Return ONLY a JSON array of causal chain objects. "
            "No markdown fences, no explanation. "
            "Find at most 10 chains."
        )

        raw = await ai_service.generate(prompt)
        return self._parse_causal_chains_json(raw)

    @staticmethod
    def _parse_causal_chains_json(raw: str) -> list[CausalChain]:
        """Parse JSON causal chains from AI response.

        Args:
            raw: Raw AI response.

        Returns:
            List of CausalChain instances.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines)

        chains: list[CausalChain] = []

        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    steps = item.get("steps", [])
                    description = item.get("description", "")

                    if isinstance(steps, list) and len(steps) >= 2:
                        # Ensure all steps are strings
                        str_steps = [str(s) for s in steps if s]
                        if len(str_steps) >= 2:
                            chains.append(
                                CausalChain(
                                    steps=str_steps,
                                    description=str(description),
                                )
                            )
        except json.JSONDecodeError:
            logger.warning("Failed to parse causal chains JSON: %.200s", cleaned)
            # Regex fallback: look for "steps" arrays
            steps_pattern = r'"steps"\s*:\s*\[(.*?)\]'
            desc_pattern = r'"description"\s*:\s*"([^"]*)"'
            for steps_match in re.finditer(steps_pattern, raw, re.DOTALL):
                step_items = re.findall(r'"([^"]+)"', steps_match.group(1))
                if len(step_items) >= 2:
                    # Try to find the corresponding description
                    desc = ""
                    desc_match = re.search(
                        desc_pattern, raw[steps_match.end() : steps_match.end() + 200]
                    )
                    if desc_match:
                        desc = desc_match.group(1)
                    chains.append(CausalChain(steps=step_items, description=desc))

        return chains

    # ------------------------------------------------------------------
    # Topic hierarchy
    # ------------------------------------------------------------------

    def _build_topic_hierarchy(
        self,
        scored_segments: list[tuple[SemanticSegment, DensityScore]],
    ) -> dict[str, Any]:
        """Build a tree: main_topic → subtopics → details.

        The main topic is inferred from the most frequent/highest-scored
        segment topics.  Subtopics are the individual segment topics,
        organised by density score.

        Args:
            scored_segments: All (segment, density) pairs.

        Returns:
            A nested dict representing the topic hierarchy.
        """
        if not scored_segments:
            return {}

        # Collect topics and scores
        topic_scores: dict[str, float] = {}
        topic_texts: dict[str, list[str]] = {}
        topic_types: dict[str, str] = {}

        for seg, ds in scored_segments:
            topic = seg.topic or f"Section {seg.id + 1}"
            if topic in topic_scores:
                topic_scores[topic] = max(topic_scores[topic], ds.ids_score)
                topic_texts[topic].append(seg.text[:100])
            else:
                topic_scores[topic] = ds.ids_score
                topic_texts[topic] = [seg.text[:100]]
                topic_types[topic] = seg.segment_type.value

        # Sort topics by score descending
        sorted_topics = sorted(
            topic_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Main topic = highest scored topic
        main_topic = sorted_topics[0][0] if sorted_topics else "Video Content"

        # Build subtopics
        subtopics: list[dict[str, Any]] = []
        for topic, score in sorted_topics:
            detail_snippets = topic_texts.get(topic, [])
            subtopics.append(
                {
                    "topic": topic,
                    "density_score": round(score, 4),
                    "segment_type": topic_types.get(topic, "core_argument"),
                    "details": detail_snippets,
                }
            )

        hierarchy: dict[str, Any] = {
            "main_topic": main_topic,
            "subtopic_count": len(subtopics),
            "subtopics": subtopics,
        }

        return hierarchy

    # ------------------------------------------------------------------
    # Utility: deduplicate entities
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_entities(entities: list[Entity]) -> list[Entity]:
        """Merge duplicate entities (case-insensitive name match).

        When duplicates are found, mentions are summed and the longest
        context is kept.  AI-provided entity types take precedence over
        regex types.

        Args:
            entities: Raw entity list with potential duplicates.

        Returns:
            Deduplicated entity list sorted by mention count descending.
        """
        registry: dict[str, Entity] = {}
        # Priority order for entity types (higher = preferred)
        type_priority = {
            "PERSON": 5,
            "ORGANIZATION": 5,
            "TECHNOLOGY": 4,
            "PRODUCT": 4,
            "EVENT": 4,
            "THEORY": 4,
            "LOCATION": 3,
            "METRIC": 3,
            "CONCEPT": 2,
            "NUMBER": 1,
            "DATE": 1,
        }

        for ent in entities:
            key = ent.name.lower().strip()
            if not key or len(key) < 2:
                continue

            if key in registry:
                existing = registry[key]
                # Sum mentions
                merged_mentions = existing.mentions + ent.mentions
                # Keep longer context
                best_context = (
                    ent.context
                    if len(ent.context) > len(existing.context)
                    else existing.context
                )
                # Prefer higher-priority type
                existing_priority = type_priority.get(existing.entity_type, 0)
                new_priority = type_priority.get(ent.entity_type, 0)
                best_type = (
                    ent.entity_type
                    if new_priority > existing_priority
                    else existing.entity_type
                )
                # Use the name casing from the entity with more mentions
                best_name = ent.name if ent.mentions > existing.mentions else existing.name

                registry[key] = Entity(
                    name=best_name,
                    entity_type=best_type,
                    mentions=merged_mentions,
                    context=best_context,
                )
            else:
                registry[key] = ent.model_copy()

        # Sort by mentions descending
        result = sorted(registry.values(), key=lambda e: e.mentions, reverse=True)

        return result
