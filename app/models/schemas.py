"""
Comprehensive Pydantic models and schemas for the VidMash pipeline.

Defines data structures for video metadata, transcript processing,
semantic segmentation, knowledge graphs, visual assets, and document reports.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SegmentType(str, Enum):
    """Classification of a transcript segment's rhetorical purpose."""

    INTRODUCTION = "introduction"
    CORE_ARGUMENT = "core_argument"
    EXAMPLE = "example"
    TANGENT = "tangent"
    CONCLUSION = "conclusion"
    DEFINITION = "definition"
    STATISTIC = "statistic"
    ACTIONABLE = "actionable"


class VisualType(str, Enum):
    """Supported visual asset types for document generation."""

    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    FLOW_DIAGRAM = "flow_diagram"
    MIND_MAP = "mind_map"
    TIMELINE = "timeline"
    COMPARISON_TABLE = "comparison_table"
    RELATIONSHIP_DIAGRAM = "relationship_diagram"


class ProcessingStage(str, Enum):
    """Pipeline processing stages for job status tracking."""

    QUEUED = "queued"
    EXTRACTING = "extracting"
    SEGMENTING = "segmenting"
    ANALYZING = "analyzing"
    BUILDING_GRAPH = "building_graph"
    GENERATING_VISUALS = "generating_visuals"
    WRITING_NARRATIVE = "writing_narrative"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class VideoInput(BaseModel):
    """Input payload for submitting a video URL for processing."""

    url: str = Field(..., description="Video URL to process")


# ---------------------------------------------------------------------------
# Metadata & transcript
# ---------------------------------------------------------------------------

class VideoMetadata(BaseModel):
    """Metadata extracted from a video source."""

    video_id: str
    title: str
    channel: str
    duration: int  # seconds
    description: str = ""
    thumbnail_url: str = ""
    upload_date: str = ""
    chapters: list[dict] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class TranscriptChunk(BaseModel):
    """A single timed chunk from the video transcript."""

    text: str
    start: float
    end: float


# ---------------------------------------------------------------------------
# Semantic analysis
# ---------------------------------------------------------------------------

class SemanticSegment(BaseModel):
    """A semantically coherent segment derived from transcript chunks."""

    id: int
    text: str
    start_time: float
    end_time: float
    topic: str = ""
    segment_type: SegmentType = SegmentType.CORE_ARGUMENT
    word_count: int = 0


class DensityScore(BaseModel):
    """Information-density metrics for a semantic segment."""

    segment_id: int
    ids_score: float  # Information Density Score
    unique_concepts: int = 0
    evidence_strength: float = 0.0
    novelty_score: float = 0.0
    repetition_factor: float = 1.0
    linguistic_importance: float = 0.0
    statistical_density: float = 0.0
    entity_richness: float = 0.0
    is_high_value: bool = False


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """A named entity extracted from the transcript."""

    name: str
    entity_type: str  # PERSON, ORG, CONCEPT, PRODUCT, DATE, NUMBER, etc.
    mentions: int = 1
    context: str = ""


class Relationship(BaseModel):
    """A relationship between two entities in the knowledge graph."""

    source: str
    target: str
    relation_type: str
    strength: float = 1.0


class CausalChain(BaseModel):
    """A sequence of cause-and-effect steps identified in the content."""

    steps: list[str]
    description: str = ""


class KnowledgeGraph(BaseModel):
    """Aggregate knowledge graph built from the video content."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    causal_chains: list[CausalChain] = Field(default_factory=list)
    topic_hierarchy: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Visual assets & narrative
# ---------------------------------------------------------------------------

class VisualAsset(BaseModel):
    """A visual element (chart, diagram, etc.) generated for the document."""

    visual_type: VisualType
    title: str
    image_path: str = ""
    data: dict = Field(default_factory=dict)
    svg_content: str = ""


class KeyQuote(BaseModel):
    """A notable quote extracted from the video."""

    text: str
    timestamp: float
    speaker: str = ""


class NarrativeSection(BaseModel):
    """A section of the generated narrative document."""

    title: str
    content: str
    section_type: str = "body"
    order: int = 0
    visuals: list[VisualAsset] = Field(default_factory=list)
    key_quotes: list[KeyQuote] = Field(default_factory=list)
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0


class ActionItem(BaseModel):
    """An actionable item identified from the video content."""

    text: str
    priority: str = "medium"  # high, medium, low
    source_timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Document report
# ---------------------------------------------------------------------------

class DocumentReport(BaseModel):
    """Complete generated document report for a processed video."""

    metadata: VideoMetadata
    executive_summary: str = ""
    sections: list[NarrativeSection] = Field(default_factory=list)
    key_takeaways: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    knowledge_graph: KnowledgeGraph = Field(default_factory=KnowledgeGraph)
    visuals: list[VisualAsset] = Field(default_factory=list)
    processing_time: float = 0.0
    compression_ratio: str = ""


# ---------------------------------------------------------------------------
# Processing status
# ---------------------------------------------------------------------------

class ProcessingStatus(BaseModel):
    """Real-time status of a video processing job."""

    job_id: str
    video_url: str
    video_title: str = ""
    stage: ProcessingStage = ProcessingStage.QUEUED
    progress: int = 0  # 0-100
    message: str = ""
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    pdf_path: str = ""
    html_path: str = ""
    error: str = ""
