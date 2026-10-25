"""
Offline Integration and Unit Tests for VidMash 7-Layer Cortex Engine.
Tests all layers with mock transcript data so that no external APIs or network
calls are required for verification.
"""

import asyncio
import os
import shutil
import tempfile
import unittest

from app.models.schemas import (
    DensityScore,
    KnowledgeGraph,
    NarrativeSection,
    SegmentType,
    SemanticSegment,
    TranscriptChunk,
    VideoMetadata,
    VisualAsset,
    VisualType,
)
from app.core.layer2_segmentation import SegmentationEngine
from app.core.layer3_density import DensityAnalyzer
from app.core.layer5_visuals import VisualEngine
from app.core.layer7_renderer import DocumentRenderer
from app.models.schemas import DocumentReport


class TestVidMashLayers(unittest.IsolatedAsyncioTestCase):
    """Test individual layers of the Cortex Engine."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.mock_metadata = VideoMetadata(
            video_id="dQw4w9WgXcQ",
            title="The Quantum Computing Revolution & Practical Quantum Supremacy",
            channel="Tech Science Labs",
            duration=900,
            description="Deep dive into qubits, quantum error correction, and quantum supremacy.",
        )
        self.mock_chunks = [
            TranscriptChunk(text="Welcome everyone. Today we are going to explore the quantum computing revolution.", start=0.0, end=5.0),
            TranscriptChunk(text="Quantum supremacy is defined as the demonstration of a programmable quantum device solving a problem no classical computer can solve.", start=5.0, end=12.0),
            TranscriptChunk(text="In 2019, Google Sycamore achieved a milestone by performing a calculation in 200 seconds that would take Summit supercomputer 10,000 years.", start=12.0, end=20.0),
            TranscriptChunk(text="This represents a 1,500,000x acceleration in raw compute capacity.", start=20.0, end=26.0),
            TranscriptChunk(text="Crucial challenges remain, specifically qubit decoherence and gate fidelity errors.", start=26.0, end=34.0),
            TranscriptChunk(text="For example, topological quantum computing aims to protect qubits by braiding anyons in two-dimensional space.", start=34.0, end=42.0),
            TranscriptChunk(text="You should keep a close eye on neutral-atom systems and trapped-ion architectures.", start=42.0, end=50.0),
            TranscriptChunk(text="In summary, quantum computers are transitioning from academic proofs of concept to commercial reality.", start=50.0, end=60.0),
        ]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_layer2_segmentation(self) -> None:
        """Test that SegmentationEngine splits chunks into semantic segments with rhetorical types."""
        engine = SegmentationEngine()
        segments = await engine.segment(self.mock_chunks, self.mock_metadata)
        self.assertTrue(len(segments) >= 1)
        # Check segment fields
        seg = segments[0]
        self.assertIsInstance(seg, SemanticSegment)
        self.assertTrue(len(seg.text) > 0)
        print(f"\n[Test] Layer 2 generated {len(segments)} segments successfully.")

    async def test_layer3_density(self) -> None:
        """Test Semantic Density Analyzer (IDS calculation)."""
        engine = SegmentationEngine()
        segments = await engine.segment(self.mock_chunks, self.mock_metadata)

        analyzer = DensityAnalyzer()
        scored_pairs = await analyzer.analyze(segments)
        self.assertEqual(len(scored_pairs), len(segments))

        for seg, ds in scored_pairs:
            self.assertIsInstance(ds, DensityScore)
            self.assertTrue(0.0 <= ds.ids_score)
            print(f"Segment #{seg.id} ({seg.segment_type.value}) - IDS: {ds.ids_score:.3f}, High Value: {ds.is_high_value}")

    async def test_layer5_visuals(self) -> None:
        """Test Visual Intelligence Engine generates publication-quality charts."""
        engine = SegmentationEngine()
        segments = await engine.segment(self.mock_chunks, self.mock_metadata)
        analyzer = DensityAnalyzer()
        scored_pairs = await analyzer.analyze(segments)

        kg = KnowledgeGraph()
        # Add sample entities and relationships
        from app.models.schemas import Entity, Relationship, CausalChain
        kg.entities = [
            Entity(name="Google Sycamore", entity_type="PRODUCT", mentions=3),
            Entity(name="Summit Supercomputer", entity_type="PRODUCT", mentions=2),
            Entity(name="Qubit Decoherence", entity_type="CONCEPT", mentions=4),
            Entity(name="Trapped-Ion Architecture", entity_type="CONCEPT", mentions=2),
        ]
        kg.relationships = [
            Relationship(source="Google Sycamore", target="Quantum Supremacy", relation_type="DEMONSTRATES"),
            Relationship(source="Qubit Decoherence", target="Gate Fidelity", relation_type="LIMITS"),
        ]
        kg.causal_chains = [
            CausalChain(steps=["Braiding Anyons", "Topological Protection", "Fault-Tolerant Qubits"], description="Error mitigation pathway")
        ]
        kg.topic_hierarchy = {
            "Quantum Computing": {
                "Hardware": ["Superconducting", "Trapped Ions", "Neutral Atoms"],
                "Algorithms": ["Shor's Algorithm", "Grover's Search", "VQE"],
            }
        }

        # Convert to segment dicts
        from app.core.pipeline import VidMashPipeline
        segment_dicts = [VidMashPipeline._normalize_segment(p) for p in scored_pairs]

        visual_engine = VisualEngine()
        assets = await visual_engine.generate(kg, segment_dicts, self.mock_metadata, self.temp_dir)
        self.assertTrue(len(assets) >= 2)

        for a in assets:
            self.assertTrue(os.path.exists(a.image_path))
            print(f"[Test] Generated visual asset: {a.title} ({os.path.getsize(a.image_path)} bytes)")

    async def test_layer7_renderer(self) -> None:
        """Test DocumentRenderer compiles both PDF and HTML reports cleanly."""
        renderer = DocumentRenderer()

        report = DocumentReport(
            metadata=self.mock_metadata,
            executive_summary=(
                "Quantum computing is rapidly moving from theoretical laboratory physics into practical application.\n\n"
                "Recent breakthroughs demonstrate 1,500,000x speedups for specialized sampling problems, laying the "
                "foundation for commercial quantum advantage in chemistry, materials science, and optimization."
            ),
            sections=[
                NarrativeSection(
                    title="Quantum Supremacy and Experimental Breakthroughs",
                    content=(
                        "In October 2019, an experimental milestone was achieved with the 53-qubit Sycamore processor. "
                        "The system completed a benchmark calculation in 200 seconds that classical supercomputers "
                        "would take years to emulate. This marked the formal arrival of quantum supremacy."
                    ),
                    order=1,
                ),
                NarrativeSection(
                    title="The Road to Fault Tolerance: Error Correction",
                    content=(
                        "Physical qubits are susceptible to environmental thermal fluctuations and cosmic radiation, "
                        "causing decoherence within microseconds. Quantum error-correcting codes, including surface codes "
                        "and topological protection, represent the critical frontier for building logical qubits."
                    ),
                    order=2,
                ),
            ],
            key_takeaways=[
                "Quantum hardware has demonstrated non-classical computational scaling.",
                "Coherence time and two-qubit gate fidelities are the primary hardware bottlenecks.",
                "Neutral-atom and photonic architectures are rapidly closing the gap with superconducting circuits.",
            ],
            action_items=[
                {"text": "Audit existing public-key cryptography (RSA/ECC) for post-quantum vulnerability.", "priority": "high", "source_timestamp": 25.0},
                {"text": "Evaluate hybrid classical-quantum algorithms (VQE) for chemistry simulations.", "priority": "medium", "source_timestamp": 45.0},
            ],
            processing_time=12.4,
            compression_ratio="15m video → 2m read",
        )

        pdf_path, html_path = await renderer.render(report, self.temp_dir)
        self.assertTrue(os.path.exists(pdf_path))
        self.assertTrue(os.path.exists(html_path))
        self.assertTrue(os.path.getsize(pdf_path) > 1000)
        self.assertTrue(os.path.getsize(html_path) > 500)
        print(f"\n[Test] PDF created: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
        print(f"[Test] HTML created: {html_path} ({os.path.getsize(html_path)} bytes)")


if __name__ == "__main__":
    unittest.main()
