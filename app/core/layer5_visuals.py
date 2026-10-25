"""
Layer 5: Visual Intelligence Engine

Auto-generates charts, diagrams, mind maps, timelines from knowledge graph
and scored segment data.  Every visual is saved as a PNG and wrapped in a
:class:`VisualAsset` model for downstream consumption by the renderer.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import numpy as np

from app.models.schemas import (
    CausalChain,
    Entity,
    KnowledgeGraph,
    Relationship,
    SegmentType,
    VideoMetadata,
    VisualAsset,
    VisualType,
)

logger = logging.getLogger(__name__)

# ── Beautiful colour palette used across all visuals ───────────────────
PALETTE = [
    "#6366f1",  # indigo
    "#8b5cf6",  # violet
    "#a78bfa",  # light violet
    "#c4b5fd",  # lavender
    "#14b8a6",  # teal
    "#2dd4bf",  # light teal
    "#f59e0b",  # amber
    "#fb923c",  # light orange
]

_DPI = 150

# Try to use a clean style — fall back gracefully if unavailable.
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # keep default


class VisualEngine:
    """Generate publication-ready charts and diagrams from pipeline data."""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def generate(
        self,
        knowledge_graph: KnowledgeGraph,
        scored_segments: list[dict[str, Any]],
        metadata: VideoMetadata,
        output_dir: str,
    ) -> list[VisualAsset]:
        """Produce all applicable visuals and return a list of assets.

        Args:
            knowledge_graph: The knowledge graph built earlier in the pipeline.
            scored_segments: Segments enriched with density scores.
            metadata: Video metadata.
            output_dir: Directory to write image files into.

        Returns:
            A list of successfully created :class:`VisualAsset` objects.
        """
        os.makedirs(output_dir, exist_ok=True)
        assets: list[VisualAsset] = []

        generators: list[tuple[str, Any]] = [
            (
                "topic_mindmap",
                lambda: self._generate_topic_mindmap(
                    knowledge_graph.topic_hierarchy,
                    os.path.join(output_dir, "topic_mindmap.png"),
                ),
            ),
            (
                "timeline",
                lambda: self._generate_timeline(
                    scored_segments,
                    metadata,
                    os.path.join(output_dir, "timeline.png"),
                ),
            ),
            (
                "entity_chart",
                lambda: self._generate_entity_chart(
                    knowledge_graph.entities,
                    os.path.join(output_dir, "entity_chart.png"),
                ),
            ),
            (
                "density_heatmap",
                lambda: self._generate_density_heatmap(
                    scored_segments,
                    os.path.join(output_dir, "density_heatmap.png"),
                ),
            ),
            (
                "relationship_diagram",
                lambda: self._generate_relationship_diagram(
                    knowledge_graph.relationships,
                    os.path.join(output_dir, "relationship_diagram.png"),
                ),
            ),
            (
                "flow_diagram",
                lambda: self._generate_flow_diagram(
                    knowledge_graph.causal_chains,
                    os.path.join(output_dir, "flow_diagram.png"),
                ),
            ),
        ]

        for name, gen_fn in generators:
            try:
                asset = gen_fn()
                if asset is not None:
                    assets.append(asset)
                    logger.info("Generated visual: %s", name)
                else:
                    logger.debug("Skipped visual %s (no data)", name)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to generate visual %s: %s", name, exc)
            finally:
                plt.close("all")

        return assets

    # ------------------------------------------------------------------
    # Mind-map
    # ------------------------------------------------------------------

    def _generate_topic_mindmap(
        self, hierarchy: dict[str, Any], output_path: str
    ) -> VisualAsset | None:
        """Create a radial mind-map of the topic hierarchy.

        Central node = video title (or "Main Topic"),
        branches = top-level topics, leaves = sub-topics.
        """
        if not hierarchy:
            return None

        fig, ax = plt.subplots(figsize=(14, 10), dpi=_DPI)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-8, 8)
        ax.axis("off")
        fig.patch.set_facecolor("#fafaff")

        # Central node
        centre = (0, 0)
        central_box = FancyBboxPatch(
            (-2.2, -0.7),
            4.4,
            1.4,
            boxstyle="round,pad=0.3",
            facecolor=PALETTE[0],
            edgecolor="white",
            linewidth=2,
        )
        ax.add_patch(central_box)
        ax.text(
            centre[0],
            centre[1],
            "Main Topic",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="white",
            family="sans-serif",
        )

        topics = list(hierarchy.items())
        n_topics = max(len(topics), 1)
        radius_main = 5.5
        radius_sub = 3.0

        for idx, (topic, subtopics) in enumerate(topics):
            angle = 2 * math.pi * idx / n_topics - math.pi / 2
            tx = radius_main * math.cos(angle)
            ty = radius_main * math.sin(angle) * 0.7  # elliptical

            colour = PALETTE[idx % len(PALETTE)]

            # Line from centre to branch
            ax.plot(
                [centre[0], tx],
                [centre[1], ty],
                color=colour,
                linewidth=2,
                alpha=0.6,
                zorder=1,
            )

            # Branch node
            branch_box = FancyBboxPatch(
                (tx - 1.8, ty - 0.5),
                3.6,
                1.0,
                boxstyle="round,pad=0.25",
                facecolor=colour,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(branch_box)

            label = str(topic)[:28]
            ax.text(
                tx,
                ty,
                label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                family="sans-serif",
                zorder=4,
            )

            # Subtopic leaves
            subs = subtopics if isinstance(subtopics, list) else []
            n_subs = max(len(subs), 1)
            for si, sub in enumerate(subs[:6]):  # max 6 leaves per branch
                sub_angle = angle + (si - n_subs / 2) * 0.35
                sx = tx + radius_sub * math.cos(sub_angle) * 0.5
                sy = ty + radius_sub * math.sin(sub_angle) * 0.4

                ax.plot(
                    [tx, sx],
                    [ty, sy],
                    color=colour,
                    linewidth=1,
                    alpha=0.4,
                    zorder=1,
                )
                leaf_box = FancyBboxPatch(
                    (sx - 1.3, sy - 0.35),
                    2.6,
                    0.7,
                    boxstyle="round,pad=0.15",
                    facecolor=colour,
                    edgecolor="white",
                    alpha=0.55,
                    linewidth=1,
                    zorder=3,
                )
                ax.add_patch(leaf_box)
                ax.text(
                    sx,
                    sy,
                    str(sub)[:22],
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    family="sans-serif",
                    zorder=4,
                )

        ax.set_title(
            "Topic Mind Map",
            fontsize=16,
            fontweight="bold",
            color=PALETTE[0],
            pad=20,
            family="sans-serif",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        return VisualAsset(
            visual_type=VisualType.MIND_MAP,
            title="Topic Mind Map",
            image_path=output_path,
        )

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def _generate_timeline(
        self,
        segments: list[dict[str, Any]],
        metadata: VideoMetadata,
        output_path: str,
    ) -> VisualAsset | None:
        """Horizontal timeline showing when key topics appear."""
        if not segments:
            return None

        fig, ax = plt.subplots(figsize=(14, max(4, len(segments) * 0.35)), dpi=_DPI)
        fig.patch.set_facecolor("#fafaff")

        type_colours: dict[str, str] = {
            "introduction": PALETTE[4],
            "core_argument": PALETTE[0],
            "example": PALETTE[1],
            "tangent": PALETTE[6],
            "conclusion": PALETTE[5],
            "definition": PALETTE[2],
            "statistic": PALETTE[7],
            "actionable": PALETTE[3],
        }

        labels: list[str] = []
        for idx, seg in enumerate(segments):
            start = float(seg.get("start_time", seg.get("start", 0)))
            end = float(seg.get("end_time", seg.get("end", start + 10)))
            topic = seg.get("topic", f"Segment {idx + 1}")[:35]
            seg_type = seg.get("segment_type", "core_argument")
            if isinstance(seg_type, SegmentType):
                seg_type = seg_type.value
            colour = type_colours.get(seg_type, PALETTE[0])

            ax.barh(
                idx,
                end - start,
                left=start,
                height=0.6,
                color=colour,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.85,
            )
            labels.append(topic)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8, family="sans-serif")
        ax.set_xlabel("Time (seconds)", fontsize=10, family="sans-serif")
        ax.set_title(
            f"Content Timeline — {metadata.title[:50]}",
            fontsize=14,
            fontweight="bold",
            color=PALETTE[0],
            pad=15,
            family="sans-serif",
        )
        ax.invert_yaxis()

        # Legend
        legend_handles = [
            mpatches.Patch(color=c, label=t.replace("_", " ").title())
            for t, c in type_colours.items()
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=7,
            framealpha=0.9,
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        return VisualAsset(
            visual_type=VisualType.TIMELINE,
            title="Content Timeline",
            image_path=output_path,
        )

    # ------------------------------------------------------------------
    # Entity bar chart
    # ------------------------------------------------------------------

    def _generate_entity_chart(
        self, entities: list[Entity], output_path: str
    ) -> VisualAsset | None:
        """Horizontal bar chart of the top 15 entities by mention count."""
        if not entities:
            return None

        sorted_ents = sorted(entities, key=lambda e: e.mentions, reverse=True)[:15]
        names = [e.name[:30] for e in sorted_ents]
        counts = [e.mentions for e in sorted_ents]

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)), dpi=_DPI)
        fig.patch.set_facecolor("#fafaff")

        # Gradient bar colours
        n = len(names)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "vidmash", [PALETTE[0], PALETTE[4]]
        )
        colours = [cmap(i / max(n - 1, 1)) for i in range(n)]

        bars = ax.barh(range(n), counts, color=colours, edgecolor="white", height=0.65)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=9, family="sans-serif")
        ax.set_xlabel("Mentions", fontsize=10, family="sans-serif")
        ax.set_title(
            "Top Entities by Mention Count",
            fontsize=14,
            fontweight="bold",
            color=PALETTE[0],
            pad=15,
            family="sans-serif",
        )
        ax.invert_yaxis()

        # Annotate counts
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=8,
                color=PALETTE[1],
                family="sans-serif",
            )

        fig.tight_layout()
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        return VisualAsset(
            visual_type=VisualType.BAR_CHART,
            title="Top Entities by Mention Count",
            image_path=output_path,
        )

    # ------------------------------------------------------------------
    # Density heat-map
    # ------------------------------------------------------------------

    def _generate_density_heatmap(
        self, scored_segments: list[dict[str, Any]], output_path: str
    ) -> VisualAsset | None:
        """Heat-map of information density across the video timeline."""
        if not scored_segments:
            return None

        scores: list[float] = []
        labels: list[str] = []
        for idx, seg in enumerate(scored_segments):
            score = float(
                seg.get("ids_score", seg.get("score", seg.get("importance", 0.5)))
            )
            scores.append(score)
            topic = seg.get("topic", f"Seg {idx + 1}")
            labels.append(str(topic)[:25])

        fig, ax = plt.subplots(figsize=(max(10, len(scores) * 0.6), 3), dpi=_DPI)
        fig.patch.set_facecolor("#fafaff")

        data = np.array(scores).reshape(1, -1)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "density", ["#c4b5fd", PALETTE[0], "#f59e0b"]
        )
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7, family="sans-serif")
        ax.set_yticks([])
        ax.set_title(
            "Information Density Across Video",
            fontsize=14,
            fontweight="bold",
            color=PALETTE[0],
            pad=15,
            family="sans-serif",
        )

        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.25, shrink=0.6)
        cbar.set_label("Density Score", fontsize=9, family="sans-serif")

        fig.tight_layout()
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        return VisualAsset(
            visual_type=VisualType.LINE_CHART,
            title="Information Density Heatmap",
            image_path=output_path,
        )

    # ------------------------------------------------------------------
    # Relationship network diagram
    # ------------------------------------------------------------------

    def _generate_relationship_diagram(
        self, relationships: list[Relationship], output_path: str
    ) -> VisualAsset | None:
        """Network diagram: nodes = entities, edges = relationships."""
        if not relationships:
            return None

        # Collect unique nodes
        node_set: set[str] = set()
        for r in relationships:
            node_set.add(r.source)
            node_set.add(r.target)

        nodes = list(node_set)
        n = len(nodes)
        if n == 0:
            return None

        # Simple spring-layout: start with circular, then iterate
        positions: dict[str, tuple[float, float]] = {}
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / n
            positions[node] = (math.cos(angle) * 5, math.sin(angle) * 5)

        # A few iterations of Fruchterman-Reingold-style repulsion/attraction
        for _ in range(50):
            disp: dict[str, tuple[float, float]] = {nd: (0.0, 0.0) for nd in nodes}
            k = 4.0 / math.sqrt(max(n, 1))

            # Repulsion
            for i_n in range(n):
                for j_n in range(i_n + 1, n):
                    dx = positions[nodes[i_n]][0] - positions[nodes[j_n]][0]
                    dy = positions[nodes[i_n]][1] - positions[nodes[j_n]][1]
                    dist = math.sqrt(dx * dx + dy * dy) + 0.01
                    force = k * k / dist
                    fx, fy = force * dx / dist, force * dy / dist
                    disp[nodes[i_n]] = (disp[nodes[i_n]][0] + fx, disp[nodes[i_n]][1] + fy)
                    disp[nodes[j_n]] = (disp[nodes[j_n]][0] - fx, disp[nodes[j_n]][1] - fy)

            # Attraction along edges
            for r in relationships:
                if r.source in positions and r.target in positions:
                    dx = positions[r.source][0] - positions[r.target][0]
                    dy = positions[r.source][1] - positions[r.target][1]
                    dist = math.sqrt(dx * dx + dy * dy) + 0.01
                    force = dist * dist / k
                    fx, fy = force * dx / dist, force * dy / dist
                    disp[r.source] = (disp[r.source][0] - fx, disp[r.source][1] - fy)
                    disp[r.target] = (disp[r.target][0] + fx, disp[r.target][1] + fy)

            # Apply displacement (damped)
            temp = 0.3
            for nd in nodes:
                dx, dy = disp[nd]
                dist = math.sqrt(dx * dx + dy * dy) + 0.01
                scale = min(dist, temp) / dist
                positions[nd] = (
                    positions[nd][0] + dx * scale,
                    positions[nd][1] + dy * scale,
                )

        fig, ax = plt.subplots(figsize=(12, 10), dpi=_DPI)
        fig.patch.set_facecolor("#fafaff")
        ax.axis("off")

        # Draw edges
        for r in relationships:
            if r.source in positions and r.target in positions:
                x0, y0 = positions[r.source]
                x1, y1 = positions[r.target]
                alpha = min(1.0, 0.3 + r.strength * 0.5)
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=PALETTE[4],
                    alpha=alpha,
                    linewidth=1 + r.strength,
                    zorder=1,
                )
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                label = r.relation_type[:18]
                ax.text(
                    mid_x,
                    mid_y,
                    label,
                    fontsize=6,
                    color="#666666",
                    ha="center",
                    va="center",
                    family="sans-serif",
                    alpha=0.7,
                    zorder=2,
                )

        # Draw nodes
        for i, nd in enumerate(nodes):
            x, y = positions[nd]
            colour = PALETTE[i % len(PALETTE)]
            circle = Circle((x, y), 0.45, facecolor=colour, edgecolor="white", linewidth=2, zorder=5)
            ax.add_patch(circle)
            ax.text(
                x,
                y - 0.7,
                nd[:20],
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
                color="#333333",
                family="sans-serif",
                zorder=6,
            )

        ax.set_title(
            "Entity Relationship Diagram",
            fontsize=16,
            fontweight="bold",
            color=PALETTE[0],
            pad=20,
            family="sans-serif",
        )

        # Auto-scale
        all_x = [p[0] for p in positions.values()]
        all_y = [p[1] for p in positions.values()]
        margin = 2.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        fig.tight_layout()
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        return VisualAsset(
            visual_type=VisualType.RELATIONSHIP_DIAGRAM,
            title="Entity Relationship Diagram",
            image_path=output_path,
        )

    # ------------------------------------------------------------------
    # Comparison table image
    # ------------------------------------------------------------------

    def _generate_comparison_table_image(
        self, data: dict[str, Any], output_path: str
    ) -> VisualAsset | None:
        """Render a styled comparison table as a PNG via matplotlib."""
        headers = data.get("headers", [])
        rows = data.get("rows", [])
        if not headers or not rows:
            return None

        n_cols = len(headers)
        n_rows = len(rows)

        fig, ax = plt.subplots(
            figsize=(max(6, n_cols * 2.5), max(2, (n_rows + 1) * 0.6)), dpi=_DPI
        )
        ax.axis("off")
        fig.patch.set_facecolor("#fafaff")

        table_data = [headers] + rows
        table = ax.table(
            cellText=table_data,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        # Style header row
        for j in range(n_cols):
            cell = table[0, j]
            cell.set_facecolor(PALETTE[0])
            cell.set_text_props(color="white", fontweight="bold", family="sans-serif")
            cell.set_edgecolor("white")

        # Style data rows
        for i in range(1, n_rows + 1):
            bg = "#f0edff" if i % 2 == 0 else "white"
            for j in range(n_cols):
                cell = table[i, j]
                cell.set_facecolor(bg)
                cell.set_edgecolor("#e0e0e0")
                cell.set_text_props(family="sans-serif")

        table.scale(1, 1.8)
        ax.set_title(
            data.get("title", "Comparison Table"),
            fontsize=14,
            fontweight="bold",
            color=PALETTE[0],
            pad=15,
            family="sans-serif",
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        return VisualAsset(
            visual_type=VisualType.COMPARISON_TABLE,
            title=data.get("title", "Comparison Table"),
            image_path=output_path,
        )

    # ------------------------------------------------------------------
    # Flow diagram (causal chains)
    # ------------------------------------------------------------------

    def _generate_flow_diagram(
        self, causal_chains: list[CausalChain], output_path: str
    ) -> VisualAsset | None:
        """Flow diagram built from causal-chain steps using matplotlib."""
        if not causal_chains:
            return None

        # Flatten all chains into a list of labelled flows
        all_flows: list[tuple[str, list[str]]] = []
        for chain in causal_chains:
            if chain.steps:
                desc = chain.description or f"Chain ({len(chain.steps)} steps)"
                all_flows.append((desc, chain.steps))

        if not all_flows:
            return None

        total_steps = sum(len(steps) for _, steps in all_flows)
        fig_height = max(4, total_steps * 0.9 + len(all_flows) * 1.2)
        fig, ax = plt.subplots(figsize=(12, fig_height), dpi=_DPI)
        fig.patch.set_facecolor("#fafaff")
        ax.axis("off")

        y_cursor = fig_height - 1.0
        box_w, box_h = 6.0, 0.7

        for chain_idx, (desc, steps) in enumerate(all_flows):
            colour = PALETTE[chain_idx % len(PALETTE)]

            # Chain title
            ax.text(
                6,
                y_cursor,
                desc[:60],
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=colour,
                family="sans-serif",
            )
            y_cursor -= 1.0

            for step_idx, step_text in enumerate(steps):
                x_center = 6.0
                # Draw box
                box = FancyBboxPatch(
                    (x_center - box_w / 2, y_cursor - box_h / 2),
                    box_w,
                    box_h,
                    boxstyle="round,pad=0.2",
                    facecolor=colour,
                    edgecolor="white",
                    alpha=0.85,
                    linewidth=1.5,
                    zorder=3,
                )
                ax.add_patch(box)
                ax.text(
                    x_center,
                    y_cursor,
                    step_text[:55],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    family="sans-serif",
                    zorder=4,
                )

                # Arrow to next step
                if step_idx < len(steps) - 1:
                    ax.annotate(
                        "",
                        xy=(x_center, y_cursor - box_h / 2 - 0.25),
                        xytext=(x_center, y_cursor - box_h / 2),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=colour,
                            lw=2,
                            connectionstyle="arc3,rad=0",
                        ),
                        zorder=2,
                    )

                y_cursor -= 1.0

            y_cursor -= 0.5  # gap between chains

        ax.set_xlim(0, 12)
        ax.set_ylim(0, fig_height)
        ax.set_title(
            "Causal Flow Diagram",
            fontsize=16,
            fontweight="bold",
            color=PALETTE[0],
            pad=20,
            family="sans-serif",
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        return VisualAsset(
            visual_type=VisualType.FLOW_DIAGRAM,
            title="Causal Flow Diagram",
            image_path=output_path,
        )
