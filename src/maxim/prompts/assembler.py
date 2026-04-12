"""PromptAssembler — single composition point for agent system messages.

B1 implementation. All structured inputs flow through PromptAssembler
to produce the final system message. During the dual-write migration
(Phase 1-2), the assembler runs alongside the legacy PromptBuilder;
at Phase 3 it becomes authoritative.

The assembler's key invariant: it consumes structured types, not raw
strings. Every input has a defined schema so the prompt can be
budget-aware, tested, and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubstrateNode:
    """A single ATL node for MemorySummary consumption.

    Lightweight view — no embedding vectors, just what the prompt needs.
    """

    node_id: str
    text: str
    modality: str  # "text" or "vision"
    confidence: float = 0.5
    reinforcement_count: int = 1
    category: str = "substrate"


@dataclass
class MemorySummary:
    """Structured summary of substrate memory for prompt composition.

    Produced by the substrate path (LinguisticEncoder → EC → ATL).
    Consumed by PromptAssembler to build the memory section of the
    system message.

    During dual-write (Phase 1), this is populated but not yet
    authoritative — the legacy transcript_chunk path still drives
    the prompt. At Phase 3 cutover, MemorySummary becomes the
    sole source of memory context.
    """

    # Active substrate nodes relevant to the current turn
    active_nodes: list[SubstrateNode] = field(default_factory=list)

    # Node that the current percept was mapped to (if any)
    current_node_id: str | None = None

    # Whether pattern completion occurred (vs separation)
    pattern_completed: bool = False

    # Summary statistics
    total_nodes: int = 0
    text_nodes: int = 0
    vision_nodes: int = 0

    @classmethod
    def from_atl(
        cls,
        atl: Any,
        current_node_id: str | None = None,
        pattern_completed: bool = False,
        modality: str | None = None,
        limit: int = 20,
    ) -> "MemorySummary":
        """Build a MemorySummary from ATL state.

        Queries the ATL for substrate-tagged concepts and packages
        them into prompt-consumable SubstrateNode instances.
        """
        nodes: list[SubstrateNode] = []

        # Get recent substrate concepts
        concepts = atl.recall(
            limit=limit,
            category="substrate",
            substrate_modality=modality,
        )
        for concept in concepts:
            nodes.append(
                SubstrateNode(
                    node_id=concept.id,
                    text=concept.name,
                    modality=concept.substrate_modality or "text",
                    confidence=concept.confidence,
                    reinforcement_count=getattr(concept, "reinforcement_count", 1),
                    category=concept.category,
                )
            )

        # Count by modality
        text_nodes = sum(1 for n in nodes if n.modality == "text")
        vision_nodes = sum(1 for n in nodes if n.modality == "vision")

        return cls(
            active_nodes=nodes,
            current_node_id=current_node_id,
            pattern_completed=pattern_completed,
            total_nodes=len(nodes),
            text_nodes=text_nodes,
            vision_nodes=vision_nodes,
        )


class PromptAssembler:
    """Composes structured inputs into prompt sections.

    Phase 1: only compose_memory_section is active (dual-write).
    Phase 3: compose() replaces the full PromptBuilder pipeline.

    The assembler is stateless — all state flows in through method
    arguments. This makes it testable and deterministic.
    """

    def compose_memory_section(self, summary: MemorySummary) -> str:
        """Compose the memory section from substrate MemorySummary.

        Returns a prompt-ready string describing what the agent
        knows from its substrate memory. Empty string if no
        substrate nodes exist.
        """
        if not summary.active_nodes:
            return ""

        lines = ["=== Substrate Memory ==="]

        # Current percept mapping
        if summary.current_node_id:
            action = "recognized" if summary.pattern_completed else "new concept"
            lines.append(f"Current input: {action} (node {summary.current_node_id[:8]})")

        # List active concepts grouped by modality
        text_nodes = [n for n in summary.active_nodes if n.modality == "text"]
        vision_nodes = [n for n in summary.active_nodes if n.modality == "vision"]

        if text_nodes:
            lines.append(f"Known concepts ({len(text_nodes)} text):")
            for node in text_nodes[:10]:
                conf = f" [{node.confidence:.0%}]" if node.reinforcement_count > 1 else ""
                lines.append(f"  - {node.text}{conf}")
            if len(text_nodes) > 10:
                lines.append(f"  ... and {len(text_nodes) - 10} more")

        if vision_nodes:
            lines.append(f"Known visual concepts ({len(vision_nodes)}):")
            for node in vision_nodes[:5]:
                lines.append(f"  - {node.text}")

        return "\n".join(lines)

    def compose_observation_section(
        self,
        percept: Any,
        summary: MemorySummary | None = None,
    ) -> str:
        """Compose the observation section with optional substrate context.

        Phase 1: builds the same output as build_observation_section
        from prompt_builder.py, but can append substrate memory.
        Phase 3: this replaces build_observation_section entirely.
        """
        lines = ["=== Current Observation ==="]

        if percept.transcript_chunk:
            lines.append(f'Heard: "{percept.transcript_chunk[:200]}"')
        if percept.detections:
            objects = [d.get("label", "?") for d in percept.detections[:5]]
            lines.append(f"Visible objects: {', '.join(objects)}")
        if percept.cli_input:
            lines.append(f'User input: "{percept.cli_input}"')

        # Substrate memory context (Phase 1: appended alongside legacy)
        if summary and summary.active_nodes:
            mem_section = self.compose_memory_section(summary)
            if mem_section:
                lines.append("")
                lines.append(mem_section)

        return "\n".join(lines) if len(lines) > 1 else ""
