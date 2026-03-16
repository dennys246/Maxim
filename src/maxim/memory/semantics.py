"""Semantics — Typed relationship manager over a DependencyGraph.

Wraps a DependencyGraph + RelationshipRegistry to provide rich typed
relationships (IS_A, HAS_PART, CAUSES, etc.) on top of the generic
graph's ASSOCIATES/CAUSES edge types.

Decoupled from any specific memory layer — ATL uses it for concept
relationships, but CrossLayerGraph or even Hippocampus could wrap
their graphs with Semantics for richer associations.
"""

from __future__ import annotations

from typing import Any

from maxim.agents.bus import DependencyGraph, EdgeType
from maxim.memory.semantic_types import (
    RelationshipRegistry,
    SemanticRelationship,
)


class Semantics:
    """Typed relationship manager over a DependencyGraph.

    Stores fine-grained relationship types in edge metadata while using
    the graph's native ASSOCIATES/CAUSES edge types as the transport.
    Symmetric relationship types get bidirectional edges.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        registry: RelationshipRegistry | None = None,
    ) -> None:
        self._graph = graph
        self._registry = registry or RelationshipRegistry()

    @property
    def registry(self) -> RelationshipRegistry:
        return self._registry

    def define(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        weight: float = 1.0,
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Define a typed relationship between two records.

        Returns True if the edge was added (False if rel_type unknown).
        """
        if self._registry.get(rel_type) is None:
            return False

        edge_meta = {
            "relationship_type": rel_type,
            "confidence": confidence,
            **(metadata or {}),
        }

        # Use CAUSES for causal types, ASSOCIATES for everything else
        edge_type = EdgeType.CAUSES if rel_type == "CAUSES" else EdgeType.ASSOCIATES

        # Ensure nodes exist
        if self._graph.get_node(source_id) is None:
            self._graph.add_node(source_id, source_id)
        if self._graph.get_node(target_id) is None:
            self._graph.add_node(target_id, target_id)

        if self._registry.is_symmetric(rel_type):
            # add_bidirectional doesn't support metadata, so call add_edge twice
            self._graph.add_edge(source_id, target_id, edge_type, weight, edge_meta)
            self._graph.add_edge(target_id, source_id, edge_type, weight, edge_meta)
        else:
            self._graph.add_edge(source_id, target_id, edge_type, weight, edge_meta)

        return True

    def remove(
        self,
        source_id: str,
        target_id: str,
        rel_type: str | None = None,
    ) -> int:
        """Remove relationship edges between two records.

        If rel_type is given, only removes edges of that type.
        Returns the number of edges removed.
        """
        removed = 0

        # Access outgoing edges directly
        with self._graph._lock:
            edges = self._graph._outgoing.get(source_id, [])
            to_keep = []
            for edge in edges:
                if edge.target != target_id:
                    to_keep.append(edge)
                    continue
                if rel_type is not None:
                    edge_rel = edge.metadata.get("relationship_type", "")
                    if edge_rel != rel_type:
                        to_keep.append(edge)
                        continue
                # Remove this edge
                removed += 1
                # Also remove from incoming
                incoming = self._graph._incoming.get(target_id, [])
                self._graph._incoming[target_id] = [
                    e for e in incoming if not (e.source == source_id and e is edge)
                ]
            self._graph._outgoing[source_id] = to_keep

            # Also remove reverse edge for symmetric types
            if rel_type and self._registry.is_symmetric(rel_type):
                rev_edges = self._graph._outgoing.get(target_id, [])
                rev_keep = []
                for edge in rev_edges:
                    if edge.target != source_id:
                        rev_keep.append(edge)
                        continue
                    edge_rel = edge.metadata.get("relationship_type", "")
                    if edge_rel != rel_type:
                        rev_keep.append(edge)
                        continue
                    removed += 1
                    incoming = self._graph._incoming.get(source_id, [])
                    self._graph._incoming[source_id] = [
                        e for e in incoming if not (e.source == target_id and e is edge)
                    ]
                self._graph._outgoing[target_id] = rev_keep

        return removed

    def find(
        self,
        record_id: str,
        rel_type: str | None = None,
        direction: str = "outgoing",
        limit: int = 20,
    ) -> list[tuple[str, SemanticRelationship]]:
        """Find relationships for a record.

        Args:
            record_id: The record to query.
            rel_type: Filter by relationship type (None = all).
            direction: 'outgoing', 'incoming', or 'both'.
            limit: Maximum results.

        Returns:
            List of (other_id, SemanticRelationship) tuples.
        """
        results: list[tuple[str, SemanticRelationship]] = []

        with self._graph._lock:
            if direction in ("outgoing", "both"):
                for edge in self._graph._outgoing.get(record_id, []):
                    edge_rel = edge.metadata.get("relationship_type", "")
                    if rel_type is not None and edge_rel != rel_type:
                        continue
                    rel = SemanticRelationship(
                        source_id=record_id,
                        target_id=edge.target,
                        relationship_type=edge_rel,
                        weight=edge.weight,
                        confidence=edge.metadata.get("confidence", 0.5),
                        metadata=edge.metadata,
                    )
                    results.append((edge.target, rel))

            if direction in ("incoming", "both"):
                for edge in self._graph._incoming.get(record_id, []):
                    edge_rel = edge.metadata.get("relationship_type", "")
                    if rel_type is not None and edge_rel != rel_type:
                        continue
                    rel = SemanticRelationship(
                        source_id=edge.source,
                        target_id=record_id,
                        relationship_type=edge_rel,
                        weight=edge.weight,
                        confidence=edge.metadata.get("confidence", 0.5),
                        metadata=edge.metadata,
                    )
                    results.append((edge.source, rel))

        return results[:limit]

    def update_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        weight: float | None = None,
        confidence: float | None = None,
        confidence_delta: float | None = None,
    ) -> bool:
        """Update an existing relationship's weight and/or confidence.

        Args:
            weight: Set absolute weight (None = unchanged).
            confidence: Set absolute confidence (None = unchanged).
            confidence_delta: Add to current confidence, clamped to [0.05, 0.95].
                Mutually exclusive with confidence.

        Returns True if edge was found and updated, False if not found.
        """
        edge_type = EdgeType.CAUSES if rel_type == "CAUSES" else EdgeType.ASSOCIATES

        # Find the edge via DependencyGraph's public API
        edge = self._graph.find_edge(
            source_id, target_id, edge_type,
            metadata_match={"relationship_type": rel_type},
        )
        if edge is None:
            return False

        # Build metadata updates
        meta_updates: dict[str, Any] = {}
        if confidence is not None:
            meta_updates["confidence"] = confidence
        elif confidence_delta is not None:
            current = edge.metadata.get("confidence", 0.5)
            meta_updates["confidence"] = max(0.05, min(0.95, current + confidence_delta))

        # Apply via DependencyGraph's public update_edge
        self._graph.update_edge(
            source_id, target_id, edge_type,
            weight=weight,
            metadata_updates=meta_updates if meta_updates else None,
        )

        # Update symmetric reverse edge
        if self._registry.is_symmetric(rel_type):
            rev_meta = dict(meta_updates) if meta_updates else {}
            self._graph.update_edge(
                target_id, source_id, edge_type,
                weight=weight,
                metadata_updates=rev_meta if rev_meta else None,
            )

        return True

    def get_all_for(self, record_id: str) -> list[SemanticRelationship]:
        """Get all relationships involving a record (both directions)."""
        pairs = self.find(record_id, direction="both", limit=100)
        return [rel for _, rel in pairs]

    def propose_type(
        self,
        name: str,
        description: str,
        symmetric: bool = False,
    ) -> bool:
        """Propose a new relationship type (agent-callable)."""
        return self._registry.register(name, description, symmetric)

    def list_types(self) -> list[str]:
        """List all registered relationship types."""
        return self._registry.all_types()

    def save_registry(self, path: str) -> None:
        """Save the relationship registry to a JSON file."""
        import json

        data = self._registry.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_registry(self, path: str) -> None:
        """Load the relationship registry from a JSON file."""
        import json
        import os

        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self._registry = RelationshipRegistry.from_dict(data)


__all__ = ["Semantics"]
