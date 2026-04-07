"""CrossLayerGraph — links memories across Hippocampus, ATL, and Angular Gyrus.

Lightweight graph tracking how records in different memory layers relate.
Supports cross-layer spreading activation for holistic memory retrieval.

Edge types capture the nature of cross-layer relationships:
- DERIVED_FROM: ATL concept derived from hippocampus episode(s)
- INSTANCE_OF: Episode is an instance of an ATL concept
- COMPUTED_FROM: AG math result derived from data in another layer
- QUANTIFIES: AG provides numerical characterization of a concept
- STATISTICALLY_CONFIRMS: AG pattern record validates an ATL concept
- TEMPORALLY_CORRELATES: SCN coupling links temporal pattern across layers
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.memory.layer import MemoryLayer

logger = logging.getLogger(__name__)


class CrossLayerEdgeType(Enum):
    """Types of edges between records in different memory layers."""

    # Episodic <-> Semantic
    DERIVED_FROM = auto()  # ATL concept derived from episode(s)
    INSTANCE_OF = auto()  # Episode is instance of ATL concept
    INFORMS = auto()  # Generic cross-layer influence

    # Math <-> Semantic
    COMPUTED_FROM = auto()  # AG result derived from another layer's data
    QUANTIFIES = auto()  # AG provides numerical characterization

    # Statistical <-> Semantic/Episodic
    STATISTICALLY_CONFIRMS = auto()  # AG PATTERN validates ATL concept
    TEMPORALLY_CORRELATES = auto()  # SCN coupling links temporal pattern


@dataclass
class CrossLayerEdge:
    """A directed edge between records in different memory layers."""

    source_layer: str
    source_id: str
    target_layer: str
    target_id: str
    edge_type: CrossLayerEdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class CrossLayerGraph:
    """Graph linking records across memory layers.

    Lightweight: only stores edge data, delegates record access to
    the layer objects passed at construction.
    """

    def __init__(
        self,
        layers: dict[str, MemoryLayer] | None = None,
        persistence_path: str | None = None,
    ) -> None:
        self._layers = layers or {}
        self._persistence_path = persistence_path

        # Edges indexed by (source_layer, source_id)
        self._outgoing: dict[tuple[str, str], list[CrossLayerEdge]] = defaultdict(list)
        # Reverse index by (target_layer, target_id)
        self._incoming: dict[tuple[str, str], list[CrossLayerEdge]] = defaultdict(list)

    def add_edge(
        self,
        source_layer: str,
        source_id: str,
        target_layer: str,
        target_id: str,
        edge_type: CrossLayerEdgeType,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a cross-layer edge."""
        edge = CrossLayerEdge(
            source_layer=source_layer,
            source_id=source_id,
            target_layer=target_layer,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )
        self._outgoing[(source_layer, source_id)].append(edge)
        self._incoming[(target_layer, target_id)].append(edge)

    def remove_record(self, layer_name: str, record_id: str) -> int:
        """Remove all edges involving a record. Returns edges removed."""
        removed = 0
        key = (layer_name, record_id)

        # Remove outgoing edges
        for edge in self._outgoing.pop(key, []):
            tgt = (edge.target_layer, edge.target_id)
            self._incoming[tgt] = [
                e
                for e in self._incoming.get(tgt, [])
                if not (e.source_layer == layer_name and e.source_id == record_id)
            ]
            removed += 1

        # Remove incoming edges
        for edge in self._incoming.pop(key, []):
            src = (edge.source_layer, edge.source_id)
            self._outgoing[src] = [
                e
                for e in self._outgoing.get(src, [])
                if not (e.target_layer == layer_name and e.target_id == record_id)
            ]
            removed += 1

        return removed

    def get_linked(
        self,
        layer_name: str,
        record_id: str,
        target_layer: str | None = None,
        edge_type: CrossLayerEdgeType | None = None,
    ) -> list[CrossLayerEdge]:
        """Get cross-layer edges from a record.

        Optional filters: target_layer, edge_type.
        """
        key = (layer_name, record_id)
        edges = self._outgoing.get(key, [])

        if target_layer is not None:
            edges = [e for e in edges if e.target_layer == target_layer]
        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]

        return edges

    def get_incoming(
        self,
        layer_name: str,
        record_id: str,
        source_layer: str | None = None,
        edge_type: CrossLayerEdgeType | None = None,
    ) -> list[CrossLayerEdge]:
        """Get cross-layer edges pointing to a record."""
        key = (layer_name, record_id)
        edges = self._incoming.get(key, [])

        if source_layer is not None:
            edges = [e for e in edges if e.source_layer == source_layer]
        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]

        return edges

    def cross_layer_activation(
        self,
        start_layer: str,
        seed_ids: list[str],
        max_depth: int = 2,
        decay: float = 0.5,
    ) -> dict[str, list[tuple[str, float]]]:
        """Spreading activation across layers.

        Returns: {layer_name: [(record_id, activation_score), ...]}
        """
        # BFS with decay
        activations: dict[tuple[str, str], float] = {}
        frontier: list[tuple[str, str, float]] = [(start_layer, sid, 1.0) for sid in seed_ids]
        visited: set[tuple[str, str]] = set()

        for depth in range(max_depth):
            next_frontier: list[tuple[str, str, float]] = []
            for layer, rid, score in frontier:
                key = (layer, rid)
                if key in visited:
                    continue
                visited.add(key)

                if key not in {(start_layer, s) for s in seed_ids}:
                    activations[key] = max(activations.get(key, 0.0), score)

                # Spread to cross-layer neighbors
                for edge in self._outgoing.get(key, []):
                    new_score = score * edge.weight * decay
                    if new_score > 0.01:
                        next_frontier.append((edge.target_layer, edge.target_id, new_score))

            frontier = next_frontier

        # Group by layer
        result: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (layer, rid), score in activations.items():
            result[layer].append((rid, score))

        # Sort each layer's results by score
        for layer in result:
            result[layer].sort(key=lambda x: x[1], reverse=True)

        return dict(result)

    def save(self, path: str | None = None) -> None:
        """Persist cross-layer graph to JSON."""
        path = path or self._persistence_path
        if path is None:
            return

        from maxim.utils.atomic_io import atomic_write_json

        atomic_write_json(path, self.to_dict(), default=None)

    def load(self, path: str | None = None) -> None:
        """Restore cross-layer graph from JSON."""
        path = path or self._persistence_path
        if path is None or not os.path.exists(path):
            return

        with open(path) as f:
            data = json.load(f)

        self._outgoing.clear()
        self._incoming.clear()

        for e_data in data.get("edges", []):
            try:
                edge_type = CrossLayerEdgeType[e_data["edge_type"]]
            except KeyError:
                continue
            self.add_edge(
                source_layer=e_data["source_layer"],
                source_id=e_data["source_id"],
                target_layer=e_data["target_layer"],
                target_id=e_data["target_id"],
                edge_type=edge_type,
                weight=e_data.get("weight", 1.0),
                metadata=e_data.get("metadata", {}),
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        edges = []
        for edge_list in self._outgoing.values():
            for edge in edge_list:
                edges.append(
                    {
                        "source_layer": edge.source_layer,
                        "source_id": edge.source_id,
                        "target_layer": edge.target_layer,
                        "target_id": edge.target_id,
                        "edge_type": edge.edge_type.name,
                        "weight": edge.weight,
                        "metadata": edge.metadata,
                    }
                )
        return {"version": "1.0", "edges": edges}

    def stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        return {
            "total_edges": sum(len(el) for el in self._outgoing.values()),
            "source_records": len(self._outgoing),
            "target_records": len(self._incoming),
        }


__all__ = [
    "CrossLayerEdge",
    "CrossLayerEdgeType",
    "CrossLayerGraph",
]
