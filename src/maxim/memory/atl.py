"""ATL (Anterior Temporal Lobe) — Semantic concept memory layer.

Implements MemoryLayer ABC for semantic knowledge: facts, concepts,
entities, and their typed relationships.  Structurally mirrors
Hippocampus but stores SemanticMemory instead of EpisodicMemory.

Brain mapping: The ATL integrates multimodal information into unified
semantic representations.  It captures *what things mean* rather than
*what happened* (episodic) or *how to compute* (angular gyrus).

Concepts are promoted from episodes (via NAc reward signals and
StatisticianAgent pattern confirmation) or ingested directly (RAG).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from uuid import uuid4

from maxim.agents.bus import DependencyGraph, EdgeType
from maxim.memory.layer import MemoryLayer
from maxim.memory.rwlock import RWLock
from maxim.memory.semantic_types import (
    CompressedSemantic,
    Concept,
    ConceptProvenance,
    RelationshipRegistry,
    SemanticMemory,
    SemanticRelationship,
)
from maxim.memory.semantics import Semantics

logger = logging.getLogger(__name__)


@dataclass
class ATLConfig:
    """Configuration for the ATL semantic memory layer."""

    max_concepts: int = 10_000
    persistence_path: str | None = None
    indexed_keys: frozenset[str] = frozenset({"name", "category"})
    retention_threshold: float = 0.2       # Lower than hippocampus — concepts more stable
    compression_threshold: float = 0.4
    max_age_without_access: float = 30 * 86400  # 30 days (slower decay)
    name_similarity_threshold: float = 0.8      # For deduplication


class ATL(MemoryLayer):
    """Anterior Temporal Lobe — Semantic concept memory.

    MemoryLayer protocol (store/get/recall/consolidate/save/load)
    plus semantic-specific methods (define_relationship, find_or_create).

    Thread-safe via RWLock.
    """

    def __init__(self, config: ATLConfig | None = None) -> None:
        self.config = config or ATLConfig()

        # Primary storage: concept_id -> SemanticMemory | CompressedSemantic
        self._concepts: dict[str, SemanticMemory | CompressedSemantic] = {}

        # Hash index: context_key -> set of concept_ids
        self._context_index: dict[str, set[str]] = defaultdict(set)
        self._concept_contexts: dict[str, set[str]] = defaultdict(set)

        # Relationship graph + typed semantics manager
        self._graph: DependencyGraph[str] = DependencyGraph()
        self._semantics = Semantics(self._graph)

        # Thread safety
        self._rwlock = RWLock()

        # Callbacks
        self._on_concept_captured: list[Callable[[str, SemanticMemory], None]] = []
        self._on_concept_deleted: list[Callable[[str], None]] = []

        # Optional SCN reference for temporal-aware strategies
        self._scn: Any = None

        # Stats
        self._stats: dict[str, int] = {"concepts_stored": 0, "queries": 0}
        self._compressed_count: int = 0

    # ------------------------------------------------------------------
    # SCN Integration
    # ------------------------------------------------------------------

    def connect_scn(self, scn: Any) -> None:
        """Connect SCN for temporal-aware eviction and consolidation.

        When connected, ATL will use TemporalAwareStrategy for smarter
        concept retention — rhythmically relevant concepts survive even
        if they haven't been accessed recently.
        """
        self._scn = scn
        logger.info("Connected SCN to ATL (temporal-aware consolidation enabled)")

    @property
    def scn(self) -> Any:
        """Get the connected SCN, if any."""
        return self._scn

    # ------------------------------------------------------------------
    # MemoryLayer Protocol
    # ------------------------------------------------------------------

    @property
    def layer_name(self) -> str:
        return "atl"

    @property
    def graph(self) -> DependencyGraph:
        return self._graph

    def store(self, record: SemanticMemory, **kwargs: Any) -> str:
        """Store a semantic concept. Returns its ID."""
        with self._rwlock.write():
            # Capacity check — evict before insert
            if len(self._concepts) >= self.config.max_concepts:
                self._evict_one()

            self._concepts[record.id] = record

            # Build index entries
            self._index_concept(record.id, record)

            # Add graph node
            self._graph.add_node(record.id, record.id)

            self._stats["concepts_stored"] += 1

        # Fire callbacks outside lock
        for cb in self._on_concept_captured:
            try:
                cb(record.id, record)
            except Exception as e:
                logger.warning("ATL capture callback failed: %s", e)

        return record.id

    def get(self, record_id: str) -> SemanticMemory | CompressedSemantic | None:
        """Retrieve a concept by ID. Updates access tracking."""
        with self._rwlock.read():
            concept = self._concepts.get(record_id)
            if concept is not None:
                concept.touch()
            return concept

    def remove(self, record_id: str) -> None:
        """Remove a concept and clean up references."""
        with self._rwlock.write():
            if record_id not in self._concepts:
                return

            # Clean up context index
            for key in self._concept_contexts.get(record_id, set()):
                self._context_index[key].discard(record_id)
                if not self._context_index[key]:
                    del self._context_index[key]
            self._concept_contexts.pop(record_id, None)

            # Clean up graph
            if self._graph.get_node(record_id) is not None:
                self._graph.remove_node(record_id)

            mem = self._concepts.pop(record_id, None)
            if isinstance(mem, CompressedSemantic):
                self._compressed_count -= 1

        # Fire callbacks outside lock
        for cb in self._on_concept_deleted:
            try:
                cb(record_id)
            except Exception as e:
                logger.warning("ATL deletion callback failed: %s", e)

    def recall_by_ids(
        self,
        record_ids: list[str],
    ) -> list[SemanticMemory | CompressedSemantic]:
        """Bulk-load concepts by ID. Skips missing IDs.

        Single lock acquisition — more efficient than iterating get().
        Does NOT update access tracking.
        """
        with self._rwlock.read():
            results = []
            for rid in record_ids:
                concept = self._concepts.get(rid)
                if concept is not None:
                    results.append(concept)
            return results

    def recall(
        self,
        limit: int = 10,
        **filters: Any,
    ) -> list[SemanticMemory | CompressedSemantic]:
        """Query concepts with optional filters.

        Supported filters: name, category, min_confidence,
        property_key, property_value, include_compressed.
        """
        self._stats["queries"] = self._stats.get("queries", 0) + 1

        name = filters.get("name")
        category = filters.get("category")
        min_confidence = filters.get("min_confidence")
        prop_key = filters.get("property_key")
        prop_value = filters.get("property_value")
        include_compressed = filters.get("include_compressed", True)

        with self._rwlock.read():
            # Use index if we have a specific filter
            if name is not None:
                candidate_ids = self._context_index.get(f"name:{name}", set())
            elif category is not None:
                candidate_ids = self._context_index.get(f"category:{category}", set())
            else:
                candidate_ids = set(self._concepts.keys())

            results: list[SemanticMemory | CompressedSemantic] = []
            for cid in candidate_ids:
                concept = self._concepts.get(cid)
                if concept is None:
                    continue

                if not include_compressed and isinstance(concept, CompressedSemantic):
                    continue

                # Apply filters
                if name is not None and concept.name != name:
                    continue
                if category is not None and concept.category != category:
                    continue
                if min_confidence is not None and concept.confidence < min_confidence:
                    continue
                if prop_key is not None and isinstance(concept, SemanticMemory):
                    if prop_value is not None:
                        if concept.properties.get(prop_key) != prop_value:
                            continue
                    elif prop_key not in concept.properties:
                        continue

                results.append(concept)

            # Sort by access time (most recent first) and limit
            results.sort(key=lambda c: c.accessed_at, reverse=True)
            return results[:limit]

    def recall_associated(
        self,
        seed_ids: list[str],
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[SemanticMemory | CompressedSemantic, float]]:
        """Spreading activation from seed concepts."""
        decay = kwargs.get("decay", 0.5)
        max_depth = kwargs.get("max_depth", 3)

        with self._rwlock.read():
            activated = self._graph.spreading_activation(
                seed_ids,
                decay=decay,
                max_depth=max_depth,
            )

            results: list[tuple[SemanticMemory | CompressedSemantic, float]] = []
            for cid, score in activated.items():
                if cid in seed_ids:
                    continue
                concept = self._concepts.get(cid)
                if concept is not None:
                    concept.touch()
                    results.append((concept, score))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]

    def save(self, path: str | None = None) -> None:
        """Persist ATL state to JSON."""
        path = path or self.config.persistence_path
        if path is None:
            return

        with self._rwlock.read():
            data = {
                "version": "1.0",
                "concepts": {
                    cid: c.to_dict() for cid, c in self._concepts.items()
                },
                "graph": self._graph.to_dict(),
                "registry": self._semantics.registry.to_dict(),
                "stats": dict(self._stats),
                "compressed_count": self._compressed_count,
            }

        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
        logger.debug("ATL saved to %s (%d concepts)", path, len(self._concepts))

    def load(self, path: str | None = None) -> None:
        """Restore ATL state from JSON."""
        path = path or self.config.persistence_path
        if path is None or not os.path.exists(path):
            return

        with open(path) as f:
            data = json.load(f)

        with self._rwlock.write():
            self._concepts.clear()
            self._context_index.clear()
            self._concept_contexts.clear()
            self._compressed_count = 0

            for cid, c_data in data.get("concepts", {}).items():
                if c_data.get("_compressed"):
                    concept = CompressedSemantic.from_dict(c_data)
                    self._compressed_count += 1
                elif c_data.get("_concept"):
                    concept = Concept.from_dict(c_data)
                else:
                    concept = SemanticMemory.from_dict(c_data)

                self._concepts[cid] = concept
                self._index_concept(cid, concept)

            # Restore graph
            graph_data = data.get("graph")
            if graph_data:
                self._graph = DependencyGraph.from_dict(graph_data)
                self._semantics = Semantics(self._graph)

            # Restore registry
            reg_data = data.get("registry")
            if reg_data:
                self._semantics._registry = RelationshipRegistry.from_dict(reg_data)

            self._stats = data.get("stats", {"concepts_stored": 0, "queries": 0})
            self._compressed_count = data.get("compressed_count", self._compressed_count)

        logger.debug("ATL loaded from %s (%d concepts)", path, len(self._concepts))

    def consolidate(self, **kwargs: Any) -> dict[str, int]:
        """Consolidation cycle: score concepts via MemoryStrategy, compress/remove."""
        strategy = self._get_memory_strategy()
        now = time.time()
        removed = 0
        compressed = 0

        with self._rwlock.write():
            to_remove: list[str] = []
            to_compress: list[str] = []

            for cid, concept in self._concepts.items():
                if isinstance(concept, CompressedSemantic):
                    score = strategy.score_for_retention(concept, now, concept.edge_count)
                    if score < self.config.retention_threshold:
                        to_remove.append(cid)
                    continue

                edge_count = len(self._graph._outgoing.get(cid, []))
                degree = edge_count
                if isinstance(concept, Concept):
                    degree += concept.ref_count()

                score = strategy.score_for_retention(concept, now, degree)

                if score < self.config.retention_threshold:
                    to_remove.append(cid)
                elif score < self.config.compression_threshold:
                    if strategy.should_compress(concept, now, degree):
                        to_compress.append(cid)

            # Remove
            for cid in to_remove:
                mem = self._concepts.get(cid)
                if isinstance(mem, CompressedSemantic):
                    self._compressed_count -= 1
                self._remove_concept(cid)
                if self._scn:
                    self._scn.unregister(cid)
                removed += 1

            # Compress
            for cid in to_compress:
                concept = self._concepts.get(cid)
                if isinstance(concept, SemanticMemory):
                    edge_count = len(self._graph._outgoing.get(cid, []))
                    comp = CompressedSemantic.from_semantic(concept, edge_count)
                    self._concepts[cid] = comp
                    self._compressed_count += 1
                    compressed += 1

        return {"removed": removed, "compressed": compressed}

    def register_capture_callback(
        self,
        callback: Callable[[str, SemanticMemory], None],
    ) -> None:
        self._on_concept_captured.append(callback)

    def register_deletion_callback(
        self,
        callback: Callable[[str], None],
    ) -> None:
        self._on_concept_deleted.append(callback)

    def stats(self) -> dict[str, Any]:
        with self._rwlock.read():
            return {
                **self._stats,
                "total_concepts": len(self._concepts),
                "compressed_count": self._compressed_count,
                "full_count": len(self._concepts) - self._compressed_count,
                "graph_nodes": len(self._graph._nodes),
                "relationship_types": len(self._semantics.list_types()),
            }

    def __len__(self) -> int:
        return len(self._concepts)

    def __iter__(self) -> Iterator[SemanticMemory | CompressedSemantic]:
        yield from self._concepts.values()

    # ------------------------------------------------------------------
    # ATL-Specific Methods
    # ------------------------------------------------------------------

    def find_or_create(
        self,
        name: str,
        category: str,
        definition: str = "",
        provenance: ConceptProvenance = ConceptProvenance.EPISODIC_CONSOLIDATION,
        source_episode_id: str | None = None,
    ) -> tuple[str, bool]:
        """Find existing concept by name or create a new one.

        Returns (concept_id, was_created).
        """
        # Try exact match first
        existing = self.recall(limit=1, name=name, category=category)
        if existing:
            concept = existing[0]
            if isinstance(concept, SemanticMemory) and source_episode_id:
                concept.reinforce(source_episode_id)
            return concept.id, False

        # Create new Concept (extends SemanticMemory with memory_refs)
        concept = Concept(
            id=str(uuid4()),
            timestamp=time.time(),
            name=name,
            category=category,
            definition=definition,
            provenance=provenance,
            source_episode_ids=[source_episode_id] if source_episode_id else [],
        )
        self.store(concept)
        return concept.id, True

    def define_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        weight: float = 1.0,
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Define a typed relationship between concepts."""
        return self._semantics.define(
            source_id, target_id, rel_type, weight, confidence, metadata,
        )

    def find_by_relationship(
        self,
        concept_id: str,
        rel_type: str | None = None,
        direction: str = "outgoing",
        limit: int = 20,
    ) -> list[tuple[str, SemanticRelationship]]:
        """Find concepts related by a specific relationship type."""
        return self._semantics.find(concept_id, rel_type, direction, limit)

    def propose_relationship_type(
        self,
        name: str,
        description: str,
        symmetric: bool = False,
    ) -> bool:
        """Propose a new relationship type (agent-callable)."""
        return self._semantics.propose_type(name, description, symmetric)

    @property
    def semantics(self) -> Semantics:
        """Access the typed relationship manager."""
        return self._semantics

    # ------------------------------------------------------------------
    # Capacity management
    # ------------------------------------------------------------------

    def _get_memory_strategy(self) -> Any:
        """Get the configured memory strategy, wrapped with SCN if available.

        ATL uses a 30-day access window (vs Hippocampus's 7-day) and
        1-week compression age. Concepts decay slower than episodes.
        """
        from maxim.memory.strategies import (
            AccessBasedStrategy,
            TemporalAwareStrategy,
        )

        base = AccessBasedStrategy(
            max_age_without_access=self.config.max_age_without_access,
            compression_age=7 * 86400,
        )

        if self._scn is not None:
            strategy = TemporalAwareStrategy(self._scn, base_strategy=base)
            strategy.prepare()
            return strategy
        return base

    def _evict_one(self) -> None:
        """Evict the lowest-scored concept to make room.

        Uses _get_memory_strategy() for consistent scoring with
        consolidate(). Must be called with write lock held.
        """
        import heapq

        strategy = self._get_memory_strategy()

        now = time.time()
        scored: list[tuple[float, str]] = []

        for cid, concept in self._concepts.items():
            edge_count = len(self._graph._outgoing.get(cid, []))
            degree = edge_count
            if isinstance(concept, Concept):
                degree += concept.ref_count()
            score = strategy.score_for_retention(concept, now, degree)
            heapq.heappush(scored, (score, cid))

        if scored:
            _, evict_id = heapq.heappop(scored)
            mem = self._concepts.get(evict_id)
            if isinstance(mem, CompressedSemantic):
                self._compressed_count -= 1
            self._remove_concept(evict_id)
            if self._scn:
                self._scn.unregister(evict_id)
            logger.debug(
                "ATL capacity eviction: removed %s (at %d/%d)",
                evict_id[:8],
                len(self._concepts),
                self.config.max_concepts,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_concept(
        self,
        concept_id: str,
        concept: SemanticMemory | CompressedSemantic,
    ) -> None:
        """Build context index entries for a concept."""
        keys: set[str] = set()

        if concept.name:
            keys.add(f"name:{concept.name}")
        if concept.category:
            keys.add(f"category:{concept.category}")

        # Index properties (full concepts only)
        if isinstance(concept, SemanticMemory):
            for pk, pv in concept.properties.items():
                keys.add(f"property:{pk}:{pv}")

        for key in keys:
            self._context_index[key].add(concept_id)
        self._concept_contexts[concept_id] = keys

    def _remove_concept(self, concept_id: str) -> None:
        """Remove a concept and clean up (lock must be held)."""
        for key in self._concept_contexts.get(concept_id, set()):
            self._context_index[key].discard(concept_id)
            if not self._context_index[key]:
                del self._context_index[key]
        self._concept_contexts.pop(concept_id, None)

        if self._graph.get_node(concept_id) is not None:
            self._graph.remove_node(concept_id)

        for cb in self._on_concept_deleted:
            try:
                cb(concept_id)
            except Exception as e:
                logger.warning("ATL deletion callback failed: %s", e)

        self._concepts.pop(concept_id, None)


__all__ = ["ATL", "ATLConfig"]
