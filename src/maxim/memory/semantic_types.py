"""Semantic memory type definitions for the ATL (Anterior Temporal Lobe).

Defines SemanticMemory, CompressedSemantic, and supporting types for
the semantic concept memory layer.  SemanticMemory captures *what things
mean* — facts, concepts, entities — stripped of the when/where of learning.

Brain mapping: The anterior temporal lobe integrates multimodal information
into unified semantic representations.  Lesions cause semantic dementia —
loss of concept knowledge while episodic memory is preserved.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from maxim.memory.types import CompressedRecord, MemoryRecord


class ConceptProvenance(Enum):
    """How a semantic concept was acquired."""

    EPISODIC_CONSOLIDATION = auto()  # Extracted from repeated episodes via NAc
    DIRECT_INGESTION = auto()        # RAG / document ingestion
    AGENT_INFERENCE = auto()         # Agent proposed a new concept
    HYBRID = auto()                  # Multiple sources


class RelationshipType(Enum):
    """Base semantic relationship types."""

    IS_A = auto()
    HAS_PART = auto()
    CAUSES = auto()
    PROPERTY_OF = auto()
    RELATED_TO = auto()
    ASSOCIATES = auto()


@dataclass
class SemanticRelationship:
    """A typed relationship between two concepts."""

    source_id: str
    target_id: str
    relationship_type: str  # String from RelationshipRegistry
    weight: float = 1.0
    confidence: float = 0.5
    provenance: ConceptProvenance = ConceptProvenance.EPISODIC_CONSOLIDATION
    metadata: dict[str, Any] = field(default_factory=dict)


class RelationshipRegistry:
    """Extensible runtime registry of relationship types.

    Pre-registers base RelationshipType values and math/statistical types.
    Agents can propose new types at runtime via register().
    """

    def __init__(self) -> None:
        self._types: dict[str, dict[str, Any]] = {}
        self._seed_base_types()
        self._seed_math_types()

    def _seed_base_types(self) -> None:
        """Register base semantic relationship types."""
        base = [
            ("IS_A", "Subsumption / taxonomy", False),
            ("HAS_PART", "Part-whole relationship", False),
            ("CAUSES", "Causal relationship", False),
            ("PROPERTY_OF", "Property attribution", False),
            ("RELATED_TO", "Generic association", True),
            ("ASSOCIATES", "Associative link", True),
        ]
        for name, desc, symmetric in base:
            self._types[name] = {
                "description": desc,
                "symmetric": symmetric,
                "builtin": True,
            }

    def _seed_math_types(self) -> None:
        """Register math/statistical relationship types."""
        math_types = [
            ("IS_PROPORTIONAL_TO", "Two metrics scale together", True),
            ("CORRELATES_WITH", "Statistical correlation confirmed by AG", True),
            ("IS_INVERSE_OF", "Inverse relationship", True),
            ("TRANSFORMS_TO", "Value transforms via known operation", False),
            ("TRENDS_WITH", "Two metrics trend same/opposite direction", True),
            ("PHASE_LOCKED_TO", "Activity phase-locked to SCN oscillator", False),
            ("PREDICTS", "One pattern predicts another", False),
        ]
        for name, desc, symmetric in math_types:
            self._types[name] = {
                "description": desc,
                "symmetric": symmetric,
                "builtin": True,
            }

    def register(
        self,
        name: str,
        description: str,
        symmetric: bool = False,
    ) -> bool:
        """Register a new relationship type at runtime.

        Returns True if newly registered, False if already exists.
        """
        if name in self._types:
            return False
        self._types[name] = {
            "description": description,
            "symmetric": symmetric,
            "builtin": False,
        }
        return True

    def get(self, name: str) -> dict[str, Any] | None:
        """Get relationship type info, or None if not registered."""
        return self._types.get(name)

    def is_symmetric(self, name: str) -> bool:
        """Check if a relationship type is symmetric."""
        info = self._types.get(name)
        return info["symmetric"] if info else False

    def all_types(self) -> list[str]:
        """List all registered relationship type names."""
        return list(self._types.keys())

    def to_dict(self) -> dict[str, Any]:
        """Serialize registry for persistence."""
        return {
            "types": {
                name: {**info}
                for name, info in self._types.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationshipRegistry:
        """Deserialize registry from persistence."""
        registry = cls()
        for name, info in data.get("types", {}).items():
            if name not in registry._types:
                registry._types[name] = info
        return registry


@dataclass
class SemanticMemory(MemoryRecord):
    """A concept, fact, or entity in semantic memory.

    The ATL's equivalent of EpisodicMemory.  Captures *what things mean*
    rather than *what happened*.

    Inherits from MemoryRecord: id, timestamp, created_at, accessed_at,
    access_count, long_term, consolidated_at, touch().
    """

    name: str = ""
    definition: str = ""
    category: str = ""  # "object", "person", "action", "fact", "causal_pattern"
    properties: dict[str, Any] = field(default_factory=dict)
    provenance: ConceptProvenance = ConceptProvenance.EPISODIC_CONSOLIDATION
    source_episode_ids: list[str] = field(default_factory=list)
    source_document: str | None = None
    confidence: float = 0.5
    reinforcement_count: int = 1
    embedding_text: str = ""

    def reinforce(self, episode_id: str) -> None:
        """Reinforce concept with a new confirming episode."""
        self.reinforcement_count += 1
        if episode_id not in self.source_episode_ids:
            self.source_episode_ids.append(episode_id)
        # Confidence: min(0.99, 0.5 + 0.1 * sqrt(reinforcement_count))
        self.confidence = min(0.99, 0.5 + 0.1 * math.sqrt(self.reinforcement_count))
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON persistence."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "long_term": self.long_term,
            "consolidated_at": self.consolidated_at,
            "name": self.name,
            "definition": self.definition,
            "category": self.category,
            "properties": self.properties,
            "provenance": self.provenance.name,
            "source_episode_ids": self.source_episode_ids,
            "source_document": self.source_document,
            "confidence": self.confidence,
            "reinforcement_count": self.reinforcement_count,
            "embedding_text": self.embedding_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticMemory:
        """Deserialize from JSON."""
        prov_str = data.get("provenance", "EPISODIC_CONSOLIDATION")
        try:
            provenance = ConceptProvenance[prov_str]
        except KeyError:
            provenance = ConceptProvenance.EPISODIC_CONSOLIDATION

        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            created_at=data.get("created_at", data["timestamp"]),
            accessed_at=data.get("accessed_at", data["timestamp"]),
            access_count=data.get("access_count", 1),
            long_term=data.get("long_term", False),
            consolidated_at=data.get("consolidated_at"),
            name=data.get("name", ""),
            definition=data.get("definition", ""),
            category=data.get("category", ""),
            properties=data.get("properties", {}),
            provenance=provenance,
            source_episode_ids=data.get("source_episode_ids", []),
            source_document=data.get("source_document"),
            confidence=data.get("confidence", 0.5),
            reinforcement_count=data.get("reinforcement_count", 1),
            embedding_text=data.get("embedding_text", ""),
        )


@dataclass
class CompressedSemantic(CompressedRecord):
    """Lightweight compressed form of a semantic memory (~100 bytes).

    Keeps essential fields for queries and scoring.
    Drops: definition, properties, source_episode_ids, embedding_text.

    Inherits from CompressedRecord: id, timestamp, created_at, accessed_at,
    access_count, long_term, consolidated_at, edge_count, touch().
    """

    name: str = ""
    category: str = ""
    confidence: float = 0.5
    reinforcement_count: int = 1
    provenance: ConceptProvenance = ConceptProvenance.EPISODIC_CONSOLIDATION

    @classmethod
    def from_semantic(
        cls,
        memory: SemanticMemory,
        edge_count: int = 0,
    ) -> CompressedSemantic:
        """Compress a full SemanticMemory to lightweight form."""
        return cls(
            id=memory.id,
            timestamp=memory.timestamp,
            created_at=memory.created_at,
            accessed_at=memory.accessed_at,
            access_count=memory.access_count,
            long_term=memory.long_term,
            consolidated_at=memory.consolidated_at,
            edge_count=edge_count,
            name=memory.name,
            category=memory.category,
            confidence=memory.confidence,
            reinforcement_count=memory.reinforcement_count,
            provenance=memory.provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "long_term": self.long_term,
            "consolidated_at": self.consolidated_at,
            "edge_count": self.edge_count,
            "name": self.name,
            "category": self.category,
            "confidence": self.confidence,
            "reinforcement_count": self.reinforcement_count,
            "provenance": self.provenance.name,
            "_compressed": True,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompressedSemantic:
        """Deserialize from storage."""
        prov_str = data.get("provenance", "EPISODIC_CONSOLIDATION")
        try:
            provenance = ConceptProvenance[prov_str]
        except KeyError:
            provenance = ConceptProvenance.EPISODIC_CONSOLIDATION

        return cls(
            id=data["id"],
            timestamp=data.get("timestamp", data.get("created_at", 0.0)),
            created_at=data.get("created_at", 0.0),
            accessed_at=data.get("accessed_at", 0.0),
            access_count=data.get("access_count", 1),
            long_term=data.get("long_term", False),
            consolidated_at=data.get("consolidated_at"),
            edge_count=data.get("edge_count", 0),
            name=data.get("name", ""),
            category=data.get("category", ""),
            confidence=data.get("confidence", 0.5),
            reinforcement_count=data.get("reinforcement_count", 1),
            provenance=provenance,
        )


__all__ = [
    "ConceptProvenance",
    "CompressedSemantic",
    "RelationshipRegistry",
    "RelationshipType",
    "SemanticMemory",
    "SemanticRelationship",
]
