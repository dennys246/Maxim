"""Memory interfaces and backends."""

from __future__ import annotations

from maxim.memory.base import InMemoryMemory, Memory, SimpleRecord

# Memory record ABCs and episodic types
from maxim.memory.types import (
    Action,
    CompressedMemory,
    CompressedRecord,
    Context,
    Decision,
    EpisodicMemory,
    MemoryRecord,
    Outcome,
    Perception,
)
from maxim.memory.state_store import StateStore
from maxim.memory.rwlock import RWLock
from maxim.memory.layer import MemoryLayer
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.strategies import (
    AccessBasedStrategy,
    CompositeStrategy,
    ImportanceBasedStrategy,
    MemoryStrategy,
    TemporalAwareStrategy,
)

# Semantic memory (ATL)
from maxim.memory.semantic_types import (
    CompressedSemantic,
    ConceptProvenance,
    RelationshipRegistry,
    RelationshipType,
    SemanticMemory,
    SemanticRelationship,
)
from maxim.memory.semantics import Semantics
from maxim.memory.atl import ATL, ATLConfig

# Cross-layer and promotion
from maxim.memory.cross_layer import (
    CrossLayerEdge,
    CrossLayerEdgeType,
    CrossLayerGraph,
)
from maxim.memory.semantic_promoter import (
    PromotionCandidate,
    PromotionConfig,
    PromotionSource,
    SemanticPromoter,
)

__all__ = [
    # Base memory
    "InMemoryMemory",
    "Memory",
    "SimpleRecord",
    # Memory record ABCs
    "CompressedRecord",
    "MemoryRecord",
    "MemoryLayer",
    # Episodic memory types
    "Action",
    "CompressedMemory",
    "Context",
    "Decision",
    "EpisodicMemory",
    "Outcome",
    "Perception",
    # Hippocampus system
    "Hippocampus",
    "HippocampusConfig",
    "RWLock",
    "StateStore",
    # Memory strategies
    "AccessBasedStrategy",
    "CompositeStrategy",
    "ImportanceBasedStrategy",
    "MemoryStrategy",
    "TemporalAwareStrategy",
    # Semantic memory (ATL)
    "ATL",
    "ATLConfig",
    "CompressedSemantic",
    "ConceptProvenance",
    "RelationshipRegistry",
    "RelationshipType",
    "SemanticMemory",
    "SemanticRelationship",
    "Semantics",
    # Cross-layer and promotion
    "CrossLayerEdge",
    "CrossLayerEdgeType",
    "CrossLayerGraph",
    "PromotionCandidate",
    "PromotionConfig",
    "PromotionSource",
    "SemanticPromoter",
]
