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
    MathContextEntry,
    MemoryRecord,
    Outcome,
    Perception,
    PredictedOutcome,
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

# Store protocols and file backends
from maxim.memory.store import (
    CausalStore,
    EpisodicStore,
    FileCausalStore,
    FileEpisodicStore,
    FileSemanticStore,
    SemanticStore,
)

# Concept type (extends SemanticMemory)
from maxim.memory.semantic_types import Concept

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
    "MathContextEntry",
    "MemoryRecord",
    "MemoryLayer",
    "PredictedOutcome",
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
    # Store protocols and file backends
    "CausalStore",
    "EpisodicStore",
    "FileCausalStore",
    "FileEpisodicStore",
    "FileSemanticStore",
    "SemanticStore",
    # Concept type
    "Concept",
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
