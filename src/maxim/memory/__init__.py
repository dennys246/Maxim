"""Memory interfaces and backends."""

from __future__ import annotations

from maxim.memory.base import InMemoryMemory, Memory, MemoryRecord

# Episodic memory system (Hippocampus)
from maxim.memory.types import (
    Action,
    CompressedMemory,
    Context,
    Decision,
    EpisodicMemory,
    Outcome,
    Perception,
)
from maxim.memory.state_store import StateStore
from maxim.memory.rwlock import RWLock
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.strategies import (
    AccessBasedStrategy,
    CompositeStrategy,
    ImportanceBasedStrategy,
    MemoryStrategy,
    TemporalAwareStrategy,
)

__all__ = [
    # Base memory
    "InMemoryMemory",
    "Memory",
    "MemoryRecord",
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
]
