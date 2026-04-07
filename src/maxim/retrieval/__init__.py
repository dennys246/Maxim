"""Retrieval module - Extensible memory retrieval signals.

Provides a plugin architecture for different retrieval strategies that can
be combined to surface relevant memories from multiple perspectives.
"""

from maxim.retrieval.signals import (
    RetrievalCandidate,
    RetrievalSignal,
)
from maxim.retrieval.orchestrator import RetrievalOrchestrator
from maxim.retrieval.temporal_signal import TemporalReferenceSignal

__all__ = [
    "RetrievalCandidate",
    "RetrievalSignal",
    "RetrievalOrchestrator",
    "TemporalReferenceSignal",
]
