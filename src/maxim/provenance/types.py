"""Core provenance types — PipelineStage, ProvenanceVerbosity, Ref, Entry, Trace.

All types support to_dict()/from_dict() for JSONL serialization.
ProvenanceTrace is thread-safe (per-trace lock on add/complete).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class PipelineStage(Enum):
    """Canonical pipeline stages for provenance tracking."""

    PERCEPTION = "perception"
    RECALL = "recall"
    DECISION = "decision"
    ACTION = "action"
    OUTCOME = "outcome"
    FORMATION = "formation"
    ENRICHMENT = "enrichment"
    LEARNING = "learning"
    CONSOLIDATION = "consolidation"


class ProvenanceVerbosity(IntEnum):
    """Provenance output verbosity levels."""

    OFF = 0  # No collection or output
    COMPACT = 1  # Stage summaries + source references
    VERBOSE = 2  # Full metadata, alternatives, timing


@dataclass
class ProvenanceRef:
    """Reference to a specific memory/concept/record.

    Uses the layer:id convention shared with A7 BioContext output.
    After A7.0a-b, all records include their ID in to_context_dict(),
    so refs can always be resolved to text.
    """

    layer: str  # "atl", "hippocampus", "angular_gyrus", "nac"
    id: str  # Record ID (UUID)
    label: str  # Human-readable: "kitchen (location)"
    confidence: float = 1.0

    def __str__(self) -> str:
        return (
            f"`{self.layer}:{self.id[:8]}` {self.label} "
            f"(confidence: {self.confidence:.2f})"
        )

    def short(self) -> str:
        """Compact format for verbosity=1."""
        return f"`{self.layer}:{self.id[:8]}` {self.label}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "id": self.id,
            "label": self.label,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceRef:
        return cls(
            layer=data["layer"],
            id=data["id"],
            label=data["label"],
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ProvenanceEntry:
    """Single step in a provenance trace."""

    timestamp: float
    stage: PipelineStage
    component: str  # "concept_extractor", "memory_agent", etc.
    action: str  # "resolved 3 concepts", "proposed navigate_to"
    sources: list[ProvenanceRef] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "stage": self.stage.value,
            "component": self.component,
            "action": self.action,
            "sources": [r.to_dict() for r in self.sources],
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceEntry:
        return cls(
            timestamp=data["timestamp"],
            stage=PipelineStage(data["stage"]),
            component=data["component"],
            action=data["action"],
            sources=[ProvenanceRef.from_dict(s) for s in data.get("sources", [])],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProvenanceTrace:
    """Full trace of a single agent cycle (perception->outcome).

    Thread-safe: add() and complete() are protected by a per-trace lock
    because record() releases the collector lock before calling trace.add().
    """

    trace_id: str
    run_id: str
    session_id: str
    started_at: float
    entries: list[ProvenanceEntry] = field(default_factory=list)
    completed: bool = False
    _persisted: bool = field(default=False, repr=False, compare=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def add(
        self,
        stage: PipelineStage,
        component: str,
        action: str,
        sources: list[ProvenanceRef] | None = None,
        confidence: float = 1.0,
        **metadata: Any,
    ) -> None:
        with self._lock:
            self.entries.append(
                ProvenanceEntry(
                    timestamp=time.time(),
                    stage=stage,
                    component=component,
                    action=action,
                    sources=sources or [],
                    confidence=confidence,
                    metadata=metadata,
                )
            )

    def complete(self) -> None:
        with self._lock:
            self.completed = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "trace",
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "completed": self.completed,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceTrace:
        trace = cls(
            trace_id=data["trace_id"],
            run_id=data["run_id"],
            session_id=data.get("session_id", "unknown"),
            started_at=data["started_at"],
            completed=data.get("completed", True),
        )
        trace.entries = [
            ProvenanceEntry.from_dict(e) for e in data.get("entries", [])
        ]
        return trace
