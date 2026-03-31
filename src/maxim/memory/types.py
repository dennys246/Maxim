"""Memory record type definitions.

This module defines the base ABCs and concrete data structures for all
memory systems.  The MemoryRecord ABC provides shared tracking fields
(id, timestamps, access counts, long-term flag) that are common across
EpisodicMemory (Hippocampus), MathMemory (Angular Gyrus), and the
future SemanticMemory (ATL).

CompressedRecord extends MemoryRecord for lightweight compressed forms.

PredictedOutcome and MathContextEntry are typed contracts for pattern
completion predictions (ATL graph chaining → MemoryAgent).
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Pattern completion contracts
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MathContextEntry:
    """AG math context for a single property of a concept.

    Typed contract for the math enrichment step in pattern completion.
    Eliminates raw dict with implicit keys.
    """

    name: str  # MathMemory name (e.g. "mug:execution_time_ms")
    verbal: str = ""  # Human-readable label (e.g. "typically ~310ms")
    confidence: float = 0.0
    domain: str = ""  # e.g. "timing", "success_rate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "verbal": self.verbal,
            "confidence": self.confidence,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MathContextEntry:
        return cls(
            name=data["name"],
            verbal=data.get("verbal", ""),
            confidence=data.get("confidence", 0.0),
            domain=data.get("domain", ""),
        )


@dataclass
class PredictedOutcome:
    """A predicted outcome from pattern completion.

    Produced by ATL graph chaining, consumed by MemoryAgent during
    FORMING stage. Typed fields enforce the contract between the
    two systems — no implicit dict key expectations.
    """

    tool: str  # Action tool used in the past
    success: bool  # Whether the past action succeeded
    goal: str | None = None  # Goal from the past decision
    confidence: float = 1.0  # Decision confidence from the past episode
    math_context: list[MathContextEntry] | None = None  # Per-concept layer stats
    source_episode_id: str = ""  # Which episode this prediction came from

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "success": self.success,
            "goal": self.goal,
            "confidence": self.confidence,
            "math_context": (
                [m.to_dict() for m in self.math_context]
                if self.math_context
                else None
            ),
            "source_episode_id": self.source_episode_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictedOutcome:
        math_ctx = data.get("math_context")
        return cls(
            tool=data["tool"],
            success=data["success"],
            goal=data.get("goal"),
            confidence=data.get("confidence", 1.0),
            math_context=(
                [MathContextEntry.from_dict(m) for m in math_ctx]
                if math_ctx
                else None
            ),
            source_episode_id=data.get("source_episode_id", ""),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Episodic memory sub-components
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Perception:
    """What Maxim observed at the start of the loop."""

    observations: dict[str, Any] = field(default_factory=dict)
    cli_input: str | None = None
    transcript: str | None = None
    detected_objects: list[str] = field(default_factory=list)
    detected_people: list[str] = field(default_factory=list)
    salience: float = 0.5
    novelty: float = 0.5


@dataclass
class Context:
    """State context at decision time.

    Note: state_ref is a hash pointing to the full state in StateStore.
    This keeps EpisodicMemory objects lightweight (~200 bytes vs ~2KB with full snapshot).
    """

    state_ref: str | None = None  # Hash reference to StateStore (not full snapshot)
    active_goal: str | None = None
    active_mode: str = "observe"
    memory_context: str | None = None  # Summary from context pool
    fear_level: float = 0.0


@dataclass
class Decision:
    """What Maxim decided to do and why."""

    intent: dict[str, Any] = field(default_factory=dict)  # {"goal": "...", "confidence": 0.9}
    reasoning: str | None = None
    alternatives_considered: list[dict[str, Any]] = field(default_factory=list)
    plan: list[dict[str, Any]] | None = None  # Multi-step plan if any
    confidence: float = 1.0


@dataclass
class Action:
    """The action that was executed."""

    tool_name: str = ""
    tool_params: dict[str, Any] = field(default_factory=dict)
    execution_start: float = 0.0
    execution_end: float = 0.0

    @property
    def execution_time_ms(self) -> float:
        return (self.execution_end - self.execution_start) * 1000


@dataclass
class Outcome:
    """What happened as a result of the action."""

    success: bool = False
    result: Any = None
    error: str | None = None
    evaluations: list[dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Memory record ABC
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MemoryRecord(ABC):
    """Abstract base class for all memory records.

    Provides common tracking fields shared by EpisodicMemory,
    MathMemory, SemanticMemory, and their compressed forms.

    Concrete subclasses must implement to_dict(), from_dict(),
    keywords(), and to_context_dict().
    """

    id: str
    timestamp: float

    # Access tracking for memory decay
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 1

    # Long-term memory flag (consolidated, resistant to removal)
    long_term: bool = False
    consolidated_at: float | None = None

    # Thread-safe access tracking
    _touch_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def touch(self) -> None:
        """Update access tracking (called on recall). Thread-safe."""
        with self._touch_lock:
            self.accessed_at = time.time()
            self.access_count += 1

    @abstractmethod
    def keywords(self) -> set[str]:
        """Extract searchable keywords from this record.

        Used by WorkingMemoryEntry for keyword-based similarity search
        and by MemoryAgent for context building. Each subclass extracts
        keywords from its own structured fields.
        """
        ...

    @abstractmethod
    def to_context_dict(self) -> dict[str, Any]:
        """Format this record for LLM context injection.

        Returns a dict suitable for inclusion in StructuredContext.
        Each subclass formats its own structured fields — no isinstance
        chains needed in build_context().
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON persistence."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        """Deserialize from JSON."""
        ...


@dataclass
class CompressedRecord(MemoryRecord):
    """Abstract base for compressed memory forms.

    Extends MemoryRecord with graph edge count.
    Concrete subclasses: CompressedMemory, CompressedMathMemory.
    """

    edge_count: int = 0


@dataclass
class CompressedMemory(CompressedRecord):
    """Lightweight summary of an old episodic memory.

    Used for long-term storage when full EpisodicMemory data is no longer needed.
    Preserves essential information for basic queries while significantly
    reducing memory footprint (~200 bytes vs ~2.5KB for full EpisodicMemory).

    Compressed memories still appear in queries but have limited data.

    Inherits from CompressedRecord: id, timestamp, created_at, accessed_at,
    access_count, long_term, consolidated_at, edge_count, touch().
    """

    run_id: str = ""

    # Essential decision data (for queries)
    goal: str | None = None
    tool_name: str = ""
    success: bool = False

    # Minimal perception summary
    had_user_input: bool = False
    object_count: int = 0
    novelty: float = 0.5
    salience: float = 0.5

    def keywords(self) -> set[str]:
        """Extract keywords from compressed episodic data."""
        kws: set[str] = set()
        if self.goal:
            kws.update(w.lower() for w in self.goal.split() if len(w) > 2)
        if self.tool_name:
            kws.add(self.tool_name.lower())
        return kws

    def to_context_dict(self) -> dict[str, Any]:
        """Format compressed memory for LLM context."""
        return {
            "id": self.id,
            "type": "compressed_episodic",
            "goal": self.goal,
            "tool": self.tool_name,
            "success": self.success,
            "had_user_input": self.had_user_input,
            "object_count": self.object_count,
        }

    @classmethod
    def from_episodic(cls, memory: "EpisodicMemory", edge_count: int = 0) -> "CompressedMemory":
        """Compress a full EpisodicMemory to lightweight form."""
        return cls(
            id=memory.id,
            timestamp=memory.timestamp,
            run_id=memory.run_id,
            created_at=memory.created_at,
            accessed_at=memory.accessed_at,
            access_count=memory.access_count,
            long_term=memory.long_term,
            consolidated_at=memory.consolidated_at,
            goal=memory.decision.intent.get("goal") or memory.context.active_goal,
            tool_name=memory.action.tool_name,
            success=memory.outcome.success,
            had_user_input=bool(memory.perception.cli_input or memory.perception.transcript),
            object_count=len(memory.perception.detected_objects),
            novelty=memory.perception.novelty,
            salience=memory.perception.salience,
            edge_count=edge_count,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "long_term": self.long_term,
            "consolidated_at": self.consolidated_at,
            "goal": self.goal,
            "tool_name": self.tool_name,
            "success": self.success,
            "had_user_input": self.had_user_input,
            "object_count": self.object_count,
            "novelty": self.novelty,
            "salience": self.salience,
            "edge_count": self.edge_count,
            "_compressed": True,  # Marker for deserialization
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompressedMemory":
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            run_id=data.get("run_id", ""),
            created_at=data.get("created_at", data["timestamp"]),
            accessed_at=data.get("accessed_at", data["timestamp"]),
            access_count=data.get("access_count", 1),
            long_term=data.get("long_term", False),
            consolidated_at=data.get("consolidated_at"),
            goal=data.get("goal"),
            tool_name=data.get("tool_name", ""),
            success=data.get("success", False),
            had_user_input=data.get("had_user_input", False),
            object_count=data.get("object_count", 0),
            novelty=data.get("novelty", 0.5),
            salience=data.get("salience", 0.5),
            edge_count=data.get("edge_count", 0),
        )


@dataclass
class EpisodicMemory(MemoryRecord):
    """A complete agentic loop cycle.

    This is the fundamental unit stored in the Hippocampus.
    Each record captures: observe -> decide -> act -> evaluate.

    Inherits from MemoryRecord: id, timestamp, created_at, accessed_at,
    access_count, long_term, consolidated_at, touch().
    """

    run_id: str = ""

    # The five phases of the loop
    perception: Perception = field(default_factory=Perception)
    context: Context = field(default_factory=Context)
    decision: Decision = field(default_factory=Decision)
    action: Action = field(default_factory=Action)
    outcome: Outcome = field(default_factory=Outcome)

    @property
    def duration_ms(self) -> float:
        return self.action.execution_time_ms

    def keywords(self) -> set[str]:
        """Extract keywords from episodic memory fields."""
        kws: set[str] = set()
        # Objects and people
        kws.update(o.lower() for o in self.perception.detected_objects)
        kws.update(p.lower() for p in self.perception.detected_people)
        # Goal
        goal = self.decision.intent.get("goal") or self.context.active_goal
        if goal:
            kws.update(w.lower() for w in goal.split() if len(w) > 2)
        # Tool
        if self.action.tool_name:
            kws.add(self.action.tool_name.lower())
        # CLI input
        if self.perception.cli_input:
            kws.update(
                w.lower() for w in self.perception.cli_input.split() if len(w) > 2
            )
        return kws

    def to_context_dict(self) -> dict[str, Any]:
        """Format episodic memory for LLM context."""
        return {
            "id": self.id,
            "type": "episodic",
            "detected_objects": self.perception.detected_objects,
            "detected_people": self.perception.detected_people,
            "goal": self.decision.intent.get("goal") or self.context.active_goal,
            "tool": self.action.tool_name,
            "success": self.outcome.success,
            "salience": self.perception.salience,
            "novelty": self.perception.novelty,
            "cli_input": self.perception.cli_input,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "long_term": self.long_term,
            "consolidated_at": self.consolidated_at,
            "perception": {
                "observations": self.perception.observations,
                "cli_input": self.perception.cli_input,
                "transcript": self.perception.transcript,
                "detected_objects": self.perception.detected_objects,
                "detected_people": self.perception.detected_people,
                "salience": self.perception.salience,
                "novelty": self.perception.novelty,
            },
            "context": {
                "state_ref": self.context.state_ref,
                "active_goal": self.context.active_goal,
                "active_mode": self.context.active_mode,
                "memory_context": self.context.memory_context,
                "fear_level": self.context.fear_level,
            },
            "decision": {
                "intent": self.decision.intent,
                "reasoning": self.decision.reasoning,
                "alternatives_considered": self.decision.alternatives_considered,
                "plan": self.decision.plan,
                "confidence": self.decision.confidence,
            },
            "action": {
                "tool_name": self.action.tool_name,
                "tool_params": self.action.tool_params,
                "execution_time_ms": self.action.execution_time_ms,
            },
            "outcome": {
                "success": self.outcome.success,
                "result": self.outcome.result,
                "error": self.outcome.error,
                "evaluations": self.outcome.evaluations,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodicMemory:
        """Deserialize from storage."""
        p = data.get("perception", {})
        c = data.get("context", {})
        d = data.get("decision", {})
        a = data.get("action", {})
        o = data.get("outcome", {})

        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            run_id=data.get("run_id", ""),
            created_at=data.get("created_at", data["timestamp"]),
            accessed_at=data.get("accessed_at", data["timestamp"]),
            access_count=data.get("access_count", 1),
            long_term=data.get("long_term", False),
            consolidated_at=data.get("consolidated_at"),
            perception=Perception(
                observations=p.get("observations", {}),
                cli_input=p.get("cli_input"),
                transcript=p.get("transcript"),
                detected_objects=p.get("detected_objects", []),
                detected_people=p.get("detected_people", []),
                salience=p.get("salience", 0.5),
                novelty=p.get("novelty", 0.5),
            ),
            context=Context(
                state_ref=c.get("state_ref"),
                active_goal=c.get("active_goal"),
                active_mode=c.get("active_mode", "observe"),
                memory_context=c.get("memory_context"),
                fear_level=c.get("fear_level", 0.0),
            ),
            decision=Decision(
                intent=d.get("intent", {}),
                reasoning=d.get("reasoning"),
                alternatives_considered=d.get("alternatives_considered", []),
                plan=d.get("plan"),
                confidence=d.get("confidence", 1.0),
            ),
            action=Action(
                tool_name=a.get("tool_name", ""),
                tool_params=a.get("tool_params", {}),
            ),
            outcome=Outcome(
                success=o.get("success", False),
                result=o.get("result"),
                error=o.get("error"),
                evaluations=o.get("evaluations", []),
            ),
        )


__all__ = [
    "MathContextEntry",
    "MemoryRecord",
    "CompressedRecord",
    "Perception",
    "Context",
    "Decision",
    "Action",
    "Outcome",
    "EpisodicMemory",
    "CompressedMemory",
    "PredictedOutcome",
]
