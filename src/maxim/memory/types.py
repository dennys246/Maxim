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

import math
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from maxim.memory.encoding import EncodingSignals

# S must stay strictly positive: R = exp(-dt/S) divides by it.
_MIN_STORAGE_STRENGTH = 1e-9


# Where an honest activation came from (memory-strength plan Phase 1). Closed on purpose:
# ``MemoryLayer.activate`` rejects anything else, so a typo cannot open a silent new bucket.
ACTIVATION_SOURCES: frozenset[str] = frozenset(
    {
        "enrichment",  # rendered into the LLM's thought response (BioEnrichmentPipeline)
        "tool",  # returned to the LLM by a memory/concept query tool
        "replan",  # rendered into the replan prompt after a failure
        "prediction",  # pattern completion: a past episode used to predict an outcome
        "planner",  # read by a planner as a strategy or reflection
    }
)


def _encoding_fields(record: Any) -> dict[str, Any]:
    """The capture-time encoding record, for episodic ``to_dict`` (memory-strength Phase 2b).

    ``encoding: None`` means "captured before encoding was recorded" -- distinct from a capture that
    recorded ``EncodingSignals.unmeasured(site)``.
    """
    return {"encoding": record.encoding.to_dict() if record.encoding is not None else None}


def _strength_fields(record: Any) -> dict[str, Any]:
    """The capture-time strength stamp, for episodic ``to_dict`` (memory-strength Phase 2c).

    Deliberately NOT on ``MemoryRecord``: a field the base declares but only some subclasses
    serialize is a silent drop waiting to happen (stamp ``S`` on a concept, save, lose it). ``S``
    lives exactly where ``encoding`` lives until ATL's path earns it, and moves to the base WITH its
    serialization in the same commit.

    Read under the record's own lock (2c-3): ``S`` and its anchor are updated together by a credited
    retrieval, and a save that caught the new ``S`` with the old anchor would reload a trace whose
    ``R = exp(-dt/S)`` never happened.
    """
    with record._touch_lock:
        return {
            "storage_strength": record.storage_strength,
            "encoding_tag": record.encoding_tag,
            "novelty_reference_size": record.novelty_reference_size,
            "retrievability_anchor_us": record.retrievability_anchor_us,
            "encoded_at_us": record.encoded_at_us,
            "capture_seq": record.capture_seq,
        }


def update_strength_atomically(
    record: Any,
    compute: "Callable[[float | None, int | None], tuple[float, int] | None]",
) -> bool:
    """Read-modify-write a trace's ``(S, anchor)`` under its OWN lock (memory-strength Phase 2c-3).

    The strength model lives in ``memory/strategies.py``; the ATOMICITY lives here, with the fields.
    ``compute`` is handed the current ``(storage_strength, retrievability_anchor_us)`` and returns
    the new pair, or ``None`` to leave the record untouched (an uncredited activation). It runs
    inside the lock, so it must take no other lock and must not call back into the store.

    Returns whether the record was changed. Lock order is store -> record, the same order
    ``MemoryLayer.activate`` and sleep already take, so this adds no new edge to the lock graph.

    **Whatever changes ``S`` must choose the anchor deliberately** -- which is why the pair is
    returned together rather than ``S`` alone. A retrieval re-anchors to now, and that is what
    makes ``R = 1``. Anything else that changes ``S`` (Phase 3's homeostatic downscale) must pick
    the anchor that PRESERVES ``R``, ``anchor' = now - S'*ln(1/R)``: both ``R`` and the protection
    floor use the current ``S`` as the time constant for an interval that has already elapsed, so
    halving ``S`` without re-anchoring retroactively deflates an interval the trace never lived
    through (measured: a floor of 0.184 becomes 0.068). Review round, Architecture #5.
    """
    with record._touch_lock:
        result = compute(record.storage_strength, record.retrievability_anchor_us)
        if result is None:
            return False
        record.storage_strength, record.retrievability_anchor_us = result
        return True


def _strength_number(value: Any, *, name: str, low: float, high: float, record_id: Any) -> float | None:
    """One persisted strength number, or ``None`` with a warning if disk cannot be trusted for it.

    As strict as ``_encoding_kwargs``, and for a sharper reason: a negative ``storage_strength``
    makes ``R = exp(-dt/S)`` exceed 1, a trace that gets MORE retrievable as it ages, and NaN
    poisons every comparison it touches. Unstamped is a state the strategy handles; nonsense is not.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        number = None
    else:
        number = float(value) if low <= float(value) <= high else None
    if number is None:
        import logging

        logging.getLogger(__name__).warning(
            "memory %s has an out-of-range %s (%r); loading it as never stamped", record_id, name, value
        )
    return number


def _reference_size(value: Any, *, record_id: Any) -> int | None:
    """The set novelty was judged against. Loud like its siblings: a trace that silently loses this
    is indistinguishable from one tagged under the other provenance, which is what it exists to
    prevent. A count, so a float or a numeric string is a writer bug, not a value to coerce."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return int(value)
    import logging

    logging.getLogger(__name__).warning(
        "memory %s has an unusable novelty_reference_size (%r); loading it as not recorded", record_id, value
    )
    return None


def _anchor_us(value: Any, *, record_id: Any) -> int | None:
    """The experience time ``R`` decays from, or ``None`` with a warning if disk cannot be trusted.

    Loud like its siblings, and for the same reason as ``storage_strength``: a negative or
    non-integer anchor makes ``dt = now - anchor`` nonsense, and a trace whose ``R`` is wrong is a
    trace forgotten (or kept) for a reason that never happened. Integer microseconds, matching
    ``experience_clock.UNIT`` exactly -- there is no conversion anywhere on this path.
    """
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return int(value)
    import logging

    logging.getLogger(__name__).warning(
        "memory %s has an unusable retrievability_anchor_us (%r); loading it as never anchored", record_id, value
    )
    return None


def _count_or_us(value: Any, *, name: str, record_id: Any) -> int | None:
    """A non-negative integer field (an experience time in µs, or a sequence number), or ``None``
    with a warning -- loud like its siblings: a trace that cannot say when it happened cannot be
    looked back over, and a silently wrong one would be tagged for a moment it did not share."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return int(value)
    import logging

    logging.getLogger(__name__).warning(
        "memory %s has an unusable %s (%r); loading it as not recorded", record_id, name, value
    )
    return None


def _strength_kwargs(data: dict[str, Any]) -> dict[str, Any]:
    """Load one trace's strength stamp. Absent (every file written before Phase 2c) = never stamped,
    which the strategy reads as "encode it now", NOT as a zero-strength trace."""
    record_id = data.get("id")
    return {
        "storage_strength": _strength_number(
            data.get("storage_strength"),
            name="storage_strength",
            low=_MIN_STORAGE_STRENGTH,
            high=math.inf,
            record_id=record_id,
        ),
        "encoding_tag": _strength_number(
            data.get("encoding_tag"), name="encoding_tag", low=0.0, high=1.0, record_id=record_id
        ),
        "novelty_reference_size": _reference_size(data.get("novelty_reference_size"), record_id=record_id),
        "retrievability_anchor_us": _anchor_us(data.get("retrievability_anchor_us"), record_id=record_id),
        "encoded_at_us": _count_or_us(data.get("encoded_at_us"), name="encoded_at_us", record_id=record_id),
        "capture_seq": _count_or_us(data.get("capture_seq"), name="capture_seq", record_id=record_id),
    }


def _situation_kwargs(data: dict[str, Any]) -> dict[str, Any]:
    """Load one trace's situation (memory-strength Phase 2S-b). Absent (every file written before
    2S-b) or ``None`` = "no situation recorded"; a malformed value warns and loads as not recorded --
    one bad trace must never fail the whole store's load."""
    raw = data.get("situation")
    if raw is None:
        return {"situation": None}
    if (
        isinstance(raw, dict)
        and raw
        and all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in raw.items())
    ):
        return {"situation": dict(raw)}
    import logging

    logging.getLogger(__name__).warning(
        "memory %s has a malformed situation (%r); loading it as not recorded", data.get("id"), raw
    )
    return {"situation": None}


def _encoding_kwargs(data: dict[str, Any]) -> dict[str, Any]:
    """Load one trace's encoding. A malformed record warns and loads as "not recorded" -- one bad
    trace must never fail the whole store's load (``load_state`` parses every record first)."""
    raw = data.get("encoding")
    if raw is None:
        return {"encoding": None}
    try:
        return {"encoding": EncodingSignals.from_dict(raw)}
    except (TypeError, ValueError, AttributeError) as e:
        import logging

        logging.getLogger(__name__).warning(
            "memory %s has a malformed encoding record (%s); loading it as not recorded", data.get("id"), e
        )
        return {"encoding": None}


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
            "math_context": ([m.to_dict() for m in self.math_context] if self.math_context else None),
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
            math_context=([MathContextEntry.from_dict(m) for m in math_ctx] if math_ctx else None),
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
    # Decision provenance (why this action was chosen)
    decision_rationale: str = ""
    tool_alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "observations": self.observations,
            "cli_input": self.cli_input,
            "transcript": self.transcript,
            "detected_objects": self.detected_objects,
            "detected_people": self.detected_people,
            "salience": self.salience,
            "novelty": self.novelty,
            "decision_rationale": self.decision_rationale,
            "tool_alternatives": self.tool_alternatives,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Perception:
        """Deserialize from storage."""
        return cls(
            observations=data.get("observations", {}),
            cli_input=data.get("cli_input"),
            transcript=data.get("transcript"),
            detected_objects=data.get("detected_objects", []),
            detected_people=data.get("detected_people", []),
            salience=data.get("salience", 0.5),
            novelty=data.get("novelty", 0.5),
            decision_rationale=data.get("decision_rationale", ""),
            tool_alternatives=data.get("tool_alternatives", []),
        )


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "state_ref": self.state_ref,
            "active_goal": self.active_goal,
            "active_mode": self.active_mode,
            "memory_context": self.memory_context,
            "fear_level": self.fear_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Context:
        """Deserialize from storage."""
        return cls(
            state_ref=data.get("state_ref"),
            active_goal=data.get("active_goal"),
            active_mode=data.get("active_mode", "observe"),
            memory_context=data.get("memory_context"),
            fear_level=data.get("fear_level", 0.0),
        )


@dataclass
class Decision:
    """What Maxim decided to do and why."""

    intent: dict[str, Any] = field(default_factory=dict)  # {"goal": "...", "confidence": 0.9}
    reasoning: str | None = None
    alternatives_considered: list[dict[str, Any]] = field(default_factory=list)
    plan: list[dict[str, Any]] | None = None  # Multi-step plan if any
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "intent": self.intent,
            "reasoning": self.reasoning,
            "alternatives_considered": self.alternatives_considered,
            "plan": self.plan,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Decision:
        """Deserialize from storage."""
        return cls(
            intent=data.get("intent", {}),
            reasoning=data.get("reasoning"),
            alternatives_considered=data.get("alternatives_considered", []),
            plan=data.get("plan"),
            confidence=data.get("confidence", 1.0),
        )


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "execution_time_ms": self.execution_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        """Deserialize from storage."""
        return cls(
            tool_name=data.get("tool_name", ""),
            tool_params=data.get("tool_params", {}),
        )


@dataclass
class Outcome:
    """What happened as a result of the action."""

    success: bool = False
    result: Any = None
    error: str | None = None
    evaluations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "evaluations": self.evaluations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Outcome:
        """Deserialize from storage."""
        return cls(
            success=data.get("success", False),
            result=data.get("result"),
            error=data.get("error"),
            evaluations=data.get("evaluations", []),
        )


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

    # Use-based consolidation (Stage 7): pressure-based SHORT_TERM → LONG_TERM
    promotion_pressure: float = 0.0
    last_scored_at: float = 0.0  # wall-clock timestamp of last scoring
    access_contexts: deque[str] = field(default_factory=lambda: deque(maxlen=10), repr=False, compare=False)

    # Honest activation (memory-strength plan Phase 1): counts only USE -- content that reached a
    # prompt, a prediction or a decision -- never bookkeeping reads. Separate from access_count on
    # purpose: nothing in the default retention path reads these, so recording them changes no
    # behaviour. A MASSED, un-deduplicated lifetime tally (deliberation re-renders the same top 3
    # every cycle): NOT a strength or importance signal. Do not rank, promote or protect on it --
    # that rebuilds access_count's use-based immortality. Phase 2 hooks MemoryLayer.activate
    # EVENTS; this count stays a diagnostic.
    activation_count: int = 0
    activation_sources: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    # Thread-safe access tracking
    _touch_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def touch(self) -> None:
        """Update access tracking (called on recall). Thread-safe."""
        with self._touch_lock:
            self.accessed_at = time.time()
            self.access_count += 1

    def activate(self, source: str) -> None:
        """Record one honest activation (this record was USED). Thread-safe.

        ``source`` must be in ``ACTIVATION_SOURCES`` -- checked here, in the type, so no path can
        open a bucket by typo. Uses the record's own lock, like ``touch()``. The WHEN of an
        activation (a tick on the experience clock) arrives with that clock in Phase 2.
        """
        if source not in ACTIVATION_SOURCES:
            raise ValueError(f"unknown activation source {source!r}; expected one of {sorted(ACTIVATION_SOURCES)}")
        with self._touch_lock:
            self.activation_count += 1
            self.activation_sources[source] = self.activation_sources.get(source, 0) + 1

    def _activation_fields(self) -> dict[str, Any]:
        """The activation state, for every subclass's ``to_dict`` (one definition, not seven)."""
        with self._touch_lock:  # count and sources move together; a save must not split them
            return {
                "activation_count": self.activation_count,
                "activation_sources": dict(self.activation_sources),
            }

    @staticmethod
    def _activation_kwargs(data: dict[str, Any]) -> dict[str, Any]:
        """Constructor kwargs from a persisted dict; files written before Phase 1 load as zero."""
        return {
            "activation_count": int(data.get("activation_count", 0)),
            "activation_sources": dict(data.get("activation_sources", {})),
        }

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
    # Carried from the full episode (memory-strength Phase 2b); see EpisodicMemory.encoding.
    encoding: EncodingSignals | None = field(default=None, repr=False, compare=False)


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

    # Carried from the full episode (Phase 2c): compression must not reset how well-learned a trace
    # is. HERE and not on CompressedRecord: its other subclasses (CompressedSemantic,
    # CompressedMathMemory) do not serialize these, and a base field only some subclasses persist is
    # the silent drop _strength_fields' own docstring argues against. ATL's forms get S with their
    # serialization, in the same commit.
    storage_strength: float | None = field(default=None, repr=False, compare=False)
    encoding_tag: float | None = field(default=None, repr=False, compare=False)
    # What novelty was judged against when this tag was computed. Recorded so a trace can say which
    # reference set it used: the Hippocampus' own trace count today, the novelty producer's set once
    # 2b-iii records it (where 2b-i's review put it). Without this the store would hold tags of two
    # provenances with nothing marking which is which.
    novelty_reference_size: int | None = field(default=None, repr=False, compare=False)
    # The experience time R decays FROM (memory-strength Phase 2c-3), in the experience clock's own
    # integer microseconds -- ``experience_clock.UNIT``, the same unit S is carried in, so no
    # conversion exists on this path to get wrong. Stamped at capture and reset by each CREDITED
    # retrieval (that is what "R = 1" means); ``None`` = a trace that predates the anchor, which the
    # store re-anchors at load rather than leaving immortal.
    retrievability_anchor_us: int | None = field(default=None, repr=False, compare=False)
    # WHEN this trace happened, in experience µs (memory-strength Phase 2d-1) -- immutable, unlike the
    # anchor above, which a credited retrieval moves. Stamped at the moment of capture: at ENQUEUE for
    # the async loop path, not when the worker gets to it. The look-back (retroactive tagging) windows
    # over it. ``capture_seq`` orders captures that share one loop pass's timestamp (a per-store
    # counter, resumed past the saved maximum on load). ``None`` = captured before 2d-1.
    encoded_at_us: int | None = field(default=None, repr=False, compare=False)
    capture_seq: int | None = field(default=None, repr=False, compare=False)

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
            **MemoryRecord._activation_kwargs(memory._activation_fields()),
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
            encoding=memory.encoding,
            # Through the locked reader, like ``_activation_fields`` above: four unlocked reads
            # could catch a credited retrieval mid-write and freeze a NEW S beside an OLD anchor
            # into the compressed record that replaces this episode -- a torn pair that is then
            # persisted, unlike a torn save, which the next save corrects (review round, Executor
            # #3). ``_strength_fields``' keys are exactly these constructor kwargs.
            **_strength_fields(memory),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            **self._activation_fields(),
            "id": self.id,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "long_term": self.long_term,
            "consolidated_at": self.consolidated_at,
            "promotion_pressure": self.promotion_pressure,
            "last_scored_at": self.last_scored_at,
            "access_contexts": list(self.access_contexts),
            "goal": self.goal,
            "tool_name": self.tool_name,
            "success": self.success,
            "had_user_input": self.had_user_input,
            "object_count": self.object_count,
            "novelty": self.novelty,
            "salience": self.salience,
            "edge_count": self.edge_count,
            **_encoding_fields(self),
            **_strength_fields(self),
            "_compressed": True,  # Marker for deserialization
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompressedMemory":
        """Deserialize from storage."""
        return cls(
            **MemoryRecord._activation_kwargs(data),
            id=data["id"],
            timestamp=data["timestamp"],
            run_id=data.get("run_id", ""),
            created_at=data.get("created_at", data["timestamp"]),
            accessed_at=data.get("accessed_at", data["timestamp"]),
            access_count=data.get("access_count", 1),
            long_term=data.get("long_term", False),
            consolidated_at=data.get("consolidated_at"),
            promotion_pressure=data.get("promotion_pressure", 0.0),
            last_scored_at=data.get("last_scored_at", 0.0),
            access_contexts=deque(data.get("access_contexts", []), maxlen=10),
            goal=data.get("goal"),
            tool_name=data.get("tool_name", ""),
            success=data.get("success", False),
            had_user_input=data.get("had_user_input", False),
            object_count=data.get("object_count", 0),
            novelty=data.get("novelty", 0.5),
            salience=data.get("salience", 0.5),
            edge_count=data.get("edge_count", 0),
            **_encoding_kwargs(data),
            **_strength_kwargs(data),
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

    # Extensible metadata bag — used by Mother Maxim for domain_tags,
    # contribution_source, witness_count, tenant_id, deidentification_model.
    # Adding this pre-publication avoids migration for persisted memories.
    metadata: dict[str, Any] = field(default_factory=dict)

    # What this trace was encoded WITH (memory-strength Phase 2b): the importance signals present at
    # capture and the site that captured it. Write-only until the Phase 2c strength strategy reads
    # it. None = captured before encoding was recorded.
    encoding: EncodingSignals | None = field(default=None, repr=False, compare=False)

    # How well-learned this trace is (memory-strength Phase 2c): S0 = s_base * (1 + k * tag) at
    # capture, and the tag it was computed from, kept so a survivor can say why it survived. Read by
    # ``StrengthStrategy`` (Phase 2c-3) and by nothing else -- the default retention path does not
    # name them. None = never stamped.
    storage_strength: float | None = field(default=None, repr=False, compare=False)
    encoding_tag: float | None = field(default=None, repr=False, compare=False)
    # What novelty was judged against when this tag was computed. Recorded so a trace can say which
    # reference set it used: the Hippocampus' own trace count today, the novelty producer's set once
    # 2b-iii records it (where 2b-i's review put it). Without this the store would hold tags of two
    # provenances with nothing marking which is which.
    novelty_reference_size: int | None = field(default=None, repr=False, compare=False)
    # The experience time R decays FROM (memory-strength Phase 2c-3), in the experience clock's own
    # integer microseconds -- ``experience_clock.UNIT``, the same unit S is carried in, so no
    # conversion exists on this path to get wrong. Stamped at capture and reset by each CREDITED
    # retrieval (that is what "R = 1" means); ``None`` = a trace that predates the anchor, which the
    # store re-anchors at load rather than leaving immortal.
    retrievability_anchor_us: int | None = field(default=None, repr=False, compare=False)
    # WHEN this trace happened, in experience µs (memory-strength Phase 2d-1) -- immutable, unlike the
    # anchor above, which a credited retrieval moves. Stamped at the moment of capture: at ENQUEUE for
    # the async loop path, not when the worker gets to it. The look-back (retroactive tagging) windows
    # over it. ``capture_seq`` orders captures that share one loop pass's timestamp (a per-store
    # counter, resumed past the saved maximum on load). ``None`` = captured before 2d-1.
    encoded_at_us: int | None = field(default=None, repr=False, compare=False)
    capture_seq: int | None = field(default=None, repr=False, compare=False)
    # The situation this trace happened in (memory-strength Phase 2S-b, #848): the loop's substrate
    # clusters at capture, ``{modality: EC cluster id}`` (interoception / audio / world). The EC node
    # ids ARE ATL concept ids, so ConceptExtractor links those concepts to the trace -- the
    # substrate-native cue that pattern completion needs, since survival percepts carry no text.
    # ``None`` = no situation recorded (a non-loop capture, a loop path that computes none, or a file
    # written before 2S-b). Episodic only: compression drops it, as it drops the concept refs. Not
    # EC's ``SituationSignature`` (a text signature keyed by memory id) -- this is substrate clusters.
    situation: dict[str, str] | None = field(default=None, repr=False, compare=False)

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
            kws.update(w.lower() for w in self.perception.cli_input.split() if len(w) > 2)
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
            **self._activation_fields(),
            "id": self.id,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "long_term": self.long_term,
            "consolidated_at": self.consolidated_at,
            "promotion_pressure": self.promotion_pressure,
            "last_scored_at": self.last_scored_at,
            "access_contexts": list(self.access_contexts),
            "perception": self.perception.to_dict(),
            "context": self.context.to_dict(),
            "decision": self.decision.to_dict(),
            "action": self.action.to_dict(),
            "outcome": self.outcome.to_dict(),
            "metadata": self.metadata,
            "situation": dict(self.situation) if self.situation is not None else None,
            **_encoding_fields(self),
            **_strength_fields(self),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodicMemory:
        """Deserialize from storage."""
        return cls(
            **MemoryRecord._activation_kwargs(data),
            id=data["id"],
            timestamp=data["timestamp"],
            run_id=data.get("run_id", ""),
            created_at=data.get("created_at", data["timestamp"]),
            accessed_at=data.get("accessed_at", data["timestamp"]),
            access_count=data.get("access_count", 1),
            long_term=data.get("long_term", False),
            consolidated_at=data.get("consolidated_at"),
            promotion_pressure=data.get("promotion_pressure", 0.0),
            last_scored_at=data.get("last_scored_at", 0.0),
            access_contexts=deque(data.get("access_contexts", []), maxlen=10),
            perception=Perception.from_dict(data.get("perception", {})),
            context=Context.from_dict(data.get("context", {})),
            decision=Decision.from_dict(data.get("decision", {})),
            action=Action.from_dict(data.get("action", {})),
            outcome=Outcome.from_dict(data.get("outcome", {})),
            metadata=data.get("metadata", {}),
            **_situation_kwargs(data),
            **_encoding_kwargs(data),
            **_strength_kwargs(data),
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
