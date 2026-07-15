"""Message bus and data types for agent communication.

Defines the message types exchanged between agents and the pub/sub bus
for decoupled communication.
"""

from __future__ import annotations

import hashlib
import logging
import math as _math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, overload

if TYPE_CHECKING:
    from maxim.agents.percept_context import Modality, PerceptContext

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class GoalPriority(Enum):
    """Priority levels for proposed goals."""

    CRITICAL = 0  # Safety, urgent user request
    HIGH = 1  # Direct user command
    MEDIUM = 2  # Inferred helpful action
    LOW = 3  # Background learning/understanding
    IDLE = 4  # Default maintenance


class MemoryTier(Enum):
    """Memory storage tier and lifecycle phase.

    FORMING is the eviction-protected phase during the agent pipeline.
    SHORT_TERM and LONG_TERM are standard decay tiers.

    Lifecycle: FORMING → SHORT_TERM → LONG_TERM → consolidated out.

    The old WORKING tier was removed in 0.8 (Working Memory unification).
    Active-reference context now lives in WorkingMemorySet, an Exec-owned
    layer — not a memory tier.  Persisted episodes with tier="working"
    are migrated to SHORT_TERM on load.
    """

    FORMING = "forming"  # Being constructed during pipeline (eviction-protected)
    SHORT_TERM = "short"  # Fast decay, recent context
    LONG_TERM = "long"  # Slow decay, consolidated knowledge


# Allowed forward transitions for MemoryTier. The progression is strictly
# one-way: skipping a tier or reversing direction indicates a bug that would
# silently corrupt consolidation. Enforced by WorkingMemoryEntry.__setattr__.
#
# 0.8: WORKING tier removed. FORMING promotes directly to SHORT_TERM
# (outcome-triggered, same rule — F6).
_TIER_FORWARD_TRANSITIONS: dict[MemoryTier, frozenset[MemoryTier]] = {
    MemoryTier.FORMING: frozenset({MemoryTier.SHORT_TERM}),
    MemoryTier.SHORT_TERM: frozenset({MemoryTier.LONG_TERM}),
    MemoryTier.LONG_TERM: frozenset(),
}


class TierTransitionError(ValueError):
    """Raised when a MemoryTier transition violates the one-way lifecycle."""


def _assert_tier_transition(old: MemoryTier, new: MemoryTier) -> None:
    """Validate a MemoryTier transition.

    Allows no-op (same-tier) writes so that idempotent sweeps don't crash,
    but rejects reversals and non-adjacent skips. The progression is:
    FORMING → SHORT_TERM → LONG_TERM.
    """
    if old == new:
        return
    allowed = _TIER_FORWARD_TRANSITIONS.get(old, frozenset())
    if new not in allowed:
        raise TierTransitionError(
            f"Illegal MemoryTier transition {old.name} → {new.name}; "
            f"legal progression is FORMING → SHORT_TERM → LONG_TERM"
        )


class SubGoalStatus(Enum):
    """Status of a sub-goal."""

    PENDING = auto()
    BLOCKED = auto()  # Waiting on dependencies
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


class FailureStrategy(Enum):
    """What to do when a sub-goal fails."""

    RETRY = auto()  # Retry up to max_retries
    SKIP = auto()  # Skip and continue to next
    ABORT_PARENT = auto()  # Fail the entire parent goal
    ESCALATE = auto()  # Raise priority and retry
    REPLAN = auto()  # Trigger plan-level re-decomposition


class EdgeType(Enum):
    """Type of dependency relationship in graphs."""

    REQUIRES = auto()  # A requires B (B must complete before A)
    ENABLES = auto()  # A enables B (completing A unlocks B)
    INHIBITS = auto()  # A inhibits B (A active means B cannot run)
    ASSOCIATES = auto()  # Bidirectional association (for memories)
    CAUSES = auto()  # A caused B (temporal/causal link)


class StopReason(Enum):
    """Why an agent loop terminated."""

    COMPLETED = "completed"
    MODE_SHUTDOWN = "mode_shutdown"
    MAX_TURNS = "max_turns"
    TOKEN_BUDGET = "token_budget"
    ENERGY_BUDGET = "energy_budget"
    USER_INTERRUPT = "user_interrupt"
    PLAN_FAILED = "plan_failed"
    NO_INTENT = "no_intent"
    LLM_TIMEOUT = "llm_timeout"
    SAFETY_GATE = "safety_gate"


# ── Deliberation intents (Stage 3) ──────────────────────────────────


class DeliberationOutcome(Enum):
    """Kind of output from a deliberation tick."""

    SPEAK = "speak"  # Deliberation converged on user-visible text
    ACTION = "action"  # Deliberation converged on a tool call / goal
    NOOP = "noop"  # Deliberation tick produced no externalizable output


@dataclass(frozen=True)
class SpeakIntent:
    """Emitted when deliberation converges on 'say something now'."""

    content: str
    channel: str = "respond"  # maps to existing respond/speak/say tool
    reason: str = ""  # provenance for review
    deliberation_hops: int = 0  # telemetry


@dataclass(frozen=True)
class ActionIntent:
    """Emitted when deliberation converges on a tool call or goal proposal."""

    tool_name: str
    tool_params: dict = field(default_factory=dict)
    reason: str = ""
    deliberation_hops: int = 0


@dataclass(frozen=True)
class NoOpIntent:
    """Deliberation tick produced no externalizable output.

    Internal state (working memory, Hebbian bindings from deliberation,
    thought accumulation) is updated regardless.  This is NOT an error —
    silence is a valid output of the cognitive loop.
    """

    reason: str = ""
    deliberation_hops: int = 0


# ToolErrorKind lives in tools/base (next to ToolOutput) to avoid the
# tools → agents → default_network → runtime → tools circular import.
# Re-exported here for backward compatibility.
from maxim.tools.base import ToolErrorKind as ToolErrorKind  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes - Perception
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Percept:
    """Structured perception output from PerceptionAgent."""

    timestamp: float
    source: str  # "vision", "transcript", "file_change", "startup", "idle"

    # What was observed
    detections: list[dict[str, Any]] = field(default_factory=list)
    transcript_chunk: str | None = None
    transcript_chunk_index: int | None = None
    file_changed: str | None = None
    cli_input: str | None = None

    # Derived signals
    salience: float = 0.0
    novelty: float = 0.0

    # Classification
    has_voice_command: bool = False
    has_maxim_keyword: bool = False
    hard_override: str | None = None  # "request_sleep", etc.

    # Exploration commands
    explore_command: dict[str, Any] | None = None  # Parsed explore command

    # Generic content and metadata. ``metadata`` remains a free-form
    # escape hatch for non-messaging attributes (pain signal params,
    # YAML scenario passthrough, legacy keys). Messaging framing
    # (channel, sender, thread, subject, latency, agent_id) flows
    # through the typed ``context`` field instead — see F0.4.
    content: str | None = None
    metadata: dict[str, Any] | None = None

    # Raw data for downstream
    raw_transcript_text: str | None = None
    maxim_runtime: dict[str, Any] | None = None

    # Sensory modality tag (optional — None for legacy percepts)
    sensory: Any = None  # SensoryTag | None

    # Typed message framing (F0.4). Optional so construction sites that
    # do not have messaging semantics (vision detection, proprioception)
    # can leave it None. F0.5 populates ``context.agent_id`` at
    # every producer; F0.6 consolidates production into named factories.
    context: "PerceptContext | None" = None

    # Explicit sensory modality (F0.4). Kept alongside ``sensory`` —
    # ``sensory`` carries rich per-modality sub-tags (populated by F0.8)
    # while this literal lets consumers switch on modality without
    # needing a populated SensoryTag.
    modality: "Modality | None" = None

    # Substrate fields (P1). Populated by LinguisticEncoder when the
    # substrate path is active. ``embedding`` is the dense vector from
    # the encoder; ``substrate_node_id`` is the ATL node that EC
    # pattern-completed or separated this percept into.
    embedding: list[float] | None = None
    substrate_node_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for session persistence. Omits large/internal fields."""
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "detections": self.detections,
            "transcript_chunk": self.transcript_chunk,
            "cli_input": self.cli_input,
            "salience": self.salience,
            "novelty": self.novelty,
            "has_voice_command": self.has_voice_command,
            "has_maxim_keyword": self.has_maxim_keyword,
            "hard_override": self.hard_override,
            "content": self.content,
            "metadata": self.metadata,
            "sensory": self.sensory.to_dict() if self.sensory and hasattr(self.sensory, "to_dict") else None,
            "context": self.context.to_dict() if self.context is not None else None,
            "modality": self.modality,
            "substrate_node_id": self.substrate_node_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Percept":
        # Rehydrate typed context separately so nested scn_tag gets
        # reconstructed as CircadianContext rather than a bare dict.
        from maxim.agents.percept_context import PerceptContext

        data = dict(data)
        ctx_raw = data.pop("context", None)
        if ctx_raw is not None and not isinstance(ctx_raw, PerceptContext):
            data["context"] = PerceptContext.from_dict(ctx_raw)
        elif ctx_raw is not None:
            data["context"] = ctx_raw
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    def to_wire_dict(self) -> dict[str, Any]:
        """Serialize for cross-process transport (peer → leader).

        Distinct from :meth:`to_dict`, which is session-persistence on
        the leader. The wire-dict is the contract a network-backed
        :class:`~maxim.simulation.sources.PerceptSource` ships through
        the perception transport (C10 prep, full transport in 1.1 —
        see ``docs/plans/deferred/mesh_perception_transport.md``).

        Carries raw observations the peer captured on-device:
        ``detections`` (vision), ``transcript_chunk_index`` +
        ``raw_transcript_text`` (STT), ``file_changed``,
        ``explore_command``, plus the basic framing every percept has.

        **Excludes by design**:

        - ``embedding``, ``substrate_node_id`` — the leader owns the
          substrate (EC, ATL, LinguisticEncoder). The peer ships raw
          observations; the leader computes substrate references on
          receipt. This is also the bio-fidelity argument — the peer
          is "a sensor," not "a partial cognition."
        - ``salience``, ``novelty`` — these are leader-computed derived
          signals (compared against the leader's memory state). The
          peer cannot compute them without the leader's substrate.
        - ``maxim_runtime`` — leader-internal runtime fields.

        **Versioning.** The wire-dict carries ``_format_version`` at
        root (CC1 contract via :mod:`maxim.utils.format_version`). The
        wire-dict path is versioned independently of the session-dict
        path — bump the version on either side without touching the
        other. The ``MeshMessage`` envelope's ``protocol_version``
        gates the mesh transport as a whole; ``_format_version`` here
        gates this payload's shape specifically so future evolution
        (new optional field, drop a field) is operator-visible at the
        payload layer without forcing a full mesh-protocol bump.

        **Large-frame upgrade path (plan Q6).** When the 1.1+ work
        needs to ship raw video / audio frames too large for inline
        JSON, the agreed convention is ``metadata["blob_ref"]`` — a
        string identifier the leader resolves via a separate
        blob-fetch endpoint. This requires no wire-dict shape change:
        ``metadata`` is already on the wire as a free-form dict, so
        receivers that don't know about blob refs see them as opaque
        metadata. Do NOT propose a top-level ``blob_ref`` field — that
        would break 1.0 receivers.
        """
        from maxim.utils.format_version import with_format_version

        payload = {
            "timestamp": self.timestamp,
            "source": self.source,
            "detections": self.detections,
            "transcript_chunk": self.transcript_chunk,
            "transcript_chunk_index": self.transcript_chunk_index,
            "file_changed": self.file_changed,
            "cli_input": self.cli_input,
            "has_voice_command": self.has_voice_command,
            "has_maxim_keyword": self.has_maxim_keyword,
            "hard_override": self.hard_override,
            "explore_command": self.explore_command,
            "content": self.content,
            "metadata": self.metadata,
            "raw_transcript_text": self.raw_transcript_text,
            "sensory": self.sensory.to_dict() if self.sensory and hasattr(self.sensory, "to_dict") else None,
            "context": self.context.to_dict() if self.context is not None else None,
            "modality": self.modality,
        }
        return with_format_version(payload)

    @classmethod
    def from_wire_dict(cls, data: dict[str, Any]) -> "Percept":
        """Reconstruct a Percept from a peer-shipped wire-dict.

        Rehydrates the typed
        :class:`~maxim.agents.percept_context.PerceptContext` the same
        way :meth:`from_dict` does. ``sensory`` follows the same
        pre-existing path as :meth:`from_dict` — the bare dict that
        :meth:`to_wire_dict` writes for it deserializes into the
        ``Any``-typed ``Percept.sensory`` field as-is, NOT into a
        typed :class:`~maxim.agents.modality.SensoryTag`. Consumers
        that need the typed form must re-construct it themselves;
        this gap is pre-existing in :meth:`from_dict` and intentional
        for now (wire format is shape-stable; rehydration policy
        follows the session-dict precedent).

        Substrate fields (``embedding``, ``substrate_node_id``) are
        intentionally absent from the wire — leader-side code
        populates them after running substrate encoding on the
        received Percept.

        The ``_format_version`` field is read off the input dict and
        silently dropped before reconstruction — the receiver tolerates
        a missing field (legacy producer) and rejects an unknown
        future version via :func:`check_format_version`.
        """
        from maxim.agents.percept_context import PerceptContext
        from maxim.utils.format_version import check_format_version

        data = dict(data)
        # Surface version drift via the standard one-warning-per-file-type
        # path. Missing field is treated as legacy ("0.x") and tolerated.
        check_format_version(data, "percept_wire")
        data.pop("_format_version", None)
        ctx_raw = data.pop("context", None)
        if ctx_raw is not None and not isinstance(ctx_raw, PerceptContext):
            data["context"] = PerceptContext.from_dict(ctx_raw)
        elif ctx_raw is not None:
            data["context"] = ctx_raw
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes - Memory
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MemoryItem:
    """A single item in salient memory.

    .. deprecated:: 2.0
        Use :class:`WorkingMemoryEntry` wrapping a :class:`MemoryRecord`
        subclass instead. ``MemoryItem`` is retained for backward
        compatibility with ``StructuredContext.relevant_memories`` and
        will be removed after downstream consumers are updated.
    """

    timestamp: float
    content: Any
    salience: float
    source: str  # "percept", "goal_outcome", "user_feedback", "inference"
    decay_rate: float = 0.1  # How fast salience decays

    # Association support
    associations: list[str] = field(default_factory=list)  # Memory IDs
    keywords: set[str] = field(default_factory=set)  # Extracted keywords
    embedding: list[float] | None = None  # Optional semantic embedding

    # Memory tier and lifecycle
    tier: MemoryTier = MemoryTier.SHORT_TERM
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    promoted_at: float | None = None

    # Cached hash (computed lazily for performance)
    _cached_memory_id: str | None = field(default=None, repr=False, compare=False)

    @property
    def memory_id(self) -> str:
        """Stable ID based on content hash (cached for performance)."""
        if self._cached_memory_id is None:
            content_str = str(self.content) if not isinstance(self.content, str) else self.content
            self._cached_memory_id = hashlib.sha256(f"{self.timestamp}:{content_str}".encode()).hexdigest()[:16]
        return self._cached_memory_id

    def access(self) -> None:
        """Mark memory as accessed (refreshes last_accessed)."""
        self.last_accessed = time.time()
        self.access_count += 1

    def should_evict_long_term(self, max_age_seconds: float) -> bool:
        """Check if long-term memory should be evicted."""
        if self.tier != MemoryTier.LONG_TERM:
            return False
        return (time.time() - self.last_accessed) > max_age_seconds


@dataclass
class WorkingMemoryEntry(Generic[T]):
    """Agent-level wrapper around a MemoryRecord for working memory.

    Holds the actual structured record (EpisodicMemory, MathMemory,
    Concept, etc.) plus agent-level metadata for working memory
    management: salience ranking, decay, lifecycle phase.

    The record itself is the canonical type stored in its memory layer
    (Hippocampus, AG, ATL). During FORMING phase, the record's
    decision/action/outcome fields are populated incrementally as the
    pipeline progresses.
    """

    record: T  # The actual MemoryRecord subclass
    salience: float = 0.5  # Agent-level ranking score
    decay_rate: float = 0.1
    source: str = "percept"  # "percept", "goal_outcome", "user_feedback", "inference", "ag_grounding"
    tier: MemoryTier = MemoryTier.SHORT_TERM

    # Pattern completion results (populated during FORMING phase)
    predicted_outcomes: "list[PredictedOutcome] | None" = None
    prediction_confidence: float = 0.0

    # Cached for search performance (extracted lazily from record)
    _keywords: set[str] | None = field(default=None, repr=False)
    _embedding: list[float] | None = field(default=None, repr=False)

    @property
    def id(self) -> str:
        return self.record.id  # type: ignore[union-attr]

    @property
    def timestamp(self) -> float:
        return self.record.timestamp  # type: ignore[union-attr]

    @property
    def is_protected(self) -> bool:
        """FORMING entries cannot be evicted."""
        return self.tier == MemoryTier.FORMING

    def touch(self) -> None:
        """Delegate access tracking to the underlying record. Thread-safe."""
        self.record.touch()  # type: ignore[union-attr]

    @property
    def keywords(self) -> set[str]:
        """Lazily extract keywords from the underlying record."""
        if self._keywords is None:
            self._keywords = self.record.keywords()  # type: ignore[union-attr]
        return self._keywords

    def invalidate_keywords(self) -> None:
        """Clear keyword cache after record mutation (e.g., FORMING → complete)."""
        self._keywords = None

    def __setattr__(self, name: str, value: Any) -> None:
        # Enforce the one-way MemoryTier lifecycle on every tier write.
        # The initial dataclass __init__ assignment is allowed through
        # because __dict__ does not yet contain "tier".
        if name == "tier" and "tier" in self.__dict__:
            _assert_tier_transition(self.__dict__["tier"], value)
        object.__setattr__(self, name, value)

    def should_evict(self, max_age_seconds: float) -> bool:
        """Check if this entry should be evicted from working memory."""
        if self.is_protected:
            return False
        if self.tier == MemoryTier.SHORT_TERM:
            return False  # SHORT_TERM eviction is buffer-based, not age-based
        # LONG_TERM: age-based eviction (consolidation handles active removal)
        age = time.time() - self.record.accessed_at  # type: ignore[union-attr]
        return age > max_age_seconds

    def current_salience(self) -> float:
        """Compute decayed salience based on age. FORMING entries don't decay."""
        if self.is_protected:
            return self.salience
        age = time.time() - self.timestamp
        return self.salience * _math.exp(-self.decay_rate * age / 60.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize wrapper + record for persistence."""

        return {
            "record_type": type(self.record).__name__,
            "record": self.record.to_dict(),  # type: ignore[union-attr]
            "salience": self.salience,
            "decay_rate": self.decay_rate,
            "source": self.source,
            "tier": self.tier.value,
            "predicted_outcomes": ([p.to_dict() for p in self.predicted_outcomes] if self.predicted_outcomes else None),
            "prediction_confidence": self.prediction_confidence,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        record_registry: "dict[str, type] | None" = None,
    ) -> "WorkingMemoryEntry":
        """Deserialize wrapper + record from persistence.

        Args:
            data: Serialized dict from to_dict().
            record_registry: Maps type names to classes for deserialization.
                Defaults to built-in types (EpisodicMemory, MathMemory, etc.).
        """
        from maxim.math.math_types import MathMemory
        from maxim.memory.semantic_types import SemanticMemory
        from maxim.memory.types import (
            CompressedMemory,
            EpisodicMemory,
            PredictedOutcome,
        )

        registry = record_registry or {
            "EpisodicMemory": EpisodicMemory,
            "MathMemory": MathMemory,
            "SemanticMemory": SemanticMemory,
            "CompressedMemory": CompressedMemory,
        }

        record_type_name = data["record_type"]
        record_cls = registry.get(record_type_name)
        if record_cls is None:
            raise ValueError(f"Unknown record type: {record_type_name}")

        record = record_cls.from_dict(data["record"])

        # Persistence migration (0.8): tier="working" → "short" after the
        # WORKING tier was removed.  Old sessions may have persisted episodes
        # at the orphan tier.
        raw_tier = data.get("tier", "short")
        if raw_tier == "working":
            raw_tier = "short"

        return cls(
            record=record,
            salience=data.get("salience", 0.5),
            decay_rate=data.get("decay_rate", 0.1),
            source=data.get("source", "unknown"),
            tier=MemoryTier(raw_tier),
            predicted_outcomes=(
                [PredictedOutcome.from_dict(p) for p in data["predicted_outcomes"]]
                if data.get("predicted_outcomes")
                else None
            ),
            prediction_confidence=data.get("prediction_confidence", 0.0),
        )


# Import for type annotation (avoid circular at module level)
if False:  # TYPE_CHECKING
    from maxim.memory.types import PredictedOutcome  # noqa: F401


@dataclass
class StructuredContext:
    """Context built by MemoryAgent for goal proposal."""

    timestamp: float

    # Current state
    current_percept: Percept | None = None
    active_goal: str | None = None
    active_goal_sub_goals: list[str] = field(default_factory=list)
    mode: str = "observe"  # "observe", "sleep", "shutdown"

    # Autonomy and internet access (Phase 1)
    autonomy_level: str = "planning"  # "planning", "supervised", "autonomous"
    internet_access: bool = False
    internet_policy_summary: str = ""

    # Exploration mode (Phase 2)
    exploration_mode: bool = False
    exploration_focus: str = ""
    exploration_session_id: str = ""
    exploration_policy: dict = field(default_factory=dict)
    exploration_curiosity: float = 1.0
    exploration_budget_remaining: dict = field(default_factory=dict)

    # DEPRECATED (0.8): sourced from WorkingMemorySet via Exec.
    # Kept for backward compatibility during prompt-builder migration.
    # Do NOT add new writers to these fields — use exec.working_memory.add().
    recent_percepts: list[Percept] = field(default_factory=list)
    recent_outcomes: list[dict] = field(default_factory=list)
    # Each entry: {"source": str, "salience": float, "content": dict}
    relevant_memories: list[Any] = field(default_factory=list)

    # Detected patterns (from vision model, NOT raw images)
    detected_objects: list[dict] = field(default_factory=list)
    detected_people: list[dict] = field(default_factory=list)
    detected_speech: list[str] = field(default_factory=list)  # DEPRECATED (0.8): sourced from WorkingMemorySet

    # Abstraction stream data (for LLM context)
    recent_logs: list[dict] = field(default_factory=list)
    goal_history: list[dict] = field(default_factory=list)
    cli_inputs: list[str] = field(default_factory=list)  # DEPRECATED (0.8): sourced from WorkingMemorySet
    # Inbound/outbound comms messages (SMS, voice, etc.)
    # Each entry: {"direction": str, "content": str, "channel": str, "sender": str, "timestamp": float}
    comms_messages: list[dict] = field(default_factory=list)
    available_environments: list[str] = field(default_factory=list)

    # DEPRECATED (0.8): conversation history sourced from WorkingMemorySet.
    # Each entry: {"user": str, "assistant": str, "timestamp": float}
    conversation_history: list[dict] = field(default_factory=list)

    # Statistical context (from StatisticianAgent via bus)
    statistical_context: str = ""
    active_pattern_count: int = 0
    statistical_suggestions: list[dict] = field(default_factory=list)

    # Knowledge context (from ATL semantic memory + AG pattern memory)
    # Each entry: {"concept_name", "definition", "category", "confidence",
    #              "source_layer", "provenance", "relationships": [...]}
    knowledge_context: list[dict] = field(default_factory=list)

    # Concept context (from ConceptContextBuilder — ATL concepts + AG stats)
    # Each entry: {"name", "category", "confidence", "episode_count",
    #              "relationships": [...], "properties": {...}}
    concept_context: list[dict] = field(default_factory=list)

    # Root goal reminder
    root_goal: str = "Understand reality and help people."

    # Working notes (persistent LLM self-context from .maxim_workspace/notes/context.md)
    working_notes: str = ""

    # Workspace file inventory (user-facing artifacts, excludes plan system files)
    workspace_files: list[dict] = field(default_factory=list)

    # Plan progress (from PlanManager when a long-horizon plan is active)
    plan_progress: PlanProgressContext | None = None

    # Causal predictions from NAc (learned outcome expectations)
    # Each entry: {"event": str, "outcome": str, "valence": str,
    #              "confidence": float, "context_match": float}
    causal_context: list[dict] = field(default_factory=list)

    # Valence associations from substrate (SEM learning loop).
    # Each entry: {"concept": str, "valence": float, "associations": list[str]}
    valence_context: list[dict] = field(default_factory=list)

    # Motor programs from Cerebellum (Phase 1b)
    # Each entry: {"name": str, "goal": str, "confidence": float,
    #              "success_rate": float, "steps": list[str], "risks": list[str]}
    motor_programs: list[dict] = field(default_factory=list)

    # Provenance: compact trace markdown for LLM context (P7)
    provenance_context: str = ""

    # Body state from Embodiment (interoception — always present when embodied)
    body_state: str = ""

    # Bio-enrichment context (L1): focused bio-system associations for the
    # current percept — memories, predictions, concepts, affordances.
    # Populated by BioEnrichmentPipeline when a novel percept passes the gate.
    bio_enrichment_context: str = ""

    # Cluster-bias annotations (Wire-A, 0.9.1 Stage 2): top-N (tool, bias)
    # pairs from NAc._cluster_reward_bias aggregated agent-wide via
    # NAc.get_agent_tool_biases. Surfaces substrate-acquired tool-level
    # reward signal to the LLM proposer so it can read priming-acquired
    # bias even on percepts the substrate didn't directly drill (the
    # Roy-2c finding fix). Populated by agent_loop.py at LLM submission;
    # rendered by PromptBuilder._add_cluster_bias_annotation_section.
    # None == disabled (env var off, no NAc wired, or cold-start agent).
    cluster_bias_annotations: list[tuple[str, float]] | None = None

    # Grayscale tool annotations (W1 sense_tool_registry MVP). List of
    # (tool_name, bias, description) for SEM-derived tools the substrate
    # has accumulated a non-zero reward bias for but that are NOT in the
    # active scene roster. Surfaces "knowable but absent" tools so the
    # LLM can reason about reaching for an equivalent active tool. Empty
    # list / None means no grayscale candidates this tick (either no
    # substrate bias or every biased tool is already active). Populated
    # by agent_loop.py at LLM submission; rendered by
    # PromptBuilder._add_grayscale_tools_section. See
    # [docs/plans/deferred/sense_tool_registry.md] § "Phase 3".
    grayscale_tool_annotations: list[tuple[str, float, str]] | None = None

    # Auto-sense: passive perception results (exteroception + interoception).
    # Populated by the agent loop's auto-sense sweep (section 1.15) on each
    # new percept.  Contains sense_presence output (visible entities +
    # affordance annotations) and self-entity sensor readings (health, etc.).
    auto_sense_context: str = ""

    # PFC deliberation: recent THOUGHT entries from WorkingMemorySet.
    # Populated by the deliberation cycle before each LLM call so the
    # LLM sees accumulated reasoning + enrichment from prior cycles.
    working_memory_thoughts: list[str] | None = None

    # Deliberation transcript: accumulating reasoning+enrichment pairs
    # from multi-cycle PFC deliberation.  Each entry pairs the LLM's
    # reasoning with the bio-system response it triggered.  Set by
    # _run_deliberation_cycles; consumed by the prompt builder's
    # _add_deliberation_transcript_section.  None for one-shot (cycle 1
    # only) deliberation — the transcript is only built when cycles 2+
    # actually run.  When present, suppresses the separate bio_enrichment
    # section to avoid rendering the current cycle's enrichment twice.
    deliberation_transcript: list[str] | None = None


@dataclass
class PlanProgressContext:
    """Injected into StructuredContext when a long-horizon plan is active."""

    plan_id: str
    objective: str
    status: str
    current_phase_index: int
    total_phases: int
    current_phase_description: str = ""
    phases_completed: int = 0
    phases_failed: int = 0
    energy_utilization: dict[str, float] = field(default_factory=dict)
    is_replanning: bool = False
    replan_count: int = 0

    def to_prompt_section(self) -> str:
        """Format as a structured prompt section for the LLM."""
        lines = [
            f"## Active Plan: {self.objective}",
            f"Status: {self.status}",
            f"Phase: {self.current_phase_index + 1}/{self.total_phases} — {self.current_phase_description}",
            f"Completed: {self.phases_completed}, Failed: {self.phases_failed}",
        ]
        if self.is_replanning:
            lines.append(f"REPLANNING (attempt {self.replan_count})")
        if self.energy_utilization:
            parts = [f"{d}: {u:.0%}" for d, u in self.energy_utilization.items()]
            lines.append(f"Energy: {', '.join(parts)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes - Goals
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SubGoal:
    """A concrete sub-goal with execution metadata."""

    id: str
    description: str
    tool_name: str
    tool_params: dict[str, Any] = field(default_factory=dict)

    # Execution state
    status: SubGoalStatus = SubGoalStatus.PENDING
    result: Any = None
    error: str | None = None
    attempts: int = 0
    max_retries: int = 2

    # Priority (None = inherit from parent)
    priority_override: GoalPriority | None = None

    # Failure handling
    on_failure: FailureStrategy = FailureStrategy.RETRY

    # Dependencies: IDs of sub-goals that must complete first
    depends_on: list[str] = field(default_factory=list)

    # Metadata
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    def effective_priority(self, parent_priority: GoalPriority) -> GoalPriority:
        """Get effective priority (override or inherited)."""
        return self.priority_override or parent_priority

    def can_execute(self, completed_ids: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_ids for dep in self.depends_on)

    def should_retry(self) -> bool:
        """Check if sub-goal should be retried."""
        if self.status != SubGoalStatus.FAILED:
            return False
        if self.on_failure not in (FailureStrategy.RETRY, FailureStrategy.ESCALATE):
            return False
        return self.attempts < self.max_retries

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "status": self.status.name,
            "result": self.result,
            "error": self.error,
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "priority_override": self.priority_override.name if self.priority_override else None,
            "on_failure": self.on_failure.name,
            "depends_on": self.depends_on,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SubGoal:
        pri = d.get("priority_override")
        return cls(
            id=d["id"],
            description=d["description"],
            tool_name=d["tool_name"],
            tool_params=d.get("tool_params", {}),
            status=SubGoalStatus[d.get("status", "PENDING")],
            result=d.get("result"),
            error=d.get("error"),
            attempts=d.get("attempts", 0),
            max_retries=d.get("max_retries", 2),
            priority_override=GoalPriority[pri] if pri else None,
            on_failure=FailureStrategy[d.get("on_failure", "RETRY")],
            depends_on=d.get("depends_on", []),
            created_at=d.get("created_at", 0.0),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
        )


@dataclass
class ProposedGoal:
    """A goal proposed by ExecAgent."""

    id: str
    description: str
    priority: GoalPriority
    tool_name: str | None = None
    tool_params: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 1.0
    parent_goal: str | None = None  # Links to root or parent goal
    expires_at: float | None = None  # Optional timeout
    created_at: float = field(default_factory=time.time)

    # Sub-goal management
    sub_goals: list[SubGoal] = field(default_factory=list)

    def get_next_executable(self) -> SubGoal | None:
        """Get the next sub-goal that can be executed."""
        completed_ids = {
            sg.id for sg in self.sub_goals if sg.status in (SubGoalStatus.COMPLETED, SubGoalStatus.SKIPPED)
        }

        # Find all executable sub-goals
        executable = [
            sg
            for sg in self.sub_goals
            if sg.status in (SubGoalStatus.PENDING, SubGoalStatus.BLOCKED) and sg.can_execute(completed_ids)
        ]

        if not executable:
            # Check for retryable failures
            retryable = [sg for sg in self.sub_goals if sg.should_retry()]
            if retryable:
                retryable.sort(key=lambda sg: sg.effective_priority(self.priority).value)
                return retryable[0]
            return None

        # Sort by effective priority (lowest value = highest priority)
        executable.sort(key=lambda sg: sg.effective_priority(self.priority).value)
        return executable[0]

    def handle_sub_goal_failure(self, sub_goal: SubGoal, error: str) -> bool:
        """Handle sub-goal failure. Returns True if parent goal should continue."""
        sub_goal.status = SubGoalStatus.FAILED
        sub_goal.error = error
        sub_goal.attempts += 1

        if sub_goal.on_failure == FailureStrategy.SKIP:
            sub_goal.status = SubGoalStatus.SKIPPED
            return True

        if sub_goal.on_failure == FailureStrategy.RETRY and sub_goal.should_retry():
            sub_goal.status = SubGoalStatus.PENDING
            return True

        if sub_goal.on_failure == FailureStrategy.ESCALATE:
            current = sub_goal.effective_priority(self.priority)
            if current.value > GoalPriority.CRITICAL.value:
                sub_goal.priority_override = GoalPriority(current.value - 1)
                sub_goal.status = SubGoalStatus.PENDING
                return True

        return False

    def is_complete(self) -> bool:
        """Check if all sub-goals are complete (or skipped)."""
        if not self.sub_goals:
            return True
        return all(sg.status in (SubGoalStatus.COMPLETED, SubGoalStatus.SKIPPED) for sg in self.sub_goals)

    def all_failed(self) -> bool:
        """Check if goal failed due to sub-goal failures."""
        return any(sg.status == SubGoalStatus.FAILED and not sg.should_retry() for sg in self.sub_goals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority.name,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "parent_goal": self.parent_goal,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
            "sub_goals": [sg.to_dict() for sg in self.sub_goals],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProposedGoal:
        return cls(
            id=d["id"],
            description=d["description"],
            priority=GoalPriority[d["priority"]],
            tool_name=d.get("tool_name"),
            tool_params=d.get("tool_params", {}),
            reasoning=d.get("reasoning", ""),
            confidence=d.get("confidence", 1.0),
            parent_goal=d.get("parent_goal"),
            expires_at=d.get("expires_at"),
            created_at=d.get("created_at", 0.0),
            sub_goals=[SubGoal.from_dict(sg) for sg in d.get("sub_goals", [])],
        )


@dataclass
class GoalAccepted:
    """Confirmation that GoalAgent accepted a proposed goal."""

    goal_id: str
    timestamp: float


@dataclass
class GoalCompleted:
    """Notification that a goal was completed."""

    goal_id: str
    success: bool
    result: Any = None
    error: str | None = None


@dataclass
class ToolCall:
    """Tool invocation from GoalAgent."""

    id: str
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)
    goal_id: str | None = None


@dataclass
class ToolResult:
    """Result of tool execution, published on the bus.

    Constructed by the agent loop from the raw ToolOutput (tools.base)
    plus action metadata (tool_call_id, tool_name, params).
    """

    tool_call_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolResult:
        return cls(
            tool_call_id=d["tool_call_id"],
            tool_name=d["tool_name"],
            success=d["success"],
            result=d.get("result"),
            error=d.get("error"),
            params=d.get("params", {}),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes - Plan Events
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PlanCreated:
    """Published when a new long-horizon plan is created."""

    plan_id: str
    objective: str
    phase_count: int
    timestamp: float


@dataclass
class PhaseStarted:
    """Published when a plan phase becomes active."""

    plan_id: str
    phase_id: str
    phase_index: int
    description: str
    timestamp: float


@dataclass
class PhaseCompleted:
    """Published when a plan phase completes (success or failure)."""

    plan_id: str
    phase_id: str
    success: bool
    result_summary: str | None = None
    error: str | None = None
    energy_spent: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class PlanCompleted:
    """Published when an entire plan completes."""

    plan_id: str
    success: bool
    phases_completed: int
    phases_total: int
    total_energy_spent: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class PlanRestored:
    """Published when a plan is restored from disk on session start."""

    plan_id: str
    objective: str
    current_phase_index: int
    status: str
    timestamp: float


@dataclass
class PlanReplanRequested:
    """Published when a phase failure triggers re-decomposition."""

    plan_id: str
    failed_phase_id: str
    reason: str
    replan_context: Any = None  # ReplanContext (Any to avoid circular import)
    timestamp: float = field(default_factory=time.time)


@dataclass
class LoopTerminated:
    """Published when agentic loop stops."""

    run_id: str
    stop_reason: StopReason
    steps_taken: int
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class StreamEvent:
    """Fine-grained event for real-time streaming."""

    kind: str  # "inference_start", "inference_end", "tool_start", "tool_end", "decision", "error"
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes - Statistical
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StatisticalInsight:
    """Published by StatisticianAgent when a pattern is confirmed."""

    timestamp: float
    insight_type: str  # "pattern", "anomaly", "temporal", "correlation"
    metric: str  # "tool:navigate:success", "goal:search:latency"
    description: str  # Actionable description
    severity: float  # 0.0-1.0
    pattern_type: str  # "trending", "cyclic", "clustering", "random"
    temporal_context: str  # "normal for this time" or "unusual for Tuesday AM"
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisSuggestion:
    """A specific, actionable analysis recommendation from StatisticianAgent."""

    metric: str  # "tool:navigate:success"
    tool_call: str  # "math"
    operation: str  # "assess_randomness", "analyze", "recall_memory"
    rationale: str  # Human-readable explanation of why this analysis is suggested
    priority: float  # 0.0-1.0, higher = more urgent
    data_type: str  # "binary", "continuous", "rate", "latency"
    fsm_state: str  # "PATTERN_FORMING", "CONFIRMED_PATTERN", etc.


@dataclass
class StatisticalSummary:
    """Periodic summary from StatisticianAgent for context building."""

    timestamp: float
    summary: str  # Actionable natural language (for StructuredContext)
    active_patterns: int  # Count of CONFIRMED_PATTERN metrics
    metrics_monitored: int  # Total metrics being tracked
    suggestions: list[AnalysisSuggestion] = field(default_factory=list)
    data_type_breakdown: dict[str, int] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Dependency Graph
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Edge:
    """A directed edge in the dependency graph."""

    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class DependencyGraph(Generic[T]):
    """
    Generic dependency graph with cycle detection and topological ordering.

    Used for:
    - Sub-goal dependency tracking (REQUIRES/ENABLES edges)
    - Memory association networks (ASSOCIATES/CAUSES edges)
    """

    def __init__(self) -> None:
        self._nodes: dict[str, T] = {}
        self._outgoing: dict[str, list[Edge]] = defaultdict(list)
        self._incoming: dict[str, list[Edge]] = defaultdict(list)
        self._lock = threading.Lock()

    def add_node(self, node_id: str, data: T) -> None:
        """Add a node to the graph."""
        with self._lock:
            self._nodes[node_id] = data

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        with self._lock:
            if node_id not in self._nodes:
                return

            for edge in self._outgoing[node_id]:
                self._incoming[edge.target] = [e for e in self._incoming[edge.target] if e.source != node_id]
            del self._outgoing[node_id]

            for edge in self._incoming[node_id]:
                self._outgoing[edge.source] = [e for e in self._outgoing[edge.source] if e.target != node_id]
            del self._incoming[node_id]

            del self._nodes[node_id]

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add a directed edge. Returns False if it would create a cycle."""
        edge = Edge(source, target, edge_type, weight, metadata or {})

        with self._lock:
            # For dependency edges, check for cycles
            if edge_type in (EdgeType.REQUIRES, EdgeType.ENABLES):
                self._outgoing[source].append(edge)
                self._incoming[target].append(edge)

                if self._has_cycle_from_unlocked(source):
                    self._outgoing[source].remove(edge)
                    self._incoming[target].remove(edge)
                    return False
            else:
                self._outgoing[source].append(edge)
                self._incoming[target].append(edge)

        return True

    def update_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing edge's weight and/or metadata in-place.

        Finds the first edge matching (source, target, edge_type) and applies
        the requested updates. Returns True if the edge was found and updated.
        """
        with self._lock:
            for edge in self._outgoing.get(source, []):
                if edge.target == target and edge.edge_type == edge_type:
                    if weight is not None:
                        edge.weight = weight
                    if metadata_updates:
                        edge.metadata.update(metadata_updates)
                    return True
        return False

    def decay_edges(
        self,
        factor: float,
        *,
        edge_types: set[EdgeType] | None = None,
        floor: float = 0.01,
        prune: bool = True,
    ) -> int:
        """Decay all matching edge weights by a multiplicative factor.

        P6 extinction mechanism: edges that are not reinforced decay
        toward zero over time. Edges below ``floor`` are pruned if
        ``prune=True``.

        Args:
            factor: Multiplicative decay (0 < factor < 1). Applied as
                    ``edge.weight *= factor``.
            edge_types: Only decay edges of these types. Default:
                        {ASSOCIATES} (Hebbian binding edges).
            floor: Minimum weight. Edges at or below this are pruned.
            prune: If True, remove edges below the floor.

        Returns:
            Number of edges pruned.
        """
        if not (0.0 < factor < 1.0):
            raise ValueError(f"decay factor must be in (0, 1), got {factor}")
        if edge_types is None:
            edge_types = {EdgeType.ASSOCIATES}

        pruned = 0
        with self._lock:
            for source_id in list(self._outgoing.keys()):
                surviving: list[Edge] = []
                for edge in self._outgoing[source_id]:
                    if edge.edge_type not in edge_types:
                        surviving.append(edge)
                        continue
                    edge.weight *= factor
                    if prune and edge.weight <= floor:
                        # Remove from incoming list too (must match source+target+type)
                        self._incoming[edge.target] = [
                            e
                            for e in self._incoming[edge.target]
                            if not (
                                e.source == edge.source and e.target == edge.target and e.edge_type == edge.edge_type
                            )
                        ]
                        pruned += 1
                    else:
                        surviving.append(edge)
                self._outgoing[source_id] = surviving
        return pruned

    def find_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType | None = None,
        metadata_match: dict[str, Any] | None = None,
    ) -> Edge | None:
        """Find an edge matching the given criteria.

        Args:
            source: Source node ID.
            target: Target node ID.
            edge_type: Filter by edge type (None = any).
            metadata_match: Filter by metadata key-value pairs (None = any).

        Returns the first matching Edge, or None.
        """
        with self._lock:
            for edge in self._outgoing.get(source, []):
                if edge.target != target:
                    continue
                if edge_type is not None and edge.edge_type != edge_type:
                    continue
                if metadata_match:
                    if not all(edge.metadata.get(k) == v for k, v in metadata_match.items()):
                        continue
                return edge
        return None

    def add_bidirectional(
        self,
        node_a: str,
        node_b: str,
        edge_type: EdgeType = EdgeType.ASSOCIATES,
        weight: float = 1.0,
    ) -> None:
        """Add bidirectional association."""
        self.add_edge(node_a, node_b, edge_type, weight)
        self.add_edge(node_b, node_a, edge_type, weight)

    def get_node(self, node_id: str) -> T | None:
        """Get node data by ID."""
        with self._lock:
            return self._nodes.get(node_id)

    def get_dependencies(
        self,
        node_id: str,
        edge_types: set[EdgeType] | None = None,
    ) -> list[str]:
        """Get IDs of nodes this node depends on."""
        with self._lock:
            edges = self._incoming.get(node_id, [])
            if edge_types:
                edges = [e for e in edges if e.edge_type in edge_types]
            return [e.source for e in edges if e.edge_type == EdgeType.REQUIRES]

    def get_dependents(self, node_id: str) -> list[str]:
        """Get IDs of nodes that depend on this node."""
        with self._lock:
            edges = self._outgoing.get(node_id, [])
            return [e.target for e in edges if e.edge_type == EdgeType.REQUIRES]

    def get_associated(
        self,
        node_id: str,
        edge_types: set[EdgeType] | None = None,
    ) -> list[tuple[str, float]]:
        """Get associated nodes with weights."""
        edge_types = edge_types or {EdgeType.ASSOCIATES, EdgeType.CAUSES}
        with self._lock:
            edges = self._outgoing.get(node_id, [])
            return [(e.target, e.weight) for e in edges if e.edge_type in edge_types]

    def _has_cycle_from_unlocked(self, start: str) -> bool:
        """Check for cycles (must hold lock)."""
        visited: set[str] = set()
        path: set[str] = set()

        def dfs(node: str) -> bool:
            if node in path:
                return True
            if node in visited:
                return False

            visited.add(node)
            path.add(node)

            for edge in self._outgoing.get(node, []):
                if edge.edge_type in (EdgeType.REQUIRES, EdgeType.ENABLES):
                    if dfs(edge.target):
                        return True

            path.remove(node)
            return False

        return dfs(start)

    def find_all_cycles(self) -> list[list[str]]:
        """Find all strongly connected components using Tarjan's algorithm."""
        with self._lock:
            index_counter = [0]
            stack: list[str] = []
            lowlinks: dict[str, int] = {}
            index: dict[str, int] = {}
            on_stack: set[str] = set()
            sccs: list[list[str]] = []

            def strongconnect(node: str) -> None:
                index[node] = index_counter[0]
                lowlinks[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack.add(node)

                for edge in self._outgoing.get(node, []):
                    if edge.edge_type not in (EdgeType.REQUIRES, EdgeType.ENABLES):
                        continue
                    successor = edge.target
                    if successor not in index:
                        strongconnect(successor)
                        lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                    elif successor in on_stack:
                        lowlinks[node] = min(lowlinks[node], index[successor])

                if lowlinks[node] == index[node]:
                    scc: list[str] = []
                    while True:
                        w = stack.pop()
                        on_stack.remove(w)
                        scc.append(w)
                        if w == node:
                            break
                    if len(scc) > 1:
                        sccs.append(scc)

            for node in self._nodes:
                if node not in index:
                    strongconnect(node)

            return sccs

    def validate(self) -> tuple[bool, list[list[str]]]:
        """Validate graph has no dependency cycles."""
        cycles = self.find_all_cycles()
        return len(cycles) == 0, cycles

    def topological_sort(self) -> list[str] | None:
        """Return nodes in topological order. Returns None if cycles exist."""
        with self._lock:
            in_degree: dict[str, int] = {node: 0 for node in self._nodes}

            for node in self._nodes:
                for edge in self._outgoing.get(node, []):
                    if edge.edge_type in (EdgeType.REQUIRES, EdgeType.ENABLES):
                        if edge.target in in_degree:
                            in_degree[edge.target] += 1

            queue = deque([node for node, degree in in_degree.items() if degree == 0])
            result: list[str] = []

            while queue:
                node = queue.popleft()
                result.append(node)

                for edge in self._outgoing.get(node, []):
                    if edge.edge_type in (EdgeType.REQUIRES, EdgeType.ENABLES):
                        target = edge.target
                        if target in in_degree:
                            in_degree[target] -= 1
                            if in_degree[target] == 0:
                                queue.append(target)

            if len(result) != len(self._nodes):
                return None

            return result

    @overload
    def spreading_activation(
        self,
        source_ids: list[str],
        initial_activation: float = 1.0,
        decay: float = 0.5,
        threshold: float = 0.1,
        max_depth: int = 3,
        node_filter: Callable[[str], bool] | None = None,
        propagate_valence: Literal[False] = False,
    ) -> dict[str, float]: ...

    @overload
    def spreading_activation(
        self,
        source_ids: list[str],
        initial_activation: float = 1.0,
        decay: float = 0.5,
        threshold: float = 0.1,
        max_depth: int = 3,
        node_filter: Callable[[str], bool] | None = None,
        propagate_valence: Literal[True] = ...,
    ) -> dict[str, tuple[float, float]]: ...

    def spreading_activation(
        self,
        source_ids: list[str],
        initial_activation: float = 1.0,
        decay: float = 0.5,
        threshold: float = 0.1,
        max_depth: int = 3,
        node_filter: Callable[[str], bool] | None = None,
        propagate_valence: bool = False,
    ) -> dict[str, float] | dict[str, tuple[float, float]]:
        """Spread activation from source nodes through association edges.

        ``node_filter`` is an optional callable ``(node_id) -> bool``
        applied to every source AND every target visited during
        traversal. Nodes for which the filter returns ``False`` are
        skipped — neither added to the result nor used as a hop. This
        is the seam P3b uses for channel-filtered retrieval and P4
        uses for modality-filtered cross-modal walks (substrate P3a
        Stage 2). Pass ``None`` (default) to disable filtering.

        When ``propagate_valence=True``, returns
        ``dict[str, tuple[float, float]]`` where each value is
        ``(activation, valence)``.  Valence propagates alongside
        activation: ``parent_valence * decay + edge.metadata["valence"]``.
        """
        if propagate_valence:
            return self._spreading_activation_with_valence(
                source_ids, initial_activation, decay, threshold, max_depth, node_filter
            )

        activations: dict[str, float] = {}
        visited_at_depth: dict[str, int] = {}

        queue: deque[tuple[str, float, int]] = deque()
        with self._lock:
            for source in source_ids:
                if source in self._nodes:
                    if node_filter is not None and not node_filter(source):
                        continue
                    queue.append((source, initial_activation, 0))
                    activations[source] = initial_activation
                    visited_at_depth[source] = 0

            while queue:
                node_id, activation, depth = queue.popleft()

                if depth >= max_depth:
                    continue

                for target, weight in [
                    (e.target, e.weight)
                    for e in self._outgoing.get(node_id, [])
                    if e.edge_type in (EdgeType.ASSOCIATES, EdgeType.CAUSES)
                ]:
                    if node_filter is not None and not node_filter(target):
                        continue

                    new_activation = activation * decay * weight

                    if new_activation < threshold:
                        continue

                    if target not in activations or new_activation > activations[target]:
                        activations[target] = new_activation
                        visited_at_depth[target] = depth + 1
                        queue.append((target, new_activation, depth + 1))

        return activations

    def _spreading_activation_with_valence(
        self,
        source_ids: list[str],
        initial_activation: float,
        decay: float,
        threshold: float,
        max_depth: int,
        node_filter: Callable[[str], bool] | None,
    ) -> dict[str, tuple[float, float]]:
        """Internal helper: spreading activation with valence tracking."""
        activations: dict[str, tuple[float, float]] = {}

        # (node_id, activation, valence, depth)
        q: deque[tuple[str, float, float, int]] = deque()
        with self._lock:
            for source in source_ids:
                if source in self._nodes:
                    if node_filter is not None and not node_filter(source):
                        continue
                    q.append((source, initial_activation, 0.0, 0))
                    activations[source] = (initial_activation, 0.0)

            while q:
                node_id, activation, valence, depth = q.popleft()

                if depth >= max_depth:
                    continue

                for edge in self._outgoing.get(node_id, []):
                    if edge.edge_type not in (EdgeType.ASSOCIATES, EdgeType.CAUSES):
                        continue
                    target = edge.target
                    if node_filter is not None and not node_filter(target):
                        continue

                    new_activation = activation * decay * edge.weight
                    if new_activation < threshold:
                        continue

                    edge_valence = edge.metadata.get("valence", 0.0)
                    new_valence = valence * decay + edge_valence

                    prev = activations.get(target)
                    if prev is None or new_activation > prev[0]:
                        activations[target] = (new_activation, new_valence)
                        q.append((target, new_activation, new_valence, depth + 1))

        return activations

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph for persistence."""
        with self._lock:
            edges = []
            for source, edge_list in self._outgoing.items():
                for edge in edge_list:
                    edges.append(
                        {
                            "source": edge.source,
                            "target": edge.target,
                            "type": edge.edge_type.name,
                            "weight": edge.weight,
                            "metadata": edge.metadata,
                        }
                    )

            return {
                "nodes": list(self._nodes.keys()),
                "edges": edges,
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DependencyGraph":
        """Deserialize graph from persistence."""
        graph: DependencyGraph = cls()
        for node_id in data.get("nodes", []):
            graph.add_node(node_id, node_id)
        for edge_data in data.get("edges", []):
            try:
                edge_type = EdgeType[edge_data["type"]]
            except KeyError:
                edge_type = EdgeType.ASSOCIATES
            graph.add_edge(
                source=edge_data["source"],
                target=edge_data["target"],
                edge_type=edge_type,
                weight=edge_data.get("weight", 1.0),
                metadata=edge_data.get("metadata", {}),
            )
        return graph


# ─────────────────────────────────────────────────────────────────────────────
# Agent Bus
# ─────────────────────────────────────────────────────────────────────────────


class AgentBus:
    """Thread-safe publish/subscribe message bus for agent communication."""

    def __init__(self) -> None:
        self._subscribers: dict[type, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, msg_type: type[T], handler: Callable[[T], None]) -> None:
        """Subscribe to messages of a specific type."""
        with self._lock:
            self._subscribers[msg_type].append(handler)

    def unsubscribe(self, msg_type: type[T], handler: Callable[[T], None]) -> None:
        """Unsubscribe from messages."""
        with self._lock:
            if handler in self._subscribers[msg_type]:
                self._subscribers[msg_type].remove(handler)

    def publish(self, message: Any) -> None:
        """Publish a message to all subscribers."""
        with self._lock:
            handlers = list(self._subscribers[type(message)])

        for handler in handlers:
            try:
                _t0 = time.monotonic()
                handler(message)
                _elapsed = time.monotonic() - _t0
                if _elapsed > 0.05:  # 50ms
                    logger.warning(
                        "Slow bus handler: %s took %.1fms for %s",
                        getattr(handler, "__qualname__", repr(handler)),
                        _elapsed * 1000,
                        type(message).__name__,
                    )
            except Exception as e:
                logger.debug(
                    "Bus handler %s raised: %s",
                    getattr(handler, "__qualname__", repr(handler)),
                    e,
                )

    def clear(self) -> None:
        """Clear all subscriptions."""
        with self._lock:
            self._subscribers.clear()
