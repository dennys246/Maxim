"""Hippocampus - Associative memory graph for agentic loops.

Phase 1 Implementation:
- Core capture and retrieval
- Hash index for O(1) lookups
- JSON persistence
- RWLock for concurrent access
- StateStore for state snapshot caching

See hippocampus_plan.md for full design and future phases.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal
from uuid import uuid4

if TYPE_CHECKING:
    from maxim.memory.strategies import MemoryStrategy
    from maxim.time.scn import SCN

from maxim.agents.bus import DependencyGraph, EdgeType
from maxim.memory.episode import (
    BoundaryRule,
    CaptureEvent,
    Episode,
    EpisodeBoundaryDetector,
    EpisodeStore,
    PendingEpisodeState,
    apply_hebbian_on_close,
    channel_change_rule,
    scn_tag_change_rule,
    tick_gap_rule,
)
from maxim.memory.hippocampus_consolidation import ConsolidationMixin
from maxim.memory.hippocampus_persistence import PersistenceMixin
from maxim.memory.hippocampus_retrieval import RetrievalMixin
from maxim.memory.layer import MemoryLayer
from maxim.memory.rwlock import RWLock
from maxim.memory.state_store import StateStore
from maxim.memory.types import (
    Action,
    CompressedMemory,
    Context,
    Decision,
    EpisodicMemory,
    Outcome,
    Perception,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HebbianConfig:
    """P3a Stage 1 — Hebbian edge update knobs for ``apply_hebbian_on_close``.

    Separated from retrieval tuning and boundary detection so each
    concern has a focused home. Stage 2 Arch-lens finding #4: the
    original flat ``EpisodeConfig`` was absorbing edge-update,
    retrieval, and boundary knobs on one dataclass, heading toward
    kitchen-sink as P3b (channel) / P4 (cross-modal) / P6 (extinction)
    add their own tuning.
    """

    # Initial edge weight for a fresh Hebbian edge between two nodes
    # co-occurring in an episode for the first time.
    init: float = 0.3

    # Per-close weight increment for an existing Hebbian edge whose
    # endpoints co-occur again in a newly-closed episode.
    delta: float = 0.1

    # Upper clamp on Hebbian edge weight — no single episode sequence
    # can drive a weight above this. Named ``max_weight`` rather than
    # ``max`` to avoid shadowing the builtin.
    max_weight: float = 1.0


@dataclass(frozen=True)
class RetrievalConfig:
    """P3a Stage 2 — multi-hop retrieval knobs for
    ``Hippocampus.retrieve_on_cue(..., multi_hop=True)``.

    These drive the ``DependencyGraph.spreading_activation`` traversal
    on the binding graph. One-hop retrieval ignores them entirely.

    **Default calibration** (hub+chain Stage 2 fixture):
    ``decay=0.7 × max_core_weight=0.4 = 0.28`` per hop. With
    ``threshold=0.001``, effective reach is
    ``log(0.001 / 1.0) / log(0.28) ≈ 5.4`` hops, bounded by
    ``max_depth=5``. If a future fixture has different weight
    stratification (e.g., P3b real-text episodes averaging ~0.2 edge
    weight), the threshold will prune earlier and multi-hop will not
    reach deep targets. Re-tune when ``retrieve_on_cue`` recall drops
    on real-text fixtures.
    """

    # Per-hop activation decay for ``spreading_activation``. Values in
    # ``[0, 1]``; higher = activation travels farther.
    decay: float = 0.7

    # Minimum activation to keep a node in the retrieved set. Nodes
    # whose multi-hop weight falls below this are pruned. Low values
    # reach deeper chains; very low values risk spreading to
    # cross-topic noise.
    threshold: float = 0.001

    # Maximum hops from the cue. Hard cap on traversal depth.
    max_depth: int = 5


@dataclass(frozen=True)
class EpisodeConfig:
    """P3a Stage 1 + Stage 2 — configuration knobs for episode binding.

    Composes ``HebbianConfig`` (edge-update) and ``RetrievalConfig``
    (edge-traversal) plus the boundary detector knob. Each concern is
    in its own dataclass so P3b/P4/P6 can extend without ballooning
    a single flat config.

    Override patterns:

    - ``HippocampusConfig(episode=EpisodeConfig(hebbian=HebbianConfig(delta=0.2)))``
    - ``HippocampusConfig(episode=EpisodeConfig(retrieval=RetrievalConfig(decay=0.8)))``
    - ``HippocampusConfig(episode=EpisodeConfig(boundary_tick_gap=100))``

    Defined ABOVE ``HippocampusConfig`` so the ``episode: EpisodeConfig``
    field's type annotation resolves eagerly under ``get_type_hints()``
    and dict-shaped YAML construction — P3.5 Round 2 Arch critical #1.
    """

    # Episode boundary detector: close the pending episode when the
    # next capture is more than this many ticks after the previous.
    boundary_tick_gap: int = 50

    # Nested edge-update knobs (Stage 1 Hebbian mechanism).
    hebbian: HebbianConfig = field(default_factory=HebbianConfig)

    # Nested multi-hop retrieval knobs (Stage 2).
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


@dataclass(frozen=True)
class HippocampusConfig:
    """Configuration for the Hippocampus."""

    # Maximum nodes before warning (not enforced in Phase 1)
    max_nodes: int = 10_000

    # State store capacity
    state_store_max_entries: int = 1000

    # Persistence path (None = no auto-persistence)
    persistence_path: str | None = None

    # Index keys that should be tracked
    indexed_keys: frozenset[str] = field(
        default_factory=lambda: frozenset({"goal", "tool", "object", "person", "success", "mode"})
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Sleep Consolidation Settings
    # ─────────────────────────────────────────────────────────────────────────

    # Enable sleep consolidation
    enable_sleep_consolidation: bool = True

    # Default time without access before memory is candidate for removal (1 week)
    max_age_without_access: float = 7 * 24 * 3600  # 604800 seconds

    # Time after which full memories are compressed to CompressedMemory (1 day)
    compression_age: float = 24 * 3600  # 86400 seconds

    # Retention threshold: memories scoring below this are removed
    retention_threshold: float = 0.3

    # Compression threshold: memories scoring below this get compressed
    compression_threshold: float = 0.6

    # Memory strategy to use: "access_based", "importance_based", "composite"
    memory_strategy: str = "access_based"

    # ─────────────────────────────────────────────────────────────────────────
    # Long-Term Memory Consolidation
    # ─────────────────────────────────────────────────────────────────────────

    # Run consolidation during sleep()
    consolidate_during_sleep: bool = True

    # Maximum candidates to track for consolidation (bounded queue)
    max_consolidation_candidates: int = 500

    # Immediate promotion thresholds (at capture time)
    immediate_promotion_salience: float = 0.95
    immediate_promotion_novelty: float = 0.95

    # Retention score boost for long-term memories (multiplicative)
    long_term_retention_boost: float = 2.0

    # Auto-save after sleep consolidation
    auto_save_after_sleep: bool = True

    # ─────────────────────────────────────────────────────────────────────────
    # Associative Graph Settings
    # ─────────────────────────────────────────────────────────────────────────

    # Enable associative edge formation during capture
    enable_associative_graph: bool = True

    # Max similar memories to link per capture
    association_limit: int = 5

    # Minimum similarity score to form an edge (0-1)
    association_threshold: float = 0.5

    # Observation dedup window — reject near-identical observations within this
    # time window to reduce memory spam from repetitive percepts.
    dedup_window_s: float = 30.0

    # Spreading activation defaults
    spreading_activation_decay: float = 0.5
    spreading_activation_max_depth: int = 3
    spreading_activation_threshold: float = 0.05

    # ─────────────────────────────────────────────────────────────────────────
    # Async Capture Settings
    # ─────────────────────────────────────────────────────────────────────────

    # Maximum pending captures before drop-oldest
    capture_queue_size: int = 100

    # ─────────────────────────────────────────────────────────────────────────
    # P3a Stage 1 — Episode binding settings
    # Held as a nested EpisodeConfig block so tests + Stage 2 sweeps can
    # override these independently of the rest of HippocampusConfig.
    # ─────────────────────────────────────────────────────────────────────────

    episode: EpisodeConfig = field(default_factory=lambda: EpisodeConfig())


class _SnapshotState:
    """Minimal state-like wrapper around a pre-resolved dict snapshot.

    capture_from_loop() accesses state attributes (e.g. state.data,
    state.processing_state). This wrapper presents a dict snapshot as
    if it were a real state object so capture_from_loop() works unchanged.
    """

    def __init__(self, data: dict[str, Any] | None) -> None:
        self.data = data or {}
        self.processing_state = self.data.get("processing_state")
        self.operational_mode = self.data.get("operational_mode")
        self.strategy = self.data.get("strategy")

    def snapshot(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass
class _CaptureRequest:
    """Immutable snapshot of capture data, queued for background processing."""

    observation: dict[str, Any]
    state_snapshot: dict[str, Any] | None
    intent: dict[str, Any]
    decision: dict[str, Any]
    action: dict[str, Any]
    result: Any
    run_id: str
    queued_at: float


class Hippocampus(PersistenceMixin, ConsolidationMixin, RetrievalMixin, MemoryLayer):
    """Hybrid hash-table graph for associative memory of agentic loops.

    Each node stores a complete EpisodicMemory (observe -> decide -> act -> outcome).
    Combines O(1) context lookup with graph-based relational queries.
    Thread-safe for concurrent access via RWLock.

    Phase 1 Features:
    - capture(): Store a complete agentic loop
    - get(): Retrieve by memory_id
    - recall(): Query with filters
    - recall_similar(): Find memories similar to current perception
    - save()/load(): JSON persistence

    Example:
        hippocampus = Hippocampus()

        # Capture a memory
        memory_id = hippocampus.capture(
            perception=Perception(detected_objects=["cup"]),
            context=Context(active_goal="find cup"),
            decision=Decision(intent={"goal": "pick up cup"}),
            action=Action(tool_name="grasp"),
            outcome=Outcome(success=True),
        )

        # Query memories
        memories = hippocampus.recall(goal="find cup", success=True)
    """

    def __init__(self, config: HippocampusConfig | None = None) -> None:
        self.config = config or HippocampusConfig()

        # Primary storage: memory_id -> EpisodicMemory or CompressedMemory
        self._memories: dict[str, EpisodicMemory | CompressedMemory] = {}

        # Hash index: context_key -> set of memory_ids
        # e.g., "goal:find cup" -> {"mem_123", "mem_456"}
        self._context_index: dict[str, set[str]] = defaultdict(set)

        # Reverse index for cleanup: memory_id -> set of context_keys
        self._memory_contexts: dict[str, set[str]] = defaultdict(set)

        # State store for full snapshots (EpisodicMemory only stores hash reference)
        self._state_store = StateStore(max_entries=self.config.state_store_max_entries)

        # Thread safety - RWLock for concurrent reads, exclusive writes
        self._rwlock = RWLock()

        # Statistics
        self._stats = {
            "memories_captured": 0,
            "queries": 0,
        }

        # Deletion callbacks for subsystem cleanup
        self._on_memory_deleted: list[Callable[[str], None]] = []

        # Capture callbacks for subsystem notification (e.g., semantic embedding)
        self._on_memory_captured: list[Callable[[str, "EpisodicMemory"], None]] = []

        # Compression callbacks for subsystem cleanup (e.g., ConceptExtractor ref removal)
        self._on_memory_compressed: list[Callable[[str], None]] = []

        # Provenance collector (wired by MaximAgent.wire_provenance)
        self._collector: Any = None

        # Consolidation candidates queue (memories to evaluate for long-term promotion)
        # Bounded deque: oldest candidates are evicted when full so new memories
        # always get a chance at promotion instead of being silently dropped.
        self._consolidation_candidates: deque[str] = deque(maxlen=self.config.max_consolidation_candidates)

        # Track compressed vs full memory counts
        self._compressed_count: int = 0

        # Associative graph: memory nodes connected by recall-triggered
        # edges. Nodes here are memory_id strings; edges represent "this
        # memory record was recalled alongside that memory record". See
        # ``_restore_graph`` for the persistence shape.
        #
        # NOTE (Round 2 Arch important #3): this is a DIFFERENT graph
        # from ``self._binding_graph`` (P3a), which holds Hebbian edges
        # between substrate node IDs (not memory record IDs). The two
        # never share node-id namespaces and never cross-query. Stage 2
        # may rename one of them for clarity; Stage 1 keeps the names
        # distinguished via field comments + docstrings on the public
        # methods that query each.
        self._graph: DependencyGraph[str] = DependencyGraph()

        # Optional SCN reference for temporal-aware strategies
        self._scn: "SCN | None" = None

        # Async capture infrastructure
        self._capture_queue: queue.Queue[_CaptureRequest] = queue.Queue(maxsize=self.config.capture_queue_size)
        self._capture_worker_thread: threading.Thread | None = None
        self._capture_stop = threading.Event()

        # Dedup tracking: hash of recent observations → timestamp
        self._recent_observations: dict[int, float] = {}

        # ─────────────────────────────────────────────────────────────
        # P3a Stage 1 — Episode binding surface
        # ─────────────────────────────────────────────────────────────

        # Standalone store for closed episodes (held here rather than
        # inlined so P3b + P5 can extend EpisodeStore without touching
        # Hippocampus — Round 1 Arch important finding #1).
        self._episode_store: EpisodeStore = EpisodeStore()

        # Separate DependencyGraph for Hebbian co-activation edges
        # between SUBSTRATE NODE IDs (not memory record IDs). Kept
        # orthogonal to self._graph (which carries associative edges
        # between memory records) and orthogonal to ATL.graph (the
        # concept topology). See memory/episode.py for the binding-
        # graph ownership rationale.
        #
        # NOTE (Round 2 Arch important #3): node-id namespaces never
        # overlap with self._graph. Callers querying by memory record
        # id get nothing from this graph; callers querying by substrate
        # node id get nothing from self._graph. retrieve_on_cue() is
        # the Stage 1 public query surface and hits ONLY this graph.
        self._binding_graph: DependencyGraph[str] = DependencyGraph()

        # Pending-episode state — the episode currently being built,
        # one per Hippocampus instance. Created lazily on the first
        # observe_episode_event() call; cleared when the boundary
        # detector decides to close.
        self._pending_episode: PendingEpisodeState | None = None

        # Episode boundary detector with three default rules.
        # P3b will append per-channel rules via detector.add_rule().
        self._episode_detector = EpisodeBoundaryDetector(
            rules=[
                tick_gap_rule(self.config.episode.boundary_tick_gap),
                channel_change_rule(),
                scn_tag_change_rule(),
            ]
        )

        # Monotonic episode-id counter. Kept simple (string, not UUID)
        # so Stage 1 tests get deterministic ids.
        self._next_episode_ordinal: int = 0
        self._episode_lock = threading.RLock()

    # ─────────────────────────────────────────────────────────────────────────
    # Factory class methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: HippocampusConfig, persistence_path: str | None = None) -> "Hippocampus":
        """Create hippocampus from config, auto-loading saved state if path exists."""
        instance = cls(config=config)
        if persistence_path and os.path.exists(persistence_path):
            instance.load(persistence_path)
        return instance

    @classmethod
    def empty(cls, config: HippocampusConfig | None = None) -> "Hippocampus":
        """Create empty hippocampus with no saved state."""
        return cls(config=config or HippocampusConfig())

    # ─────────────────────────────────────────────────────────────────────────
    # SCN Integration
    # ─────────────────────────────────────────────────────────────────────────

    def connect_scn(self, scn: "SCN") -> None:
        """Connect SCN for temporal-aware sleep consolidation.

        When connected, the Hippocampus will:
        - Use TemporalAwareStrategy for smarter memory retention
        - Protect sole representatives of time slots
        - Preserve memories in rhythmic patterns
        - Enable temporal clustering for batch operations

        Args:
            scn: SCN instance to use for temporal queries
        """
        self._scn = scn
        # Register for memory deletion notifications
        self.register_deletion_callback(scn.remove_memory)
        logger.info("Connected SCN to Hippocampus (temporal-aware consolidation enabled)")

    @property
    def scn(self) -> "SCN | None":
        """Get the connected SCN, if any."""
        return self._scn

    # ─────────────────────────────────────────────────────────────────────────
    # MemoryLayer Protocol
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def layer_name(self) -> str:
        return "hippocampus"

    @property
    def graph(self) -> DependencyGraph:
        return self._graph

    def store(self, record: "EpisodicMemory", **kwargs: Any) -> str:
        """MemoryLayer protocol: store a pre-built EpisodicMemory."""
        return self.capture(record=record, **kwargs)

    def remove(self, record_id: str) -> None:
        """MemoryLayer protocol: remove a memory by ID."""
        with self._rwlock.write():
            if record_id in self._memories:
                mem = self._memories[record_id]
                if isinstance(mem, CompressedMemory):
                    self._compressed_count -= 1
                self._remove_memory(record_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Core Operations
    # ─────────────────────────────────────────────────────────────────────────

    def capture(
        self,
        perception: Perception | None = None,
        context: Context | None = None,
        decision: Decision | None = None,
        action: Action | None = None,
        outcome: Outcome | None = None,
        run_id: str = "",
        *,
        record: EpisodicMemory | None = None,
        state_snapshot: dict[str, Any] | None = None,
    ) -> str:
        """Capture a complete agentic loop as an episodic memory.

        Supports two paths:
        1. Individual args (existing API): perception, context, etc.
        2. Pre-built record (MemoryLayer protocol): record=EpisodicMemory(...)

        Args:
            perception: What was observed.
            context: State context at decision time.
            decision: What was decided and why.
            action: What action was taken.
            outcome: What happened as a result.
            run_id: Optional run identifier.
            record: Pre-built EpisodicMemory (overrides individual args).
            state_snapshot: Optional full state to store in StateStore.

        Returns:
            The memory_id of the captured memory.
        """
        memory_id, memory = self._build_capture_record(
            perception=perception,
            context=context,
            decision=decision,
            action=action,
            outcome=outcome,
            run_id=run_id,
            record=record,
            state_snapshot=state_snapshot,
        )

        with self._rwlock.write():
            self._insert_and_index_locked(memory_id, memory)

        self._emit_capture_telemetry(memory_id, memory)
        return memory_id

    def _build_capture_record(
        self,
        *,
        perception: Perception | None,
        context: Context | None,
        decision: Decision | None,
        action: Action | None,
        outcome: Outcome | None,
        run_id: str,
        record: EpisodicMemory | None,
        state_snapshot: dict[str, Any] | None,
    ) -> tuple[str, EpisodicMemory]:
        """Build a capture-ready (id, EpisodicMemory) tuple.

        Two paths: a pre-built record passes through unchanged, otherwise
        the individual percept/decision/action/outcome args are normalized
        and wrapped in a fresh ``EpisodicMemory``. Pre-lock so callers can
        keep critical-section work to the minimum.
        """
        if record is not None:
            return record.id, record

        memory_id = str(uuid4())
        now = time.time()

        # Store state snapshot and get reference
        if state_snapshot is not None:
            if context is None:
                context = Context()
            context.state_ref = self._state_store.store(state_snapshot)

        # Clamp salience to valid range
        if perception is not None and perception.salience is not None:
            perception.salience = max(0.0, min(1.0, perception.salience))

        memory = EpisodicMemory(
            id=memory_id,
            timestamp=now,
            run_id=run_id,
            created_at=now,
            accessed_at=now,
            access_count=1,
            perception=perception or Perception(),
            context=context or Context(),
            decision=decision or Decision(),
            action=action or Action(),
            outcome=outcome or Outcome(),
        )
        return memory_id, memory

    def _insert_and_index_locked(self, memory_id: str, memory: EpisodicMemory) -> None:
        """Insert + index a memory inside the write lock.

        Caller must hold ``self._rwlock.write()``. Handles capacity
        eviction, indexing, stats bookkeeping, immediate-promotion checks,
        consolidation-candidate enrollment, and associative-graph linking.
        """
        # Capacity check — evict before insert
        if len(self._memories) >= self.config.max_nodes:
            self._evict_one()

        self._memories[memory_id] = memory
        self._index_memory(memory_id, memory)

        # Update stats
        self._stats["memories_captured"] += 1
        if memory.outcome.success:
            self._stats["successful"] = self._stats.get("successful", 0) + 1
        else:
            self._stats["failed"] = self._stats.get("failed", 0) + 1

        # Check for immediate promotion to long-term (very high importance)
        if (
            memory.perception.salience >= self.config.immediate_promotion_salience
            or memory.perception.novelty >= self.config.immediate_promotion_novelty
        ):
            memory.long_term = True
            memory.consolidated_at = time.time()
            self._stats["long_term_count"] = self._stats.get("long_term_count", 0) + 1
            logger.debug("Immediately promoted memory %s to long-term", memory_id[:8])
        elif (
            memory.perception.salience > 0.7
            or memory.perception.novelty > 0.7
            or memory.perception.cli_input
            or memory.perception.transcript
        ):
            self._add_consolidation_candidate(memory_id)

        # Form associative edges to similar recalled memories
        if self.config.enable_associative_graph:
            self._form_associations(memory_id, memory)

    def _emit_capture_telemetry(self, memory_id: str, memory: EpisodicMemory) -> None:
        """Emit logging, sim traces, and capture-callback notifications.

        Runs after the write lock is released so callbacks don't deadlock
        or block other captures. Best-effort: callback failures are logged
        as warnings, sim-logger failures as debug.
        """
        logger.debug(
            "Captured memory %s: goal=%s, tool=%s, success=%s",
            memory_id[:8],
            memory.context.active_goal,
            memory.action.tool_name,
            memory.outcome.success,
        )

        # Simulation verbosity (best-effort trace).
        # Low-salience observations (settling cycles) go to debug only —
        # real action captures (tool_name set) always print.
        try:
            from maxim.simulation.sim_logger import sim_memory, sim_debug

            is_observation = not memory.action.tool_name
            msg = f"Captured: {memory.action.tool_name or 'observation'} (salience={memory.perception.salience:.2f})"
            if is_observation:
                sim_debug("HIPPOCAMPUS", msg, goal=memory.context.active_goal, success=memory.outcome.success)
            else:
                sim_memory(msg, goal=memory.context.active_goal, success=memory.outcome.success)
        except Exception as e:
            logger.debug("sim_logger unavailable for capture trace: %s", e)

        # Notify capture callbacks (e.g., for semantic embedding)
        for callback in self._on_memory_captured:
            try:
                callback(memory_id, memory)
            except Exception as e:
                logger.warning("Memory capture callback failed: %s", e)

    def store_observation(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Lightweight observation capture for NPC agents.

        Stores a text percept as a minimal EpisodicMemory without full
        decision/action/outcome structure.  Suitable for NPC agents that
        witness events but don't make complex tool-based decisions.

        Applies dedup: near-identical observations within ``dedup_window_s``
        (default 30s) are silently dropped to reduce memory spam.

        Args:
            text: The observation text (scene narrative, outcome description).
            metadata: Optional dict with extra context (agent_id, encounter, etc.).

        Returns:
            Memory ID (UUID string), or empty string if deduped.
        """
        # Dedup check — reject near-identical observations within window
        obs_hash = hash(text.strip().lower()[:200])
        now = time.time()
        window = self.config.dedup_window_s
        last_seen = self._recent_observations.get(obs_hash)
        if last_seen is not None and (now - last_seen) < window:
            return ""  # Deduped

        # Update dedup tracker (prune old entries)
        self._recent_observations[obs_hash] = now
        cutoff = now - window
        self._recent_observations = {h: t for h, t in self._recent_observations.items() if t > cutoff}

        perception = Perception(
            observations={"text": text, **(metadata or {})},
            detected_objects=[],
            detected_people=[],
            tool_alternatives=[],
            salience=0.5,
            novelty=0.3,
        )
        return self.capture(perception=perception)

    def capture_from_loop(
        self,
        observation: dict[str, Any],
        state: Any,
        intent: dict[str, Any],
        decision: dict[str, Any],
        action: dict[str, Any],
        result: Any,
        evaluations: list[dict[str, Any]] | None = None,
        run_id: str = "",
    ) -> str:
        """Convenience method to capture from agent_loop outputs.

        Converts the raw loop data into structured EpisodicMemory components.
        """
        # Extract perception
        perception = Perception(
            observations=observation if isinstance(observation, dict) else {},
            cli_input=observation.get("cli_input") if isinstance(observation, dict) else None,
            transcript=observation.get("transcript") if isinstance(observation, dict) else None,
            detected_objects=observation.get("detected_objects", []) if isinstance(observation, dict) else [],
            detected_people=observation.get("detected_people", []) if isinstance(observation, dict) else [],
            salience=observation.get("salience", 0.5) if isinstance(observation, dict) else 0.5,
            novelty=observation.get("novelty", 0.5) if isinstance(observation, dict) else 0.5,
        )

        # Extract state snapshot
        state_snapshot = None
        if hasattr(state, "snapshot") and callable(state.snapshot):
            try:
                state_snapshot = state.snapshot()
            except Exception as e:
                logger.warning("State snapshot failed during capture: %s", e)
        elif hasattr(state, "data"):
            state_snapshot = dict(state.data) if hasattr(state.data, "items") else {}

        # Build context
        ctx = Context(
            active_goal=intent.get("goal") if intent else None,
            active_mode=state_snapshot.get("mode", "observe") if state_snapshot else "observe",
        )

        # Build decision
        dec = Decision(
            intent=intent or {},
            reasoning=intent.get("reasoning") if intent else None,
            confidence=intent.get("confidence", 1.0) if intent else 1.0,
        )

        # Build action
        act = Action(
            tool_name=action.get("tool", "") if action else "",
            tool_params=action.get("params", {}) if action else {},
        )

        # Build outcome
        success = True
        error = None
        if hasattr(result, "success"):
            success = bool(result.success)
        if hasattr(result, "error"):
            error = str(result.error) if result.error else None

        out = Outcome(
            success=success,
            result=result
            if not hasattr(result, "to_dict")
            else result.to_dict()
            if hasattr(result, "to_dict")
            else str(result),
            error=error,
            evaluations=evaluations or [],
        )

        return self.capture(
            perception=perception,
            context=ctx,
            decision=dec,
            action=act,
            outcome=out,
            run_id=run_id,
            state_snapshot=state_snapshot,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Async Capture (Passive Hippocampus)
    # ─────────────────────────────────────────────────────────────────────────

    def start_capture_worker(
        self,
        thread_registry: Any | None = None,
    ) -> None:
        """Start the background capture worker thread.

        Args:
            thread_registry: Optional ThreadRegistry for coordinated shutdown.
        """
        self._capture_stop.clear()
        self._capture_worker_thread = threading.Thread(
            target=self._capture_worker,
            daemon=True,
            name="hippocampus-capture",
        )
        self._capture_worker_thread.start()
        if thread_registry is not None:
            thread_registry.register("hippocampus-capture", self._capture_worker_thread)

    def stop_capture_worker(self) -> None:
        """Stop the background capture worker. Drains queue first."""
        self.flush(timeout=5.0)
        self._capture_stop.set()
        if self._capture_worker_thread is not None and self._capture_worker_thread.is_alive():
            self._capture_worker_thread.join(timeout=2.0)

    def capture_from_loop_async(self, **kwargs: Any) -> None:
        """Non-blocking capture: queue for background processing.

        Snapshots mutable data immediately to avoid closure issues.
        If the queue is full, drops the oldest capture (logged warning).

        Accepts the same kwargs as capture_from_loop().
        """
        # Snapshot state NOW to avoid stale references
        state = kwargs.get("state")
        state_snapshot = None
        if hasattr(state, "snapshot") and callable(state.snapshot):
            try:
                state_snapshot = state.snapshot()
            except Exception as e:
                logger.warning("State snapshot failed during capture: %s", e)
        elif hasattr(state, "data") and hasattr(state.data, "items"):
            state_snapshot = dict(state.data)

        request = _CaptureRequest(
            observation=dict(kwargs.get("observation", {})),
            state_snapshot=state_snapshot,
            intent=dict(kwargs.get("intent", {})),
            decision=dict(kwargs.get("decision", {})),
            action=dict(kwargs.get("action", {})),
            result=kwargs.get("result"),
            run_id=kwargs.get("run_id", ""),
            queued_at=time.time(),
        )
        try:
            self._capture_queue.put(request, timeout=0.1)
        except queue.Full:
            # Queue full — drop oldest to make room.  The get+put is not
            # atomic, but the worst case is a dropped item (acceptable)
            # rather than a crash (unacceptable).
            logger.warning("Hippocampus capture queue full, dropping oldest")
            try:
                self._capture_queue.get_nowait()
                self._capture_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._capture_queue.put_nowait(request)
            except queue.Full:
                logger.warning("Hippocampus capture queue still full after drop — item lost")

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until all queued captures are processed.

        Call before session-end consolidation or final save.
        Returns True if queue drained within timeout.

        Uses the Queue's internal condition variable (``all_tasks_done``)
        instead of polling ``empty()`` — this is both more reliable and
        more efficient (no busy-wait).
        """
        if self._capture_queue.unfinished_tasks == 0:
            return True
        with self._capture_queue.all_tasks_done:
            if self._capture_queue.unfinished_tasks > 0:
                self._capture_queue.all_tasks_done.wait(timeout=timeout)
        if self._capture_queue.unfinished_tasks > 0:
            logger.warning(
                "Hippocampus flush timed out, %d items remain",
                self._capture_queue.unfinished_tasks,
            )
            return False
        return True

    def _capture_worker(self) -> None:
        """Background thread: drain capture queue, process each."""
        while not self._capture_stop.is_set():
            try:
                request = self._capture_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_capture(request)
            except Exception as e:
                logger.error("Async capture failed: %s", e)
            finally:
                self._capture_queue.task_done()

    def _process_capture(self, request: _CaptureRequest) -> None:
        """Process a queued capture request. Runs on worker thread."""
        latency_ms = (time.time() - request.queued_at) * 1000
        logger.debug("Processing capture (queue latency: %.1fms)", latency_ms)
        self.capture_from_loop(
            observation=request.observation,
            state=_SnapshotState(request.state_snapshot),
            intent=request.intent,
            decision=request.decision,
            action=request.action,
            result=request.result,
            run_id=request.run_id,
        )

    def get(self, memory_id: str) -> EpisodicMemory | CompressedMemory | None:
        """Get a memory by its ID.

        Updates the access timestamp and count.

        Args:
            memory_id: The memory ID to retrieve.

        Returns:
            The memory (EpisodicMemory or CompressedMemory), or None if not found.
        """
        with self._rwlock.write():  # Write lock because we update access tracking
            memory = self._memories.get(memory_id)
            if memory is not None:
                memory.accessed_at = time.time()
                memory.access_count += 1
            return memory

    def recall_by_ids(
        self,
        record_ids: list[str],
    ) -> list[EpisodicMemory | CompressedMemory]:
        """Bulk-load memories by ID. Skips missing IDs.

        Single lock acquisition for the entire batch — more efficient
        than iterating get() which acquires a write lock per call.
        Does NOT update access tracking (callers doing bulk analysis
        shouldn't inflate access counts).
        """
        with self._rwlock.read():
            results = []
            for rid in record_ids:
                mem = self._memories.get(rid)
                if mem is not None:
                    results.append(mem)
            return results

    def search_by_content(self, query: str, limit: int = 20) -> list[EpisodicMemory | CompressedMemory]:
        """Search memories by text content across all fields.

        Searches perception.observations, outcome.result, and
        decision.reasoning for the query string (case-insensitive).

        Args:
            query: Text to search for.
            limit: Maximum number of results.

        Returns:
            Matching memories, most recent first.
        """
        if not query:
            return []

        query_lower = query.lower()
        results: list[EpisodicMemory | CompressedMemory] = []

        with self._rwlock.read():
            for memory in self._memories.values():
                if self._memory_matches_query(memory, query_lower):
                    results.append(memory)

            results.sort(key=lambda m: m.timestamp, reverse=True)
            return results[:limit]

    @staticmethod
    def _memory_matches_query(memory: Any, query_lower: str) -> bool:
        """Check if a memory's text content matches the query."""
        # Check perception observations
        perception = getattr(memory, "perception", None)
        if perception is not None:
            obs = getattr(perception, "observations", {})
            if obs and query_lower in str(obs).lower():
                return True

        # Check outcome result
        outcome = getattr(memory, "outcome", None)
        if outcome is not None:
            result = getattr(outcome, "result", None)
            if result and query_lower in str(result).lower():
                return True

        # Check decision reasoning
        decision = getattr(memory, "decision", None)
        if decision is not None:
            reasoning = getattr(decision, "reasoning", None)
            if reasoning and query_lower in reasoning.lower():
                return True

        # Check compressed summary
        summary = getattr(memory, "summary", None)
        if summary and query_lower in summary.lower():
            return True

        return False

    def get_state(self, state_ref: str) -> dict[str, Any] | None:
        """Retrieve a full state snapshot from the StateStore.

        Args:
            state_ref: The state reference from a memory's context.state_ref.

        Returns:
            The full state dictionary, or None if evicted.
        """
        return self._state_store.get(state_ref)

    # ─────────────────────────────────────────────────────────────────────────
    # Index Views (for subsystem access)
    # ─────────────────────────────────────────────────────────────────────────

    def get_memories_by_index(self, index_key: str) -> list[EpisodicMemory]:
        """Get all memories matching an index key.

        Args:
            index_key: Full index key like "goal:find cup" or "tool:speak".

        Returns:
            List of matching memories.
        """
        with self._rwlock.read():
            memory_ids = self._context_index.get(index_key, set())
            return [self._memories[mid] for mid in memory_ids if mid in self._memories]

    def index_keys(self) -> list[str]:
        """Return all index keys."""
        with self._rwlock.read():
            return list(self._context_index.keys())

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return hippocampus statistics."""
        with self._rwlock.read():
            # Count long-term memories
            long_term_count = sum(1 for m in self._memories.values() if getattr(m, "long_term", False))

            # Count graph nodes and edges
            graph_stats = self._graph.to_dict()
            graph_node_count = len(graph_stats.get("nodes", []))
            graph_edge_count = len(graph_stats.get("edges", []))

            return {
                **self._stats,
                "total_memories": len(self._memories),
                "compressed_memories": self._compressed_count,
                "full_memories": len(self._memories) - self._compressed_count,
                "long_term_memories": long_term_count,
                "index_keys": len(self._context_index),
                "consolidation_candidates": len(self._consolidation_candidates),
                "state_store": self._state_store.stats(),
                "graph_nodes": graph_node_count,
                "graph_edges": graph_edge_count,
                "graph_enabled": self.config.enable_associative_graph,
            }

    def check_consistency(self) -> list[str]:
        """Check for consistency issues between memories and indices.

        Returns:
            List of issue descriptions (empty if consistent).
        """
        issues = []

        with self._rwlock.read():
            # Check: All indexed memory IDs exist
            for index_key, memory_ids in self._context_index.items():
                for memory_id in memory_ids:
                    if memory_id not in self._memories:
                        issues.append(f"Orphan index entry: {index_key} -> {memory_id[:8]}")

            # Check: All memories are indexed
            for memory_id in self._memories:
                if memory_id not in self._memory_contexts:
                    issues.append(f"Unindexed memory: {memory_id[:8]}")

            # Check: Reverse index consistency
            for memory_id, index_keys in self._memory_contexts.items():
                for index_key in index_keys:
                    if memory_id not in self._context_index.get(index_key, set()):
                        issues.append(f"Reverse index mismatch: {memory_id[:8]} claims {index_key}")

            # Check: Graph node consistency (nodes should reference existing memories)
            graph_data = self._graph.to_dict()
            for node_id in graph_data.get("nodes", []):
                if node_id not in self._memories:
                    issues.append(f"Orphan graph node: {node_id[:8]} not in memories")

            # Check: Graph edge consistency (edges should reference existing nodes)
            for edge_data in graph_data.get("edges", []):
                source = edge_data.get("source", "")
                target = edge_data.get("target", "")
                if source not in self._memories:
                    issues.append(f"Graph edge source missing: {source[:8]}")
                if target not in self._memories:
                    issues.append(f"Graph edge target missing: {target[:8]}")

        return issues

    def repair_consistency(self) -> int:
        """Repair consistency issues.

        Returns:
            Number of issues repaired.
        """
        with self._rwlock.write():
            repaired = 0

            # Remove orphan index entries
            for index_key in list(self._context_index.keys()):
                original_count = len(self._context_index[index_key])
                valid_ids = {mid for mid in self._context_index[index_key] if mid in self._memories}
                removed = original_count - len(valid_ids)
                if removed > 0:
                    self._context_index[index_key] = valid_ids
                    repaired += removed

                # Remove empty index entries
                if not self._context_index[index_key]:
                    del self._context_index[index_key]

            # Rebuild reverse index
            self._memory_contexts.clear()
            for index_key, memory_ids in self._context_index.items():
                for memory_id in memory_ids:
                    self._memory_contexts[memory_id].add(index_key)

            # Re-index any unindexed memories
            for memory_id, memory in self._memories.items():
                if memory_id not in self._memory_contexts or not self._memory_contexts[memory_id]:
                    self._index_memory(memory_id, memory)
                    repaired += 1

            # Remove orphan graph nodes (nodes whose memories no longer exist)
            graph_data = self._graph.to_dict()
            for node_id in graph_data.get("nodes", []):
                if node_id not in self._memories:
                    self._graph.remove_node(node_id)
                    repaired += 1

            if repaired > 0:
                logger.info("Repaired %d consistency issues", repaired)

            return repaired

    def __len__(self) -> int:
        """Return number of memories."""
        with self._rwlock.read():
            return len(self._memories)

    def __iter__(self) -> Iterator[EpisodicMemory | CompressedMemory]:
        """Iterate over all memories (both full and compressed)."""
        with self._rwlock.read():
            yield from self._memories.values()

    # ─────────────────────────────────────────────────────────────────────────
    # P3a Stage 1 — Episode binding (public surface)
    # ─────────────────────────────────────────────────────────────────────────

    def observe_episode_event(self, event: CaptureEvent) -> None:
        """Feed an event into the episode boundary detector.

        If no pending episode exists, this opens one. Otherwise the
        detector's rule list decides whether to close the pending
        episode first (which triggers Hebbian updates on the binding
        graph) and open a new one.

        This method is separate from ``capture()`` / ``capture_from_loop``
        to keep P3a Stage 1 fully isolated from the existing memory
        capture path. Production integration (wiring the agent loop to
        call this alongside capture) is a post-Stage-1 task.
        """
        with self._episode_lock:
            if self._pending_episode is None:
                self._pending_episode = self._start_episode(event)
                return

            if self._episode_detector.should_close(self._pending_episode, event):
                self._close_pending_episode_locked()
                self._pending_episode = self._start_episode(event)
                return

            # Extend the pending episode with this event
            pending = self._pending_episode
            pending.last_tick = event.tick
            if event.sender_id is not None:
                pending.sender_ids.add(event.sender_id)
            if event.thread_id is not None and pending.thread_id is None:
                pending.thread_id = event.thread_id
            for node_id in event.activated_nodes:
                pending.activated_nodes.append(node_id)

    def finalize_pending_episode(self) -> Episode | None:
        """Force-close the current pending episode, if any.

        Used by tests + by end-of-session teardown. Returns the closed
        episode (or ``None`` if there was nothing pending).
        """
        with self._episode_lock:
            if self._pending_episode is None:
                return None
            return self._close_pending_episode_locked()

    def retrieve_on_cue(
        self,
        cue_node_id: str,
        limit: int = 10,
        *,
        multi_hop: bool = True,
        node_filter: Callable[[str], bool] | None = None,
    ) -> list[tuple[str, float]]:
        """Return substrate nodes co-activated with the cue, ranked by binding weight.

        **Two retrieval modes:**

        - ``multi_hop=True`` (default, Stage 2+ primary path) — walks
          the binding graph via
          ``DependencyGraph.spreading_activation`` with decay
          ``self.config.episode.retrieval.decay``, threshold
          ``self.config.episode.retrieval.threshold``, and max depth
          ``self.config.episode.retrieval.max_depth``. Recovers nodes
          reachable through intermediate edges, including transitive
          structure that bag-of-words baselines (e.g., TF-IDF) cannot
          represent. The Hebbian binding mechanism's structural
          advantage over TF-IDF manifests here. Stage 2's
          architectural finding: one-hop Hebbian ≈ TF-IDF on bag-of-
          words; the value lives in multi-hop traversal.
        - ``multi_hop=False`` (explicit opt-in) — one-hop retrieval
          via ``DependencyGraph.get_associated``. Returns direct
          neighbors of the cue sorted by edge weight. Used by Stage 1
          mechanism tests that specifically exercise the one-hop
          semantics. Production callers (P3b / P4 / P5 / P6) should
          leave the default alone.

        **Default flipped from one-hop to multi-hop in Stage 2** per
        the Arch-lens review — the Stage 1 "backward compat" hedge
        protected exactly one test suite while silently degrading every
        future caller that forgets the kwarg. Stage 1 mechanism tests
        explicitly opt into ``multi_hop=False``.

        ``node_filter`` is an optional callable ``(node_id) -> bool``
        applied to every intermediate and destination node visited
        during traversal. Edges through nodes the filter rejects are
        not followed (the multi-hop path drops them before the decay
        step; the one-hop path drops them from the returned list).
        This is the P3b channel-filter / P4 modality-filter seam —
        reserved in Stage 2 so P3b doesn't have to reach into the
        binding graph internals or rebuild this method.

        **Return shape is distinct from ``recall``/``recall_similar``**.
        Those methods return ``list[EpisodicMemory | CompressedMemory]``
        (memory records). This method returns
        ``list[tuple[node_id, weight]]`` where ``node_id`` is a
        SUBSTRATE node identifier (from ``LinguisticEncoder`` + EC),
        NOT a memory record id. The two namespaces never overlap.

        Queries ``self._binding_graph`` ONLY. Does NOT touch
        ``self._graph`` (memory-record associations).
        """
        if multi_hop:
            retrieval_cfg = self.config.episode.retrieval
            activations = self._binding_graph.spreading_activation(
                source_ids=[cue_node_id],
                initial_activation=1.0,
                decay=retrieval_cfg.decay,
                threshold=retrieval_cfg.threshold,
                max_depth=retrieval_cfg.max_depth,
                node_filter=node_filter,
            )
            # spreading_activation returns dict[str, float] including the
            # cue itself with activation 1.0; filter it out.
            ranked = sorted(
                ((node, score) for node, score in activations.items() if node != cue_node_id),
                key=lambda pair: pair[1],
                reverse=True,
            )
            return ranked[:limit]

        associated = self._binding_graph.get_associated(
            cue_node_id,
            edge_types={EdgeType.ASSOCIATES},
        )
        # get_associated returns list[(neighbor, weight)]; sort + cap
        if node_filter is not None:
            associated = [(n, w) for n, w in associated if node_filter(n)]
        associated.sort(key=lambda pair: pair[1], reverse=True)
        return associated[:limit]

    def add_boundary_rule(self, rule: BoundaryRule) -> None:
        """Append a boundary rule to the episode detector.

        Public P3b extension seam (Round 2 Arch important #1). P3b
        channel integration will call this to register per-channel
        rules without reaching into ``self._episode_detector``
        directly. The rule is appended to the existing rule list and
        consulted on every subsequent ``observe_episode_event`` call.

        **Boundary rules are NOT persisted** (Round 1 Exec important
        #3 for P3b). After ``Hippocampus.load_state``, the reloaded
        instance's ``_episode_detector`` contains only whatever rules
        the new instance's ``__init__`` registered (the defaults).
        Callers that added P3b channel rules via this method must
        re-add them at construction time on every load.
        """
        self._episode_detector.add_rule(rule)

    def episode_filter(
        self,
        *,
        membership_mode: Literal["any", "exclusive"] = "any",
        **criteria: Any,
    ) -> Callable[[str], bool]:
        """Convenience forwarder for ``EpisodeStore.episode_membership_filter``.

        Builds a ``node_filter`` callable suitable for
        ``Hippocampus.retrieve_on_cue(cue, node_filter=...)`` by
        forwarding ``**criteria`` to the underlying store. The general
        filter lives on ``EpisodeStore`` (Round 1 Arch important #3) —
        that's where the inverted index is. This method exists for
        ergonomics on Hippocampus call sites; it adds no logic of its
        own.

        **Round 2 fold: replaces the earlier
        ``channel_membership_filter(channel, sender)`` per-axis
        alias** (Round 2 Arch important #1). The narrow alias would
        have forced P4 cross-modal to add ``modality_membership_filter``,
        P5 to add ``stress_membership_filter``, and so on — N
        per-axis aliases each requiring sync with the underlying
        signature. The general ``**criteria`` form takes any
        combination of ``Episode`` field matchers and inherits future
        axes for free.

        Examples::

            # P3b: SMS channel only
            h.episode_filter(channel="sms")

            # P3b: SMS + alice as sender (sender_ids is a tuple — pass
            # the single string to match via membership)
            h.episode_filter(channel="sms", sender_ids="alice")

            # P4 will use: vision modality only
            h.episode_filter(modality="vision")

            # Combinations
            h.episode_filter(channel="sms", thread_id="t_123")

        ``membership_mode``:
        - ``"any"`` (default, Stage 1): node retained if it appears
          in ≥1 matching episode.
        - ``"exclusive"`` (committed parameter shape, primary
          consumer is P4 cross-modal): node retained ONLY if every
          episode containing it matches all criteria.

        Unknown criterion keys raise ``ValueError`` at filter build
        time (Round 2 cross-confirmed critical). Collection-typed
        criterion VALUES raise ``TypeError`` (subset semantics are
        deferred to a future plan).
        """
        return self._episode_store.episode_membership_filter(
            membership_mode=membership_mode,
            **criteria,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # P3a Stage 1 — Episode binding (private helpers)
    # ─────────────────────────────────────────────────────────────────────────

    def _start_episode(self, event: CaptureEvent) -> PendingEpisodeState:
        """Open a fresh pending episode seeded by ``event``. Caller must
        hold ``self._episode_lock``."""
        self._next_episode_ordinal += 1
        pending = PendingEpisodeState(
            id=f"ep_{self._next_episode_ordinal}",
            start_tick=event.tick,
            last_tick=event.tick,
            channel=event.channel,
            sender_ids=set(),
            thread_id=event.thread_id,
            activated_nodes=list(event.activated_nodes),
            reward_events=[],
            scn_tag=event.scn_tag,
        )
        if event.sender_id is not None:
            pending.sender_ids.add(event.sender_id)
        return pending

    def _close_pending_episode_locked(self) -> Episode:
        """Finalize ``self._pending_episode``, add it to the store, and
        apply Hebbian updates to the binding graph. Caller must hold
        ``self._episode_lock``."""
        assert self._pending_episode is not None, "caller must check before closing"
        episode = self._pending_episode.finalize()
        self._pending_episode = None

        # Add to store first (so the store has the episode before Hebbian
        # edges reference its nodes). EpisodeStore takes its own lock;
        # since we already hold self._episode_lock, the ordering is
        # deterministic: episode_lock → episode_store._lock →
        # binding_graph._lock.
        #
        # **Lock-inversion safety (P3b Round 2 self-found critical):**
        # this ordering would deadlock against any caller that holds
        # binding_graph._lock and then tries to acquire
        # episode_store._lock. The most natural such caller is a
        # node_filter callback passed to retrieve_on_cue —
        # spreading_activation holds binding_graph._lock for the entire
        # BFS and invokes node_filter while the lock is held. P3b's
        # EpisodeStore.episode_membership_filter would have hit this
        # exact inversion if it had been a lazy lookup. The fix lives
        # on the FILTER side: episode_membership_filter snapshots the
        # allowed-node set under episode_store._lock once at filter
        # construction and returns a lock-free closure. Any future
        # node_filter implementation that re-acquires
        # episode_store._lock at call time MUST adopt the same
        # snapshot pattern, OR this code MUST be reordered to acquire
        # binding_graph._lock first. See
        # tests/substrate/test_p3b_channel_integration.py
        # ::TestConcurrency::test_filter_closure_holds_no_lock_after_construction.
        self._episode_store.add(episode)

        # Hebbian updates on the binding graph
        hebbian_cfg = self.config.episode.hebbian
        apply_hebbian_on_close(
            self._binding_graph,
            episode,
            hebbian_init=hebbian_cfg.init,
            hebbian_delta=hebbian_cfg.delta,
            hebbian_max=hebbian_cfg.max_weight,
        )

        return episode

    # ─────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _index_memory(self, memory_id: str, memory: EpisodicMemory) -> None:
        """Build hash index entries for a memory (lock must be held)."""
        index_entries: list[str] = []

        # Index by goal
        if memory.context.active_goal:
            index_entries.append(f"goal:{memory.context.active_goal}")

        # Index by tool
        if memory.action.tool_name:
            index_entries.append(f"tool:{memory.action.tool_name}")

        # Index by success/failure
        index_entries.append(f"success:{memory.outcome.success}")

        # Index by mode
        if memory.context.active_mode:
            index_entries.append(f"mode:{memory.context.active_mode}")

        # Index by detected objects
        for obj in memory.perception.detected_objects:
            index_entries.append(f"object:{obj}")

        # Index by detected people
        for person in memory.perception.detected_people:
            index_entries.append(f"person:{person}")

        # Add all entries to indices
        for index_key in index_entries:
            self._context_index[index_key].add(memory_id)
            self._memory_contexts[memory_id].add(index_key)

    def _make_index_key(self, key: str, value: Any) -> str:
        """Create an index key from a key-value pair."""
        return f"{key}:{value}"

    def _remove_memory(self, memory_id: str) -> None:
        """Remove a memory and clean up all references (lock must be held).

        This method handles:
        1. Context index cleanup
        2. Associative graph edge cleanup
        3. Subsystem notification via callbacks
        """
        # Clean up context index
        for index_key in self._memory_contexts.get(memory_id, set()):
            self._context_index[index_key].discard(memory_id)
            if not self._context_index[index_key]:
                del self._context_index[index_key]

        self._memory_contexts.pop(memory_id, None)

        # Clean up associative graph node and its edges
        if self._graph.get_node(memory_id) is not None:
            self._graph.remove_node(memory_id)

        # Notify subsystems of deletion
        for callback in self._on_memory_deleted:
            try:
                callback(memory_id)
            except Exception as e:
                logger.warning("Memory deletion callback failed: %s", e)

        self._memories.pop(memory_id, None)

    def _evict_one(self) -> None:
        """Evict the lowest-scored memory to make room.

        Reuses _get_memory_strategy() for consistent scoring between
        store-time eviction and sleep consolidation. Never evicts
        long-term memories at store time. Must be called with write lock held.
        """
        import heapq

        from maxim.memory.types import CompressedMemory

        strategy = self._get_memory_strategy()

        now = time.time()
        scored: list[tuple[float, str]] = []

        for memory_id, record in self._memories.items():
            if record.long_term:
                continue
            edge_count = (
                record.edge_count
                if isinstance(record, CompressedMemory)
                else len(self._graph.get_associated(memory_id))
            )
            score = strategy.score_for_retention(record, now, edge_count)
            heapq.heappush(scored, (score, memory_id))

        if scored:
            _, evict_id = heapq.heappop(scored)
            self._remove_memory(evict_id)
            logger.debug("Evicted memory %s at capacity", evict_id[:8])

    def register_deletion_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback to be notified when memories are deleted.

        Subsystems like SCN, NAc, and EC can register to clean up their
        own indices when memories are removed.

        Args:
            callback: Function that takes memory_id and handles cleanup.
        """
        self._on_memory_deleted.append(callback)

    def register_capture_callback(
        self,
        callback: Callable[[str, "EpisodicMemory"], None],
    ) -> None:
        """Register a callback to be notified when memories are captured.

        Subsystems like EC can register to schedule async semantic embedding
        when new memories are captured.

        Args:
            callback: Function that takes (memory_id, memory) and handles processing.
        """
        self._on_memory_captured.append(callback)

    def register_compression_callback(
        self,
        callback: Callable[[str], None],
    ) -> None:
        """Register a callback to be notified when memories are compressed.

        Subsystems like ConceptExtractor can register to clean up references
        when a full EpisodicMemory is replaced with CompressedMemory.

        Args:
            callback: Function that takes memory_id and handles cleanup.
        """
        self._on_memory_compressed.append(callback)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Sleep Consolidation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_memory_strategy(self) -> "MemoryStrategy":
        """Get the configured memory strategy.

        If SCN is connected, wraps the base strategy with TemporalAwareStrategy
        for temporal-context-aware memory consolidation.
        """
        from maxim.memory.strategies import (
            AccessBasedStrategy,
            CompositeStrategy,
            ImportanceBasedStrategy,
            TemporalAwareStrategy,
        )

        strategy_name = self.config.memory_strategy

        # Build base strategy
        if strategy_name == "access_based":
            base_strategy = AccessBasedStrategy(
                max_age_without_access=self.config.max_age_without_access,
                compression_age=self.config.compression_age,
            )
        elif strategy_name == "importance_based":
            base_strategy = ImportanceBasedStrategy(
                compression_age=self.config.compression_age,
            )
        elif strategy_name == "composite":
            base_strategy = CompositeStrategy(
                [
                    (
                        AccessBasedStrategy(
                            max_age_without_access=self.config.max_age_without_access,
                            compression_age=self.config.compression_age,
                        ),
                        0.6,
                    ),
                    (
                        ImportanceBasedStrategy(
                            compression_age=self.config.compression_age,
                        ),
                        0.4,
                    ),
                ]
            )
        else:
            # Default to access-based
            base_strategy = AccessBasedStrategy(
                max_age_without_access=self.config.max_age_without_access,
                compression_age=self.config.compression_age,
            )

        # Wrap with TemporalAwareStrategy if SCN is connected
        if self._scn is not None:
            strategy = TemporalAwareStrategy(
                scn=self._scn,
                base_strategy=base_strategy,
            )
            # Pre-compute bin populations for efficient lookups
            strategy.prepare()
            return strategy

        return base_strategy

    def touch(self, memory_id: str) -> None:
        """Update the accessed_at timestamp for a memory.

        Called automatically when a memory is retrieved via get() or recall().
        Can also be called manually to mark a memory as "used".
        """
        with self._rwlock.write():
            self._touch_internal(memory_id)

    def _touch_internal(self, memory_id: str) -> None:
        """Internal touch (lock must be held)."""
        if memory_id in self._memories:
            record = self._memories[memory_id]
            record.accessed_at = time.time()
            record.access_count += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal Clustering (SCN-integrated consolidation)
    # ─────────────────────────────────────────────────────────────────────────
