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

import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator
from uuid import uuid4

if TYPE_CHECKING:
    from maxim.memory.strategies import MemoryStrategy
    from maxim.time.scn import SCN

from maxim.agents.bus import DependencyGraph, EdgeType
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


@dataclass
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
        default_factory=lambda: frozenset(
            {"goal", "tool", "object", "person", "success", "mode"}
        )
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

    # Spreading activation defaults
    spreading_activation_decay: float = 0.5
    spreading_activation_max_depth: int = 3
    spreading_activation_threshold: float = 0.05

    # ─────────────────────────────────────────────────────────────────────────
    # Async Capture Settings
    # ─────────────────────────────────────────────────────────────────────────

    # Maximum pending captures before drop-oldest
    capture_queue_size: int = 100


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


class Hippocampus(MemoryLayer):
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
        self._consolidation_candidates: list[str] = []

        # Track compressed vs full memory counts
        self._compressed_count: int = 0

        # Associative graph: memory nodes connected by recall-triggered edges
        self._graph: DependencyGraph[str] = DependencyGraph()

        # Optional SCN reference for temporal-aware strategies
        self._scn: "SCN | None" = None

        # Async capture infrastructure
        self._capture_queue: queue.Queue[_CaptureRequest] = queue.Queue(
            maxsize=self.config.capture_queue_size
        )
        self._capture_worker_thread: threading.Thread | None = None
        self._capture_stop = threading.Event()

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
        if record is not None:
            # Pre-built path: use record directly
            memory = record
            memory_id = memory.id
        else:
            # Existing path: construct from individual args
            memory_id = str(uuid4())
            now = time.time()

            # Store state snapshot and get reference
            if state_snapshot is not None:
                if context is None:
                    context = Context()
                context.state_ref = self._state_store.store(state_snapshot)

            # Create the memory
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

        with self._rwlock.write():
            # Capacity check — evict before insert
            if len(self._memories) >= self.config.max_nodes:
                self._evict_one()

            # Store the memory
            self._memories[memory_id] = memory

            # Build index entries
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

            # Add to consolidation candidates if potentially important
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

        logger.debug(
            "Captured memory %s: goal=%s, tool=%s, success=%s",
            memory_id[:8],
            memory.context.active_goal,
            memory.action.tool_name,
            memory.outcome.success,
        )

        # Notify capture callbacks (e.g., for semantic embedding)
        for callback in self._on_memory_captured:
            try:
                callback(memory_id, memory)
            except Exception as e:
                logger.warning("Memory capture callback failed: %s", e)

        return memory_id

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
            except Exception:
                pass
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
            result=result if not hasattr(result, "to_dict") else result.to_dict() if hasattr(result, "to_dict") else str(result),
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
            thread_registry.register(
                "hippocampus-capture", self._capture_worker_thread
            )

    def stop_capture_worker(self) -> None:
        """Stop the background capture worker. Drains queue first."""
        self.flush(timeout=5.0)
        self._capture_stop.set()
        if (
            self._capture_worker_thread is not None
            and self._capture_worker_thread.is_alive()
        ):
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
            except Exception:
                pass
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
            self._capture_queue.put_nowait(request)
        except queue.Full:
            logger.warning("Hippocampus capture queue full, dropping oldest")
            try:
                self._capture_queue.get_nowait()
                self._capture_queue.task_done()
            except queue.Empty:
                pass
            self._capture_queue.put_nowait(request)

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until all queued captures are processed.

        Call before session-end consolidation or final save.
        Returns True if queue drained within timeout.
        """
        deadline = time.monotonic() + timeout
        while not self._capture_queue.empty():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "Hippocampus flush timed out, %d items remain",
                    self._capture_queue.qsize(),
                )
                return False
            time.sleep(0.05)
        # Final check: all task_done() calls completed
        with self._capture_queue.all_tasks_done:
            remaining = max(0, deadline - time.monotonic())
            if self._capture_queue.unfinished_tasks > 0:
                self._capture_queue.all_tasks_done.wait(timeout=remaining)
        return self._capture_queue.unfinished_tasks == 0

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

    def recall(
        self,
        limit: int = 10,
        *,
        goal: str | None = None,
        tool: str | None = None,
        success: bool | None = None,
        object_detected: str | None = None,
        person_detected: str | None = None,
        mode: str | None = None,
        time_after: float | None = None,
        time_before: float | None = None,
        include_compressed: bool = True,
    ) -> list[EpisodicMemory | CompressedMemory]:
        """Find memories matching filters (hash lookup + filtering).

        All filters are AND-ed together. Use None to skip a filter.

        Args:
            limit: Maximum number of memories to return.
            goal: Match memories with this goal text.
            tool: Match memories that used this tool.
            success: Match memories with this success status.
            object_detected: Match memories that detected this object.
            person_detected: Match memories that detected this person.
            mode: Match memories in this mode.
            time_after: Only memories after this timestamp.
            time_before: Only memories before this timestamp.
            include_compressed: Include CompressedMemory results (default True).

        Returns:
            List of matching memories (EpisodicMemory or CompressedMemory), most recent first.
        """
        with self._rwlock.write():  # Write for stats update
            self._stats["queries"] = self._stats.get("queries", 0) + 1

            # Build index filters
            index_filters: dict[str, Any] = {}
            if goal is not None:
                index_filters["goal"] = goal
            if tool is not None:
                index_filters["tool"] = tool
            if success is not None:
                index_filters["success"] = success
            if object_detected is not None:
                index_filters["object"] = object_detected
            if person_detected is not None:
                index_filters["person"] = person_detected
            if mode is not None:
                index_filters["mode"] = mode

            # Find candidates via index intersection
            candidates: set[str] | None = None

            if index_filters:
                for key, value in index_filters.items():
                    index_key = self._make_index_key(key, value)
                    matching = self._context_index.get(index_key, set())
                    if candidates is None:
                        candidates = matching.copy()
                    else:
                        candidates &= matching

                    # Early exit if no matches
                    if not candidates:
                        return []
            else:
                # No filters - all memories are candidates
                candidates = set(self._memories.keys())

            # Filter by time range and collect results
            results: list[EpisodicMemory | CompressedMemory] = []
            for memory_id in candidates:
                memory = self._memories.get(memory_id)
                if memory is None:
                    continue

                # Skip compressed if not wanted
                if not include_compressed and isinstance(memory, CompressedMemory):
                    continue

                # Time filters
                if time_after is not None and memory.timestamp < time_after:
                    continue
                if time_before is not None and memory.timestamp > time_before:
                    continue

                results.append(memory)

            # Sort by timestamp (most recent first) and limit
            results.sort(key=lambda m: m.timestamp, reverse=True)
            return results[:limit]

    def recall_similar(
        self,
        perception: Perception,
        limit: int = 5,
        include_compressed: bool = True,
    ) -> list[EpisodicMemory | CompressedMemory]:
        """Find memories similar to the given perception.

        Scores memories by overlapping detected objects and people.
        Phase 4 will add semantic similarity via embeddings.

        Args:
            perception: Current perception to match against.
            limit: Maximum number of memories to return.
            include_compressed: Include CompressedMemory results.

        Returns:
            List of similar memories, most similar first.
        """
        with self._rwlock.write():  # Write for stats
            self._stats["queries"] = self._stats.get("queries", 0) + 1

            scores: dict[str, float] = defaultdict(float)

            # Score by overlapping detected objects
            for obj in perception.detected_objects:
                index_key = f"object:{obj}"
                for memory_id in self._context_index.get(index_key, set()):
                    scores[memory_id] += 1.0

            # Score by overlapping detected people
            for person in perception.detected_people:
                index_key = f"person:{person}"
                for memory_id in self._context_index.get(index_key, set()):
                    scores[memory_id] += 1.0

            # Sort by score, then by timestamp
            def get_timestamp(mid: str) -> float:
                mem = self._memories.get(mid)
                return mem.timestamp if mem else 0.0

            scored_ids = sorted(
                scores.keys(),
                key=lambda mid: (scores[mid], get_timestamp(mid)),
                reverse=True,
            )

            # Collect results
            results: list[EpisodicMemory | CompressedMemory] = []
            for memory_id in scored_ids[:limit]:
                memory = self._memories.get(memory_id)
                if memory is not None:
                    if not include_compressed and isinstance(memory, CompressedMemory):
                        continue
                    results.append(memory)

            return results

    def recall_associated(
        self,
        seed_ids: list[str],
        limit: int = 10,
        *,
        decay: float | None = None,
        max_depth: int | None = None,
        threshold: float | None = None,
        include_compressed: bool = True,
    ) -> list[tuple[EpisodicMemory | CompressedMemory, float]]:
        """Retrieve memories via spreading activation through the associative graph.

        Starting from one or more seed memories, follows associative edges
        formed during capture() to find related memories that may not share
        direct perceptual features but are linked through chains of recall.

        This enables context-bridging recall: a memory about finding a cup
        (linked when the coffee-making memory was formed) becomes reachable
        from a "make coffee" query, even though "cup" and "coffee" share
        no direct index keys.

        Args:
            seed_ids: Memory IDs to start activation from.
            limit: Maximum memories to return.
            decay: Activation decay per hop (default: config value).
            max_depth: Maximum hops from seed (default: config value).
            threshold: Minimum activation to include (default: config value).
            include_compressed: Include CompressedMemory results.

        Returns:
            List of (memory, activation_score) tuples, highest activation first.
            Seed memories are excluded from results.
        """
        decay = decay if decay is not None else self.config.spreading_activation_decay
        max_depth = max_depth if max_depth is not None else self.config.spreading_activation_max_depth
        threshold = threshold if threshold is not None else self.config.spreading_activation_threshold

        with self._rwlock.read():
            # Run spreading activation on the graph
            activations = self._graph.spreading_activation(
                seed_ids,
                initial_activation=1.0,
                decay=decay,
                threshold=threshold,
                max_depth=max_depth,
            )

            # Collect results, excluding seeds
            seed_set = set(seed_ids)
            results: list[tuple[EpisodicMemory | CompressedMemory, float]] = []

            for memory_id, activation in activations.items():
                if memory_id in seed_set:
                    continue

                memory = self._memories.get(memory_id)
                if memory is None:
                    continue

                if not include_compressed and isinstance(memory, CompressedMemory):
                    continue

                results.append((memory, activation))

            # Sort by activation score descending
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]

    def get_associated_ids(
        self,
        memory_id: str,
    ) -> list[tuple[str, float]]:
        """Get directly associated memory IDs and weights for a memory.

        Returns one-hop neighbors only (no spreading activation).

        Args:
            memory_id: The memory to query.

        Returns:
            List of (neighbor_memory_id, edge_weight) tuples.
        """
        with self._rwlock.read():
            return self._graph.get_associated(memory_id)

    def get_edge_count(self, memory_id: str) -> int:
        """Get the number of associative edges for a memory.

        Args:
            memory_id: The memory to query.

        Returns:
            Number of outgoing association edges.
        """
        with self._rwlock.read():
            return len(self._graph.get_associated(memory_id))

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
            return [
                self._memories[mid]
                for mid in memory_ids
                if mid in self._memories
            ]

    def index_keys(self) -> list[str]:
        """Return all index keys."""
        with self._rwlock.read():
            return list(self._context_index.keys())

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> None:
        """Save hippocampus to JSON file.

        Args:
            path: File path. Uses config.persistence_path if None.
        """
        path = path or self.config.persistence_path
        if not path:
            raise ValueError("No persistence path specified")

        with self._rwlock.read():
            # Serialize memories (handles both EpisodicMemory and CompressedMemory)
            memories_data = [m.to_dict() for m in self._memories.values()]

            # Serialize index (convert sets to lists)
            index_data = {k: list(v) for k, v in self._context_index.items()}

            # Serialize associative graph edges
            graph_data = self._graph.to_dict()

            payload = {
                "version": "3.0",  # Updated for associative graph support
                "saved_at": time.time(),
                "memories": memories_data,
                "context_index": index_data,
                "stats": dict(self._stats),
                "compressed_count": self._compressed_count,
                "associative_graph": graph_data,
            }

        # Write atomically
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, path)

        logger.info("Saved hippocampus to %s (%d memories)", path, len(memories_data))

    def load(self, path: str | None = None) -> None:
        """Load hippocampus from JSON file.

        Args:
            path: File path. Uses config.persistence_path if None.
        """
        path = path or self.config.persistence_path
        if not path:
            raise ValueError("No persistence path specified")

        if not os.path.exists(path):
            logger.warning("Hippocampus file not found: %s", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Version check - support 1.0, 2.0, and 3.0
        version = payload.get("version", "0.0")
        if version not in ("1.0", "2.0", "3.0"):
            raise ValueError(f"Unsupported hippocampus version: {version}")

        with self._rwlock.write():
            # Clear existing data
            self._memories.clear()
            self._context_index.clear()
            self._memory_contexts.clear()
            self._compressed_count = 0

            # Load memories (detect compressed vs full)
            for m_data in payload.get("memories", []):
                if m_data.get("_compressed", False):
                    # CompressedMemory
                    memory = CompressedMemory.from_dict(m_data)
                    self._memories[memory.id] = memory
                    self._compressed_count += 1
                else:
                    # Full EpisodicMemory
                    memory = EpisodicMemory.from_dict(m_data)
                    self._memories[memory.id] = memory

            # Load index
            self._context_index = defaultdict(
                set,
                {k: set(v) for k, v in payload.get("context_index", {}).items()},
            )

            # Rebuild reverse index
            for index_key, memory_ids in self._context_index.items():
                for memory_id in memory_ids:
                    self._memory_contexts[memory_id].add(index_key)

            # Load stats
            self._stats = payload.get("stats", {})

            # Restore compressed count from payload if available
            if "compressed_count" in payload:
                self._compressed_count = payload["compressed_count"]

            # Restore associative graph (v3.0+)
            self._graph = DependencyGraph()
            graph_data = payload.get("associative_graph")
            if graph_data:
                self._restore_graph(graph_data)

        edge_count = sum(
            len(self._graph.get_associated(mid))
            for mid in self._memories
            if self._graph.get_node(mid) is not None
        )

        logger.info(
            "Loaded hippocampus from %s (%d memories, %d compressed, %d edges)",
            path,
            len(self._memories),
            self._compressed_count,
            edge_count,
        )

    def load_with_recovery(
        self,
        path: str | None = None,
        on_error: str = "warn_and_continue",
    ) -> tuple[bool, str | None]:
        """Load with automatic recovery on failure.

        Args:
            path: File path. Uses config.persistence_path if None.
            on_error: "warn_and_continue" | "restore_backup" | "raise"

        Returns:
            (success, error_message) - error_message is None on success.
        """
        path = path or self.config.persistence_path
        if not path:
            return False, "No persistence path specified"

        if not os.path.exists(path):
            logger.info("No existing hippocampus file, starting fresh")
            return True, None

        try:
            self.load(path)
            return True, None

        except json.JSONDecodeError as e:
            error_msg = f"Corrupt hippocampus file: {e}"
            logger.warning(error_msg)

            if on_error == "restore_backup":
                backup_path = f"{path}.backup"
                if os.path.exists(backup_path):
                    try:
                        self.load(backup_path)
                        logger.info("Restored from backup: %s", backup_path)
                        return True, "Restored from backup (original corrupt)"
                    except Exception as be:
                        logger.error("Backup also corrupt: %s", be)

            if on_error == "raise":
                raise

            # warn_and_continue: start fresh
            logger.warning("Starting with empty hippocampus due to corrupt file")
            return True, error_msg

        except Exception as e:
            error_msg = f"Failed to load hippocampus: {e}"
            logger.error(error_msg)

            if on_error == "raise":
                raise

            return True, error_msg

    def save_with_backup(self, path: str | None = None) -> None:
        """Save with automatic backup of previous version.

        Args:
            path: File path. Uses config.persistence_path if None.
        """
        import shutil

        path = path or self.config.persistence_path
        if not path:
            raise ValueError("No persistence path specified")

        # Create backup of existing file
        if os.path.exists(path):
            backup_path = f"{path}.backup"
            try:
                shutil.copy2(path, backup_path)
            except Exception as e:
                logger.warning("Failed to create backup: %s", e)

        # Save normally
        self.save(path)

    # ─────────────────────────────────────────────────────────────────────────
    # Statistics and Debugging
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return hippocampus statistics."""
        with self._rwlock.read():
            # Count long-term memories
            long_term_count = sum(
                1 for m in self._memories.values() if getattr(m, "long_term", False)
            )

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
                valid_ids = {
                    mid for mid in self._context_index[index_key]
                    if mid in self._memories
                }
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

    def _restore_graph(self, graph_data: dict[str, Any]) -> None:
        """Restore the associative graph from serialized data (lock must be held).

        Reconstructs nodes and edges from the graph's to_dict() output.

        Args:
            graph_data: Serialized graph from DependencyGraph.to_dict().
        """
        # Restore nodes - only for memories that still exist
        for node_id in graph_data.get("nodes", []):
            if node_id in self._memories:
                self._graph.add_node(node_id, node_id)

        # Restore edges (only ASSOCIATES edges, skip duplicates from bidirectional)
        seen_pairs: set[tuple[str, str]] = set()
        for edge_data in graph_data.get("edges", []):
            source = edge_data.get("source", "")
            target = edge_data.get("target", "")
            weight = edge_data.get("weight", 1.0)
            edge_type_name = edge_data.get("type", "ASSOCIATES")

            # Only restore association edges
            if edge_type_name != "ASSOCIATES":
                continue

            # Skip if either memory no longer exists
            if source not in self._memories or target not in self._memories:
                continue

            # Avoid duplicating bidirectional edges
            pair = (min(source, target), max(source, target))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Ensure nodes exist
            if self._graph.get_node(source) is None:
                self._graph.add_node(source, source)
            if self._graph.get_node(target) is None:
                self._graph.add_node(target, target)

            self._graph.add_bidirectional(source, target, EdgeType.ASSOCIATES, weight)

    def _form_associations(
        self,
        memory_id: str,
        memory: EpisodicMemory,
    ) -> None:
        """Form bidirectional edges between a new memory and recalled similar memories.

        Called during capture() while write lock is held. This is the core
        mechanism for building the associative graph: when a new memory is
        stored, we find similar existing memories and link them, mimicking
        how biological hippocampal replay strengthens synaptic connections.

        Args:
            memory_id: ID of the newly captured memory.
            memory: The EpisodicMemory being captured.
        """
        # Recall similar memories (lock already held)
        similar = self._recall_similar_unlocked(
            memory.perception,
            limit=self.config.association_limit,
            exclude_id=memory_id,
        )

        if not similar:
            return

        # Register the new memory as a node in the graph
        self._graph.add_node(memory_id, memory_id)

        edges_formed = 0
        for recalled_mem, score in similar:
            if score < self.config.association_threshold:
                continue

            # Compute edge weight
            weight = self._compute_association_weight(memory, recalled_mem, score)

            # Ensure the recalled memory is a node in the graph
            if self._graph.get_node(recalled_mem.id) is None:
                self._graph.add_node(recalled_mem.id, recalled_mem.id)

            # Form bidirectional edge (new ↔ recalled)
            self._graph.add_bidirectional(
                memory_id, recalled_mem.id, EdgeType.ASSOCIATES, weight
            )

            # Touch the recalled memory (strengthens its retention)
            self._touch_internal(recalled_mem.id)

            edges_formed += 1

        if edges_formed > 0:
            self._stats["edges_formed"] = (
                self._stats.get("edges_formed", 0) + edges_formed
            )
            logger.debug(
                "Formed %d associative edges for memory %s",
                edges_formed,
                memory_id[:8],
            )

    def _recall_similar_unlocked(
        self,
        perception: Perception,
        limit: int = 5,
        exclude_id: str | None = None,
    ) -> list[tuple[EpisodicMemory | CompressedMemory, float]]:
        """Find similar memories with scores (lock must already be held).

        Returns (memory, score) tuples sorted by score descending.
        Used internally by capture() to form associative edges without
        re-acquiring the write lock.

        Args:
            perception: Current perception to match against.
            limit: Maximum memories to return.
            exclude_id: Memory ID to exclude (the one being captured).

        Returns:
            List of (memory, similarity_score) tuples.
        """
        scores: dict[str, float] = defaultdict(float)

        # Score by overlapping detected objects
        for obj in perception.detected_objects:
            index_key = f"object:{obj}"
            for memory_id in self._context_index.get(index_key, set()):
                if memory_id != exclude_id:
                    scores[memory_id] += 1.0

        # Score by overlapping detected people
        for person in perception.detected_people:
            index_key = f"person:{person}"
            for memory_id in self._context_index.get(index_key, set()):
                if memory_id != exclude_id:
                    scores[memory_id] += 1.0

        if not scores:
            return []

        # Normalize scores to 0-1 range
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            normalized = {mid: s / max_score for mid, s in scores.items()}
        else:
            normalized = scores

        # Sort by score, then by timestamp for tiebreaking
        def sort_key(mid: str) -> tuple[float, float]:
            mem = self._memories.get(mid)
            ts = mem.timestamp if mem else 0.0
            return (normalized[mid], ts)

        sorted_ids = sorted(normalized.keys(), key=sort_key, reverse=True)

        results: list[tuple[EpisodicMemory | CompressedMemory, float]] = []
        for memory_id in sorted_ids[:limit]:
            memory = self._memories.get(memory_id)
            if memory is not None:
                results.append((memory, normalized[memory_id]))

        return results

    def _compute_association_weight(
        self,
        new_memory: EpisodicMemory,
        recalled_memory: EpisodicMemory | CompressedMemory,
        similarity_score: float,
    ) -> float:
        """Compute edge weight for an associative connection.

        Combines perceptual similarity with goal and temporal proximity.

        Args:
            new_memory: The memory being captured.
            recalled_memory: A memory recalled during capture.
            similarity_score: Raw similarity from _recall_similar_unlocked.

        Returns:
            Edge weight (0-1).
        """
        weight = similarity_score * 0.6  # Base: perceptual overlap

        # Boost for shared goal
        new_goal = new_memory.context.active_goal or ""
        if isinstance(recalled_memory, CompressedMemory):
            old_goal = recalled_memory.goal or ""
        else:
            old_goal = recalled_memory.context.active_goal or ""

        if new_goal and old_goal and new_goal == old_goal:
            weight += 0.25
        elif new_goal and old_goal:
            # Partial goal overlap via word intersection
            w1 = set(new_goal.lower().split())
            w2 = set(old_goal.lower().split())
            common = {"the", "a", "an", "to", "and", "or", "in", "on", "at"}
            w1 -= common
            w2 -= common
            if w1 and w2:
                overlap = len(w1 & w2) / len(w1 | w2)
                weight += 0.25 * overlap

        # Temporal proximity bonus (closer in time = stronger link)
        time_diff = abs(new_memory.timestamp - recalled_memory.timestamp)
        temporal_bonus = 0.15 / (1.0 + time_diff / 3600)  # Decays over hours
        weight += temporal_bonus

        return min(1.0, weight)

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

    def sleep(
        self,
        strategy: "MemoryStrategy | None" = None,
    ) -> dict[str, int]:
        """Perform sleep-like memory consolidation.

        This is the primary memory management process, inspired by biological
        sleep consolidation. It:
        1. Promotes important memories to long-term storage
        2. Compresses old, infrequently-accessed full memories
        3. Removes stale memories that haven't been accessed in too long

        Should be called periodically (e.g., during idle time, end of session).

        Args:
            strategy: Memory strategy to use. Defaults to config.memory_strategy.

        Returns:
            Dict with counts: {"compressed": N, "removed": M, "preserved": P, "promoted": Q}
        """

        with self._rwlock.write():
            results = self._sleep(strategy)

        # Log consolidation activity OUTSIDE rwlock (P3f — Tier 2)
        if hasattr(self, "_collector") and self._collector and \
           self._collector.verbosity >= 1:
            from maxim.provenance.types import PipelineStage
            self._collector.log_activity(
                PipelineStage.CONSOLIDATION, "hippocampus",
                f"Promoted {results['promoted']}, compressed {results['compressed']}, "
                f"removed {results['removed']}",
            )

        return results

    def _sleep(
        self,
        strategy: "MemoryStrategy | None" = None,
    ) -> dict[str, int]:
        """Internal sleep implementation (lock must be held)."""
        if not self.config.enable_sleep_consolidation:
            return {
                "compressed": 0,
                "removed": 0,
                "preserved": len(self._memories),
                "promoted": 0,
            }

        # Step 1: Consolidate important memories to long-term (before any removal)
        consolidation_results = {"promoted": 0}
        if self.config.consolidate_during_sleep:
            consolidation_results = self._consolidate()

        # Get or create strategy
        if strategy is None:
            strategy = self._get_memory_strategy()

        now = time.time()
        results = {
            "compressed": 0,
            "removed": 0,
            "preserved": 0,
            "promoted": consolidation_results.get("promoted", 0),
        }

        # Process each memory
        memories_to_remove: list[str] = []
        memories_to_compress: list[str] = []

        for memory_id, record in list(self._memories.items()):
            # Skip if already compressed
            if isinstance(record, CompressedMemory):
                # Still evaluate for removal (use stored edge_count)
                score = strategy.score_for_retention(record, now, record.edge_count)
                # Apply long-term boost
                if record.long_term:
                    score *= self.config.long_term_retention_boost
                    score = min(1.0, score)

                if score < self.config.retention_threshold:
                    memories_to_remove.append(memory_id)
                else:
                    results["preserved"] += 1
                continue

            # Full EpisodicMemory - evaluate
            edge_count = len(self._graph.get_associated(memory_id))
            score = strategy.score_for_retention(record, now, edge_count)
            # Apply long-term boost
            if record.long_term:
                score *= self.config.long_term_retention_boost
                score = min(1.0, score)

            if score < self.config.retention_threshold:
                # Remove entirely
                memories_to_remove.append(memory_id)
            elif score < self.config.compression_threshold:
                # Candidate for compression
                if strategy.should_compress(record, now, 0):
                    memories_to_compress.append(memory_id)
                else:
                    results["preserved"] += 1
            else:
                # Keep as full record
                results["preserved"] += 1

        # Perform compression
        for memory_id in memories_to_compress:
            self._compress_memory(memory_id)
            results["compressed"] += 1

        # Perform removal
        for memory_id in memories_to_remove:
            self._remove_memory(memory_id)
            results["removed"] += 1

        # Update stats
        self._stats["consolidations"] = self._stats.get("consolidations", 0) + 1
        self._stats["compressions"] = self._stats.get("compressions", 0) + results["compressed"]
        self._stats["removals"] = self._stats.get("removals", 0) + results["removed"]

        # Auto-save if configured
        if self.config.auto_save_after_sleep and self.config.persistence_path:
            self.save_with_backup(self.config.persistence_path)

        logger.info(
            "Sleep consolidation complete: compressed=%d, removed=%d, preserved=%d, promoted=%d",
            results["compressed"],
            results["removed"],
            results["preserved"],
            results["promoted"],
        )

        return results

    def _compress_memory(self, memory_id: str) -> None:
        """Compress a full EpisodicMemory to CompressedMemory (lock must be held)."""
        if memory_id not in self._memories:
            return

        record = self._memories[memory_id]
        if not isinstance(record, EpisodicMemory):
            return  # Already compressed or invalid

        # Create compressed version with actual edge count from the graph
        actual_edge_count = len(self._graph.get_associated(memory_id))
        compressed = CompressedMemory.from_episodic(record, edge_count=actual_edge_count)

        # Replace in storage
        self._memories[memory_id] = compressed  # type: ignore[assignment]
        self._compressed_count += 1

        logger.debug("Compressed memory %s", memory_id[:8])

        # Notify subsystems of compression (e.g., ConceptExtractor ref cleanup)
        for callback in self._on_memory_compressed:
            try:
                callback(memory_id)
            except Exception as e:
                logger.warning("Memory compression callback failed: %s", e)

    # ─────────────────────────────────────────────────────────────────────────
    # Long-Term Memory Consolidation
    # ─────────────────────────────────────────────────────────────────────────

    def consolidate(
        self,
        *,
        force_ids: list[str] | None = None,
        max_promotions: int = 100,
    ) -> dict[str, int]:
        """Consolidate important short-term memories into long-term storage.

        Long-term memories are resistant (but not immune) to removal during
        sleep consolidation. This mimics biological memory consolidation
        where important experiences get "burned in" over time.

        Can be called:
        - Automatically during sleep() (default behavior)
        - Manually to force promotion of specific memories
        - Periodically during idle time

        Args:
            force_ids: Specific memory IDs to promote (bypasses threshold check).
            max_promotions: Maximum memories to promote in one call.

        Returns:
            Dict with counts: {"promoted": N, "already_long_term": M, "skipped": P}
        """
        with self._rwlock.write():
            return self._consolidate(force_ids=force_ids, max_promotions=max_promotions)

    def _consolidate(
        self,
        *,
        force_ids: list[str] | None = None,
        max_promotions: int = 100,
    ) -> dict[str, int]:
        """Internal consolidation (lock must be held)."""
        now = time.time()
        results = {"promoted": 0, "already_long_term": 0, "skipped": 0}

        # Process forced promotions first
        if force_ids:
            for memory_id in force_ids[:max_promotions]:
                promoted = self._promote_to_long_term(memory_id, now)
                if promoted:
                    results["promoted"] += 1
                else:
                    results["already_long_term"] += 1
            return results

        # Evaluate candidates from the queue
        candidates_to_evaluate = list(self._consolidation_candidates)
        self._consolidation_candidates.clear()

        for memory_id in candidates_to_evaluate:
            if results["promoted"] >= max_promotions:
                # Re-queue remaining candidates for next consolidation
                self._consolidation_candidates.append(memory_id)
                continue

            if memory_id not in self._memories:
                continue

            record = self._memories[memory_id]
            if getattr(record, "long_term", False):
                results["already_long_term"] += 1
                continue

            # Check promotion criteria
            if self._should_promote(record, now):
                if self._promote_to_long_term(memory_id, now):
                    results["promoted"] += 1
                else:
                    results["already_long_term"] += 1
            else:
                results["skipped"] += 1

        return results

    def _should_promote(
        self,
        record: EpisodicMemory | CompressedMemory,
        now: float,
    ) -> bool:
        """Decide if a memory should be promoted to long-term.

        Promotion criteria (any of these triggers promotion):
        1. Very high salience (> 0.9)
        2. Very high novelty (> 0.9)
        3. User interaction with successful outcome
        4. High access count (> 5)
        """
        # Extract fields (works for both types)
        if isinstance(record, CompressedMemory):
            novelty = record.novelty
            salience = record.salience
            had_user_input = record.had_user_input
            success = record.success
            access_count = record.access_count
        else:
            novelty = record.perception.novelty
            salience = record.perception.salience
            had_user_input = bool(record.perception.cli_input or record.perception.transcript)
            success = record.outcome.success
            access_count = record.access_count

        # Criterion 1: Very high salience
        if salience > 0.9:
            return True

        # Criterion 2: Very high novelty
        if novelty > 0.9:
            return True

        # Criterion 3: Successful user interaction
        if had_user_input and success:
            return True

        # Criterion 4: Frequently accessed
        if access_count >= 5:
            return True

        return False

    def _promote_to_long_term(self, memory_id: str, now: float) -> bool:
        """Promote a memory to long-term status (lock must be held)."""
        if memory_id not in self._memories:
            return False

        record = self._memories[memory_id]
        if record is None:
            return False

        if getattr(record, "long_term", False):
            return False  # Already long-term

        # Update the record
        record.long_term = True
        record.consolidated_at = now

        self._stats["long_term_count"] = self._stats.get("long_term_count", 0) + 1
        logger.debug("Promoted memory %s to long-term", memory_id[:8])
        return True

    def _add_consolidation_candidate(self, memory_id: str) -> None:
        """Add a memory to the consolidation candidate queue (lock must be held).

        Called during capture() for memories that might be promotable but
        don't meet immediate promotion criteria.
        """
        # Bounded queue to prevent unbounded growth
        if len(self._consolidation_candidates) < self.config.max_consolidation_candidates:
            self._consolidation_candidates.append(memory_id)

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

    def sleep_with_clustering(
        self,
        max_per_cluster: int = 3,
    ) -> dict[str, int]:
        """Perform temporal cluster-based sleep consolidation.

        This is the most efficient consolidation method when SCN is connected.
        Instead of evaluating each memory individually, it groups memories by
        temporal cluster (hour, day) and keeps only the best representatives.

        This can reduce 10K memories to ~500 while maintaining full temporal
        coverage.

        Args:
            max_per_cluster: Maximum memories to keep per (hour, day) cluster.
                Default is 3 (best, most accessed, most recent).

        Returns:
            Dict with counts including "clusters_processed".

        Raises:
            RuntimeError: If SCN is not connected.
        """
        if self._scn is None:
            raise RuntimeError("SCN must be connected for cluster-based consolidation")

        with self._rwlock.write():
            return self._sleep_with_clustering(max_per_cluster)

    def _sleep_with_clustering(
        self,
        max_per_cluster: int = 3,
    ) -> dict[str, int]:
        """Internal clustering implementation (lock must be held)."""
        from maxim.memory.strategies import TemporalAwareStrategy

        if self._scn is None:
            return self._sleep()

        now = time.time()
        results = {
            "compressed": 0,
            "removed": 0,
            "preserved": 0,
            "promoted": 0,
            "clusters_processed": 0,
        }

        # Step 1: Consolidate important memories first
        if self.config.consolidate_during_sleep:
            consolidation_results = self._consolidate()
            results["promoted"] = consolidation_results.get("promoted", 0)

        # Step 2: Get all temporal clusters from SCN
        clusters = self._scn.get_all_clusters()
        results["clusters_processed"] = len(clusters)

        # Step 3: Build strategy for selecting representatives
        base_strategy = self._get_memory_strategy()
        if isinstance(base_strategy, TemporalAwareStrategy):
            strategy = base_strategy
        else:
            strategy = TemporalAwareStrategy(
                scn=self._scn,
                base_strategy=base_strategy,
            )
            strategy.prepare()

        memories_to_remove: list[str] = []
        memories_to_compress: list[str] = []

        # Step 4: Process each cluster
        for (hour, day), cluster_ids in clusters.items():
            if len(cluster_ids) <= max_per_cluster:
                # Small cluster - keep all, evaluate individually
                for memory_id in cluster_ids:
                    record = self._memories.get(memory_id)
                    if record is None:
                        continue

                    score = strategy.score_for_retention(record, now, 0)
                    if record.long_term:
                        score *= self.config.long_term_retention_boost
                        score = min(1.0, score)

                    if score < self.config.retention_threshold:
                        memories_to_remove.append(memory_id)
                    elif score < self.config.compression_threshold:
                        if isinstance(record, EpisodicMemory):
                            if strategy.should_compress(record, now, 0):
                                memories_to_compress.append(memory_id)
                            else:
                                results["preserved"] += 1
                        else:
                            results["preserved"] += 1
                    else:
                        results["preserved"] += 1
                continue

            # Large cluster - select best representatives
            # Score all memories in cluster
            scored: list[tuple[str, float]] = []
            for memory_id in cluster_ids:
                record = self._memories.get(memory_id)
                if record is None:
                    continue

                score = strategy.score_for_retention(record, now, 0)
                if record.long_term:
                    score *= self.config.long_term_retention_boost
                    score = min(1.0, score)

                scored.append((memory_id, score))

            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)

            # Keep top max_per_cluster, compress/remove the rest
            for i, (memory_id, score) in enumerate(scored):
                record = self._memories.get(memory_id)
                if record is None:
                    continue

                if i < max_per_cluster:
                    # Keep as representative
                    if score < self.config.compression_threshold:
                        if isinstance(record, EpisodicMemory):
                            memories_to_compress.append(memory_id)
                        else:
                            results["preserved"] += 1
                    else:
                        results["preserved"] += 1
                else:
                    # Not a top representative - remove if low score, else compress
                    if score < self.config.retention_threshold:
                        memories_to_remove.append(memory_id)
                    elif isinstance(record, EpisodicMemory):
                        memories_to_compress.append(memory_id)
                    else:
                        # Already compressed, remove
                        memories_to_remove.append(memory_id)

        # Step 5: Handle memories not in any cluster (no temporal signature)
        clustered_ids = set()
        for cluster_ids in clusters.values():
            clustered_ids.update(cluster_ids)

        for memory_id, record in self._memories.items():
            if memory_id in clustered_ids:
                continue

            # No temporal data - use standard evaluation
            score = strategy.score_for_retention(record, now, 0)
            if record.long_term:
                score *= self.config.long_term_retention_boost
                score = min(1.0, score)

            if score < self.config.retention_threshold:
                memories_to_remove.append(memory_id)
            elif score < self.config.compression_threshold:
                if isinstance(record, EpisodicMemory):
                    if strategy.should_compress(record, now, 0):
                        memories_to_compress.append(memory_id)
                    else:
                        results["preserved"] += 1
                else:
                    results["preserved"] += 1
            else:
                results["preserved"] += 1

        # Step 6: Execute compression and removal
        for memory_id in memories_to_compress:
            self._compress_memory(memory_id)
            results["compressed"] += 1

        for memory_id in memories_to_remove:
            self._remove_memory(memory_id)
            results["removed"] += 1

        # Update stats
        self._stats["consolidations"] = self._stats.get("consolidations", 0) + 1
        self._stats["compressions"] = self._stats.get("compressions", 0) + results["compressed"]
        self._stats["removals"] = self._stats.get("removals", 0) + results["removed"]

        # Auto-save if configured
        if self.config.auto_save_after_sleep and self.config.persistence_path:
            self.save_with_backup(self.config.persistence_path)

        logger.info(
            "Cluster consolidation complete: clusters=%d, compressed=%d, removed=%d, preserved=%d",
            results["clusters_processed"],
            results["compressed"],
            results["removed"],
            results["preserved"],
        )

        return results


__all__ = ["Hippocampus", "HippocampusConfig"]
