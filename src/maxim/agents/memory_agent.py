"""MemoryAgent: Core memory system with salience, associations, and persistence.

THIS IS THE CENTRAL RESEARCH FOCUS OF THE PROJECT.

The MemoryAgent maintains salient memories using WorkingMemoryEntry wrappers
around structured MemoryRecord subclasses, builds associations via similarity,
and provides structured context for goal proposal.

Memory lifecycle: FORMING → WORKING → SHORT_TERM → LONG_TERM → consolidated out.
FORMING entries are created at percept time and filled incrementally through
the pipeline (Decision → Action → Outcome). Pattern completion can attach
predicted outcomes during FORMING.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable

from maxim.agents.base import Agent
from maxim.agents.bus import (
    AgentBus,
    DependencyGraph,
    EdgeType,
    GoalAccepted,
    GoalCompleted,
    MemoryItem,
    MemoryTier,
    Percept,
    ProposedGoal,
    StatisticalSummary,
    StructuredContext,
    ToolResult,
    WorkingMemoryEntry,
)
from maxim.agents.output_mixin import AgentOutputMixin
from maxim.memory.types import (
    Action,
    Context,
    Decision,
    EpisodicMemory,
    MathContextEntry,
    Outcome,
    Perception,
    PredictedOutcome,
)
from maxim.utils.logging import log_swallowed_exception
from maxim.utils.structured_logging import get_abstraction_buffer


class AssociationIndex:
    """
    Index for fast similarity-based memory retrieval.

    Two-tier lookup:
    1. Keyword overlap (fast, coarse) - always available
    2. Embedding similarity (slow, precise) - optional, lazy-computed
    """

    STOPWORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "to",
            "of",
            "and",
            "or",
            "in",
            "on",
            "at",
            "for",
            "it",
            "this",
            "that",
            "with",
        }
    )

    def __init__(self, embedding_model: str | None = None) -> None:
        self._keyword_index: dict[str, set[str]] = defaultdict(set)
        self._entries: dict[str, WorkingMemoryEntry] = {}
        self._embedding_model = embedding_model
        self._embedder = None  # Lazy init
        self._lock = threading.Lock()

    def add(self, entry: WorkingMemoryEntry) -> None:
        """Add entry to index."""
        mid = entry.id
        with self._lock:
            self._entries[mid] = entry

            # Use keywords from the underlying record
            keywords = entry.keywords
            for kw in keywords:
                self._keyword_index[kw].add(mid)

    def remove(self, entry_id: str) -> None:
        """Remove entry from index."""
        with self._lock:
            if entry_id not in self._entries:
                return
            entry = self._entries[entry_id]
            for kw in entry.keywords:
                self._keyword_index[kw].discard(entry_id)
            del self._entries[entry_id]

    def find_similar(
        self,
        query: str | WorkingMemoryEntry,
        top_k: int = 5,
        use_embeddings: bool = False,
    ) -> list[tuple[WorkingMemoryEntry, float]]:
        """Find entries similar to query."""
        with self._lock:
            if isinstance(query, WorkingMemoryEntry):
                query_keywords = query.keywords
                query_text = str(query.record.to_context_dict())
            else:
                query_keywords = self._extract_keywords(query)
                query_text = query

            # Stage 1: Keyword overlap (Jaccard similarity)
            candidates: dict[str, float] = {}
            for kw in query_keywords:
                for mid in self._keyword_index.get(kw, set()):
                    if mid not in candidates:
                        candidates[mid] = 0.0
                    entry = self._entries.get(mid)
                    if entry:
                        intersection = len(query_keywords & entry.keywords)
                        union = len(query_keywords | entry.keywords)
                        if union > 0:
                            candidates[mid] = max(candidates[mid], intersection / union)

            # Stage 2: Optional embedding refinement
            if use_embeddings and self._embedding_model and candidates:
                candidates = self._refine_with_embeddings(query_text, candidates)

            # Sort and return top_k
            sorted_results = sorted(
                [
                    (self._entries[mid], score)
                    for mid, score in candidates.items()
                    if mid in self._entries
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            return sorted_results[:top_k]

    def build_associations(self, entry: WorkingMemoryEntry, threshold: float = 0.3) -> None:
        """Automatically build associations based on similarity."""
        similar = self.find_similar(entry, top_k=5)
        # Associations are now handled at the graph level, not on the entry itself

    def _extract_keywords(self, content: Any) -> set[str]:
        """Extract keywords from raw content for indexing."""
        if isinstance(content, str):
            text = content
        elif isinstance(content, dict):
            text = " ".join(str(v) for v in content.values() if v)
        elif hasattr(content, "raw_transcript_text"):
            text = getattr(content, "raw_transcript_text", "") or ""
        else:
            text = str(content)

        words = text.lower().split()
        return {w for w in words if len(w) > 2 and w not in self.STOPWORDS}

    def _refine_with_embeddings(
        self,
        query_text: str,
        candidates: dict[str, float],
    ) -> dict[str, float]:
        """Refine candidate scores using embedding similarity."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(
                    self._embedding_model or "all-MiniLM-L6-v2"
                )
            except ImportError:
                return candidates

        try:
            import numpy as np

            query_emb = self._embedder.encode(query_text, convert_to_numpy=True)

            for mid in candidates:
                entry = self._entries.get(mid)
                if not entry:
                    continue
                if entry._embedding is None:
                    entry_text = str(entry.record.to_context_dict())
                    entry._embedding = self._embedder.encode(
                        entry_text, convert_to_numpy=True
                    ).tolist()

                mem_emb = np.array(entry._embedding)
                similarity = np.dot(query_emb, mem_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(mem_emb) + 1e-8
                )
                candidates[mid] = 0.3 * candidates[mid] + 0.7 * float(similarity)
        except Exception as e:
            log_swallowed_exception(e, operation="memory_similarity_boost")

        return candidates


class MemoryAssociationGraph:
    """Graph-based memory associations for spreading activation."""

    def __init__(self) -> None:
        self._graph: DependencyGraph[WorkingMemoryEntry] = DependencyGraph()
        self._temporal_index: dict[int, list[str]] = defaultdict(list)

    def add_memory(self, entry: WorkingMemoryEntry) -> None:
        """Add entry to the association graph."""
        self._graph.add_node(entry.id, entry)

        hour_bucket = int(entry.timestamp // 3600)
        self._temporal_index[hour_bucket].append(entry.id)

        self._build_temporal_associations(entry)

    def associate(
        self,
        entry_a: WorkingMemoryEntry,
        entry_b: WorkingMemoryEntry,
        weight: float = 1.0,
        edge_type: EdgeType = EdgeType.ASSOCIATES,
    ) -> None:
        """Create bidirectional association."""
        self._graph.add_bidirectional(
            entry_a.id, entry_b.id, edge_type, weight
        )

    def add_causal_link(
        self,
        cause: WorkingMemoryEntry,
        effect: WorkingMemoryEntry,
        weight: float = 1.0,
    ) -> None:
        """Record causal relationship."""
        self._graph.add_edge(
            cause.id, effect.id, EdgeType.CAUSES, weight
        )

    def _build_temporal_associations(
        self,
        entry: WorkingMemoryEntry,
        window_hours: int = 1,
        max_associations: int = 5,
    ) -> None:
        """Associate with entries close in time."""
        hour_bucket = int(entry.timestamp // 3600)

        nearby_ids: list[str] = []
        for offset in range(-window_hours, window_hours + 1):
            nearby_ids.extend(self._temporal_index.get(hour_bucket + offset, []))

        nearby_ids = [
            mid
            for mid in nearby_ids
            if mid != entry.id and self._graph.get_node(mid) is not None
        ]

        for mid in nearby_ids[:max_associations]:
            other = self._graph.get_node(mid)
            if other:
                time_diff = abs(entry.timestamp - other.timestamp)
                weight = 1.0 / (1.0 + time_diff / 3600)
                self._graph.add_bidirectional(
                    entry.id, mid, EdgeType.ASSOCIATES, weight
                )

    def get_related_memories(
        self,
        query_entries: list[WorkingMemoryEntry],
        top_k: int = 10,
        activation_decay: float = 0.5,
    ) -> list[tuple[WorkingMemoryEntry, float]]:
        """Get related entries via spreading activation."""
        source_ids = [e.id for e in query_entries]

        activations = self._graph.spreading_activation(
            source_ids,
            initial_activation=1.0,
            decay=activation_decay,
            threshold=0.05,
            max_depth=4,
        )

        results: list[tuple[WorkingMemoryEntry, float]] = []
        for mid, activation in activations.items():
            if mid not in source_ids:
                entry = self._graph.get_node(mid)
                if entry:
                    results.append((entry, activation))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class MemoryAgent(Agent, AgentOutputMixin):
    """
    Maintains salient memory and builds structured context.

    THIS IS THE CORE EXPLORATION FOCUS OF THE PROJECT.

    Responsibilities:
    - Preserve salient moments using WorkingMemoryEntry wrappers
    - Staged memory formation (FORMING → WORKING → SHORT_TERM → LONG_TERM)
    - Pattern completion during FORMING via optional hook
    - Recall similar states to enrich StructuredContext
    - Apply salience decay over time
    - Track goal outcomes for learning
    - Build StructuredContext for ExecAgent
    - Persist memories across sessions (v2.0 format)
    - Output to sandbox and shared directories via AgentOutputMixin
    """

    agent_name = "memory"

    ROOT_GOAL = "Understand reality and help people."

    DEFAULT_LONG_TERM_MAX_AGE = 7 * 24 * 60 * 60  # 7 days

    def __init__(
        self,
        bus: AgentBus,
        *,
        name: str | None = None,
        enabled: bool = True,
        max_short_term: int = 100,
        max_long_term: int = 500,
        salience_threshold: float = 0.2,
        decay_interval: float = 1.0,
        context_window: int = 10,
        persistence_path: str | None = None,
        long_term_max_age: float | None = None,
        reset_on_startup: bool = False,
        enable_embeddings: bool = False,
        output_manager: Any | None = None,
        memory_hub: Any | None = None,
        plan_manager: Any | None = None,
        workspace_path: str | None = None,
    ) -> None:
        super().__init__(name=name, enabled=enabled)
        self._bus = bus
        self._plan_manager = plan_manager

        # Initialize output mixin for sandbox/shared output support
        self._init_output(agent_name="memory", output_manager=output_manager)
        self._max_short_term = max_short_term
        self._max_long_term = max_long_term
        self._salience_threshold = salience_threshold
        self._decay_interval = decay_interval
        self._context_window = context_window
        self._persistence_path = persistence_path
        self._long_term_max_age = long_term_max_age or self.DEFAULT_LONG_TERM_MAX_AGE
        self._reset_on_startup = reset_on_startup

        # Three-pool memory storage (single-track ownership)
        self._forming_pool: dict[str, WorkingMemoryEntry] = {}  # keyed by run_id
        self._short_term: deque[WorkingMemoryEntry] = deque(maxlen=max_short_term)
        self._long_term: list[WorkingMemoryEntry] = []
        self._recent_percepts: deque[Percept] = deque(maxlen=context_window)
        self._recent_outcomes: deque[dict] = deque(maxlen=context_window)
        self._cli_inputs: deque[str] = deque(maxlen=20)
        self._comms_messages: deque[dict] = deque(maxlen=20)

        # Association systems
        self._association_index = AssociationIndex(
            embedding_model="all-MiniLM-L6-v2" if enable_embeddings else None
        )
        self._association_graph = MemoryAssociationGraph()

        # Abstraction stream
        self._abstraction = get_abstraction_buffer()

        # State
        self._active_goal: str | None = None
        self._active_goal_description: str | None = None
        self._active_sub_goals: list[str] = []
        self._mode: str = "observe"
        self._last_decay_time: float = time.time()
        self._last_promotion_time: float = time.time()
        self._did_startup: bool = False
        self._lock = threading.Lock()

        # Pattern completion hook (set by ATL/MemoryHub wiring)
        self._pattern_completion_fn: Callable[[EpisodicMemory], list[PredictedOutcome]] | None = None

        # Hippocampus reference (for piggybacking in get_concept_context)
        self._hippocampus: Any | None = None

        # Load persisted memories
        if not self._reset_on_startup and self._persistence_path:
            self._load_memories_from_disk()

        # Statistical context (from StatisticianAgent via bus)
        self._latest_statistical_summary: str = ""
        self._active_pattern_count: int = 0
        self._statistical_suggestions: list[dict] = []

        # Optional MemoryHub for multi-layer knowledge queries
        self._memory_hub = memory_hub

        # Workspace path for reading working notes
        self._workspace_path = workspace_path

        # Subscribe to messages
        self._bus.subscribe(Percept, self._on_percept)
        self._bus.subscribe(ToolResult, self._on_tool_result)
        self._bus.subscribe(GoalCompleted, self._on_goal_completed)
        self._bus.subscribe(ProposedGoal, self._on_goal_proposed)
        self._bus.subscribe(GoalAccepted, self._on_goal_accepted)
        self._bus.subscribe(StatisticalSummary, self._on_statistical_summary)

    # ── Wiring ────────────────────────────────────────────────────────────

    def connect_hippocampus(self, hippocampus: Any) -> None:
        """Wire Hippocampus reference for piggybacking in get_concept_context."""
        self._hippocampus = hippocampus

    def set_pattern_completion_fn(
        self, fn: Callable[[EpisodicMemory], list[PredictedOutcome]]
    ) -> None:
        """Wire pattern completion hook (implemented in ATL concept plan)."""
        self._pattern_completion_fn = fn

    # ── Staged Memory Formation ───────────────────────────────────────────

    def _begin_memory_formation(
        self, percept: Percept, run_id: str
    ) -> WorkingMemoryEntry:
        """Create a FORMING EpisodicMemory from percept + current agentic state."""
        now = time.time()

        # Sweep WORKING entries from pool → SHORT_TERM
        self._flush_working_to_short_term()

        # Build Context from current agentic state
        context = Context(
            active_goal=self._active_goal_description,
            active_mode=self._mode,
            fear_level=0.0,
        )

        episodic = EpisodicMemory(
            id=f"wm-{now:.0f}-{run_id[:8]}",
            timestamp=now,
            run_id=run_id,
            perception=Perception(
                detected_objects=list(
                    det.get("label", det.get("class_name", ""))
                    for det in (percept.detections or [])
                    if det.get("label") or det.get("class_name")
                ),
                detected_people=[
                    det.get("label", "person")
                    for det in (percept.detections or [])
                    if det.get("class_id") == 0
                ],
                salience=percept.salience,
                novelty=percept.novelty,
                cli_input=percept.cli_input or percept.raw_transcript_text,
                observations=percept.metadata or {},
            ),
            context=context,
            # Decision, Action, Outcome left as defaults (empty)
        )

        entry = WorkingMemoryEntry(
            record=episodic,
            salience=self._compute_memory_salience(percept),
            source="percept",
            tier=MemoryTier.FORMING,
            decay_rate=0.05 if percept.has_maxim_keyword else 0.1,
        )

        # Pattern completion hook
        if self._pattern_completion_fn:
            try:
                entry.predicted_outcomes = self._pattern_completion_fn(episodic)
                if entry.predicted_outcomes:
                    entry.prediction_confidence = self._compute_prediction_confidence(
                        entry.predicted_outcomes
                    )
            except Exception as e:
                log_swallowed_exception(e, operation="pattern_completion")

        self._forming_pool[run_id] = entry
        self._association_index.add(entry)
        self._association_graph.add_memory(entry)
        return entry

    def _update_forming_decision(self, run_id: str, decision: Decision) -> None:
        """Fill in the decision on a FORMING episodic memory."""
        entry = self._forming_pool.get(run_id)
        if entry is None or entry.tier != MemoryTier.FORMING:
            return
        assert isinstance(entry.record, EpisodicMemory)
        entry.record.decision = decision
        entry.invalidate_keywords()

    def _update_forming_action(self, run_id: str, action: Action) -> None:
        """Fill in the action on a FORMING episodic memory."""
        entry = self._forming_pool.get(run_id)
        if entry is None or entry.tier != MemoryTier.FORMING:
            return
        assert isinstance(entry.record, EpisodicMemory)
        entry.record.action = action
        entry.invalidate_keywords()

    def _complete_forming_memory(self, run_id: str, outcome: Outcome) -> None:
        """Fill in the outcome and transition FORMING → WORKING.

        Entry stays in _forming_pool as WORKING until next cycle sweeps it
        into _short_term via _flush_working_to_short_term().
        """
        entry = self._forming_pool.get(run_id)
        if entry is None or entry.tier != MemoryTier.FORMING:
            return
        assert isinstance(entry.record, EpisodicMemory)
        entry.record.outcome = outcome
        entry.tier = MemoryTier.WORKING
        entry.invalidate_keywords()

    def _flush_working_to_short_term(self) -> None:
        """Transition WORKING entries in the pool to SHORT_TERM, move to _short_term."""
        to_remove = []
        for run_id, entry in self._forming_pool.items():
            if entry.tier == MemoryTier.WORKING:
                entry.tier = MemoryTier.SHORT_TERM
                self._short_term.appendleft(entry)
                to_remove.append(run_id)
        for run_id in to_remove:
            del self._forming_pool[run_id]

    def _compute_prediction_confidence(self, predictions: list[PredictedOutcome]) -> float:
        """Compute confidence from success rate, action consistency, and sample size."""
        if not predictions:
            return 0.0

        n = len(predictions)

        # Success rate
        successes = sum(1 for p in predictions if p.success)
        success_rate = successes / n

        # Action consistency: what fraction used the most common action?
        action_counts: dict[str, int] = {}
        for p in predictions:
            action_counts[p.tool] = action_counts.get(p.tool, 0) + 1
        most_common_count = max(action_counts.values()) if action_counts else 0
        consistency = most_common_count / n

        # Sample size dampening: confidence is low until ~5 matching episodes
        sample_factor = min(n / 5.0, 1.0)

        return success_rate * consistency * sample_factor

    # ── Bus Callbacks ─────────────────────────────────────────────────────

    def _on_percept(self, percept: Percept) -> None:
        """Process incoming percept."""
        with self._lock:
            self._recent_percepts.append(percept)

            # Update mode
            if percept.maxim_runtime and isinstance(percept.maxim_runtime, dict):
                mode = percept.maxim_runtime.get("mode")
                if isinstance(mode, str):
                    self._mode = mode.strip().lower()

            # Track CLI inputs
            if percept.cli_input:
                self._cli_inputs.append(percept.cli_input)

            # Track comms messages (SMS, voice, etc.)
            if percept.source.startswith("comms:") and percept.content:
                self._comms_messages.append({
                    "direction": "inbound",
                    "content": percept.content,
                    "channel": (percept.metadata or {}).get("channel", ""),
                    "sender": (percept.metadata or {}).get("sender", ""),
                    "timestamp": percept.timestamp,
                })

            # Store as memory if salient — use staged formation
            if percept.salience > self._salience_threshold or percept.has_maxim_keyword:
                run_id = f"percept-{percept.timestamp:.0f}"
                self._begin_memory_formation(percept, run_id)

    def _on_tool_result(self, result: ToolResult) -> None:
        """Track tool results as outcomes."""
        outcome = {
            "timestamp": time.time(),
            "tool_name": result.tool_name,
            "success": result.success,
            "error": result.error,
        }
        with self._lock:
            self._recent_outcomes.append(outcome)

            # Store significant outcomes as standalone entries
            if not result.success:
                episodic = EpisodicMemory(
                    id=f"wm-fail-{time.time():.0f}",
                    timestamp=time.time(),
                    action=Action(tool_name=result.tool_name),
                    outcome=Outcome(
                        success=False,
                        error=result.error,
                    ),
                )
                entry = WorkingMemoryEntry(
                    record=episodic,
                    salience=0.8,
                    source="goal_outcome",
                    decay_rate=0.02,
                )
                self._add_memory(entry)

    def _on_goal_completed(self, completed: GoalCompleted) -> None:
        """Track goal completion."""
        with self._lock:
            if self._active_goal == completed.goal_id:
                self._active_goal = None
                self._active_goal_description = None
                self._active_sub_goals = []

            outcome_data = {
                "timestamp": time.time(),
                "goal_id": completed.goal_id,
                "success": completed.success,
                "error": completed.error,
            }
            self._recent_outcomes.append(outcome_data)

            # Store as memory
            episodic = EpisodicMemory(
                id=f"wm-goal-{time.time():.0f}",
                timestamp=time.time(),
                outcome=Outcome(
                    success=completed.success,
                    error=completed.error,
                ),
            )
            entry = WorkingMemoryEntry(
                record=episodic,
                salience=0.7 if completed.success else 0.9,
                source="goal_outcome",
                decay_rate=0.03,
            )
            self._add_memory(entry)

    def _on_goal_proposed(self, goal: ProposedGoal) -> None:
        """Observe all proposed goals."""
        episodic = EpisodicMemory(
            id=f"wm-prop-{time.time():.0f}",
            timestamp=time.time(),
            decision=Decision(
                intent={"goal": goal.description},
                confidence=goal.confidence,
            ),
        )
        entry = WorkingMemoryEntry(
            record=episodic,
            salience=0.6,
            source="goal_proposed",
            decay_rate=0.08,
        )
        with self._lock:
            self._add_memory(entry)

    def _on_goal_accepted(self, accepted: GoalAccepted) -> None:
        """Track when goals are accepted."""
        with self._lock:
            self._active_goal = accepted.goal_id

    def _on_statistical_summary(self, summary: StatisticalSummary) -> None:
        """Receive statistical summary from StatisticianAgent via bus."""
        self._latest_statistical_summary = summary.summary
        self._active_pattern_count = summary.active_patterns

        # Capture analysis suggestions
        raw_suggestions = getattr(summary, "suggestions", [])
        self._statistical_suggestions = []
        for s in raw_suggestions:
            try:
                self._statistical_suggestions.append({
                    "metric": s.metric,
                    "tool_call": s.tool_call,
                    "operation": s.operation,
                    "rationale": s.rationale,
                    "priority": s.priority,
                    "data_type": s.data_type,
                    "fsm_state": s.fsm_state,
                })
            except AttributeError:
                pass

    # ── Memory Management ─────────────────────────────────────────────────

    def _add_memory(self, entry: WorkingMemoryEntry) -> None:
        """Add entry to stores and indexes (must hold lock)."""
        self._short_term.append(entry)
        self._association_index.add(entry)
        self._association_graph.add_memory(entry)

        # Build keyword-based associations
        similar = self._association_index.find_similar(entry, top_k=3)
        for other_entry, score in similar:
            if score > 0.3 and other_entry.id != entry.id:
                self._association_graph.associate(entry, other_entry, weight=score)

        # Check for promotion
        self._check_promotions()

    def _compute_memory_salience(self, percept: Percept) -> float:
        """Compute retention weight using PerceptionAgent salience."""
        base = percept.salience

        if percept.has_maxim_keyword:
            base = max(base, 0.9)

        if percept.novelty > 0.7:
            base = min(1.0, base + 0.2)

        for det in percept.detections:
            if det.get("class_id") == 0:  # Person class
                base = min(1.0, base + 0.1)
                break

        return base

    def _apply_decay(self) -> None:
        """Apply salience decay to memories (must hold lock)."""
        now = time.time()
        if now - self._last_decay_time < self._decay_interval:
            return

        elapsed = now - self._last_decay_time
        self._last_decay_time = now

        # Decay short-term
        surviving_short: deque[WorkingMemoryEntry] = deque(maxlen=self._max_short_term)
        for entry in self._short_term:
            entry.salience -= entry.decay_rate * elapsed
            if entry.salience > self._salience_threshold:
                surviving_short.append(entry)
            else:
                self._association_index.remove(entry.id)
        self._short_term = surviving_short

        # Evict old long-term memories
        surviving_long: list[WorkingMemoryEntry] = []
        for entry in self._long_term:
            if not entry.should_evict(self._long_term_max_age):
                surviving_long.append(entry)
            else:
                self._association_index.remove(entry.id)
        self._long_term = surviving_long

    def _check_promotions(self) -> None:
        """Check for memories to promote (must hold lock)."""
        now = time.time()
        if now - self._last_promotion_time < 10.0:
            return
        self._last_promotion_time = now

        for entry in list(self._short_term):
            if entry.should_promote():
                entry.tier = MemoryTier.LONG_TERM
                self._long_term.append(entry)
                self._association_index.build_associations(entry)

                # Limit long-term size
                if len(self._long_term) > self._max_long_term:
                    oldest = min(self._long_term, key=lambda e: e.record.accessed_at)
                    self._long_term.remove(oldest)
                    self._association_index.remove(oldest.id)

    def _get_relevant_memories(self, current: Percept | None) -> list[WorkingMemoryEntry]:
        """Get memories relevant to current context using both systems.

        Includes FORMING/WORKING entries from the forming pool (current episode).
        """
        # Include forming pool entries (current pipeline episodes)
        forming_entries = list(self._forming_pool.values())
        all_memories = forming_entries + list(self._short_term) + self._long_term

        if current is None:
            sorted_entries = sorted(
                all_memories, key=lambda e: e.current_salience(), reverse=True
            )
            return list(sorted_entries[: self._context_window])

        # Stage 1: Keyword similarity
        query = current.raw_transcript_text or str(current.detections)
        keyword_similar = self._association_index.find_similar(
            query, top_k=self._context_window
        )

        # Stage 2: Spreading activation from keyword matches
        seed_entries = [entry for entry, _ in keyword_similar[:3]]
        if seed_entries:
            graph_related = self._association_graph.get_related_memories(
                seed_entries, top_k=self._context_window
            )
        else:
            graph_related = []

        # Combine and deduplicate
        seen: set[str] = set()
        combined: list[tuple[WorkingMemoryEntry, float]] = []

        # Always include forming pool entries (current episode context)
        for entry in forming_entries:
            seen.add(entry.id)
            combined.append((entry, entry.current_salience() + 1.0))  # Boost forming entries

        for entry, score in keyword_similar:
            if entry.id not in seen:
                seen.add(entry.id)
                combined.append((entry, score * 0.5 + entry.current_salience() * 0.5))

        for entry, activation in graph_related:
            if entry.id not in seen:
                seen.add(entry.id)
                combined.append((entry, activation * 0.4 + entry.current_salience() * 0.6))

        combined.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in combined[: self._context_window]]

    def _extract_detected_objects(self, percepts: list[Percept]) -> list[dict]:
        """Extract unique detected objects from recent percepts."""
        seen_tracks: set[Any] = set()
        objects: list[dict] = []
        untracked_count = 0
        for p in reversed(percepts):
            for det in p.detections:
                track_id = det.get("track_id")
                if track_id is not None:
                    if track_id not in seen_tracks:
                        seen_tracks.add(track_id)
                        objects.append(det)
                else:
                    if untracked_count < 5:
                        objects.append(det)
                        untracked_count += 1
        return objects[:10]

    def _extract_detected_people(self, percepts: list[Percept]) -> list[dict]:
        """Extract detected people."""
        people: list[dict] = []
        for p in reversed(percepts):
            for det in p.detections:
                if det.get("class_id") == 0:
                    people.append(det)
        return people[:5]

    def _extract_speech(self, percepts: list[Percept]) -> list[str]:
        """Extract recent speech."""
        speech: list[str] = []
        for p in reversed(percepts):
            if p.raw_transcript_text:
                speech.append(p.raw_transcript_text)
        return speech[:5]

    # ── Working notes ──────────────────────────────────────────────────────

    WORKING_NOTES_MAX_CHARS = 2000

    def _read_working_notes(self) -> str:
        """Read .maxim_workspace/notes/context.md if it exists."""
        if not self._workspace_path:
            return ""

        notes_path = os.path.join(self._workspace_path, "notes", "context.md")
        if not os.path.isfile(notes_path):
            return ""

        try:
            with open(notes_path, "r") as f:
                content = f.read()
        except OSError:
            return ""

        if len(content) > self.WORKING_NOTES_MAX_CHARS:
            content = content[: self.WORKING_NOTES_MAX_CHARS]
            content += (
                "\n\n[TRUNCATED — notes exceed "
                f"{self.WORKING_NOTES_MAX_CHARS} chars. "
                "Edit the file to prune stale entries.]"
            )

        return content.strip()

    # ── Workspace context ──────────────────────────────────────────────────

    WORKSPACE_FILE_CAP = 10
    _WORKSPACE_EXCLUDE = frozenset({
        os.path.join("plans", "ACTIVE_PLAN.md"),
        os.path.join("plans", "history.md"),
        os.path.join("notes", "context.md"),
    })

    def _scan_workspace_files(self) -> list[dict]:
        """Scan .maxim_workspace/ for user-facing artifacts."""
        if not self._workspace_path or not os.path.isdir(self._workspace_path):
            return []

        entries: list[dict] = []
        for dirpath, _dirs, filenames in os.walk(self._workspace_path):
            if "history_archive" in dirpath:
                continue
            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, self._workspace_path)

                if rel_path in self._WORKSPACE_EXCLUDE:
                    continue
                if fname.startswith(".") or fname.endswith(".tmp"):
                    continue

                try:
                    stat = os.stat(full_path)
                    entries.append({
                        "path": rel_path,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    })
                except OSError:
                    continue

        entries.sort(key=lambda e: e["modified"], reverse=True)
        return entries[: self.WORKSPACE_FILE_CAP]

    # ── Context Building ──────────────────────────────────────────────────

    def build_context(self, persist_snapshot: bool = False) -> StructuredContext:
        """Build structured context for goal proposal.

        Uses polymorphic to_context_dict() on each record — no isinstance chains.
        Includes pattern prediction context for FORMING entries.
        """
        with self._lock:
            self._apply_decay()

            recent = list(self._recent_percepts)
            current = recent[-1] if recent else None

            # Include sandbox tools if output manager is configured
            available_tools = [
                "focus_interests",
                "maxim_command",
                "read_file",
                "write_file",
                "execute_file",
            ]
            if self._output_manager is not None:
                available_tools.extend([
                    "read_data_file",
                    "read_sandbox_file",
                    "write_sandbox_file",
                    "execute_sandbox_script",
                    "list_other_outputs",
                    "write_shared_output",
                ])

            # Get relevant memories and convert to MemoryItem-compatible dicts
            relevant_entries = self._get_relevant_memories(current)

            # Build context items using polymorphic to_context_dict()
            context_items: list[dict] = []
            for entry in relevant_entries:
                ctx = entry.record.to_context_dict()
                ctx["salience"] = entry.current_salience()
                ctx["tier"] = entry.tier.value
                # Include predictions for FORMING entries
                if entry.predicted_outcomes:
                    ctx["predictions"] = [
                        {
                            "tool": p.tool,
                            "success": p.success,
                            "goal": p.goal,
                            "confidence": p.confidence,
                            "math_context": (
                                [m.to_dict() for m in p.math_context]
                                if p.math_context
                                else None
                            ),
                        }
                        for p in entry.predicted_outcomes
                    ]
                    ctx["prediction_confidence"] = entry.prediction_confidence
                context_items.append(ctx)

            # Build MemoryItem-compatible list for StructuredContext.relevant_memories
            relevant_memory_items = self._entries_to_memory_items(relevant_entries)

            context = StructuredContext(
                timestamp=time.time(),
                current_percept=current,
                active_goal=self._active_goal_description,
                active_goal_sub_goals=list(self._active_sub_goals),
                mode=self._mode,
                recent_percepts=recent[-self._context_window :],
                recent_outcomes=list(self._recent_outcomes),
                relevant_memories=relevant_memory_items,
                detected_objects=self._extract_detected_objects(recent),
                detected_people=self._extract_detected_people(recent),
                detected_speech=self._extract_speech(recent),
                recent_logs=self._abstraction.get_recent(n=15),
                goal_history=self._abstraction.get_by_event("goal_proposed", n=5),
                cli_inputs=list(self._cli_inputs),
                comms_messages=list(self._comms_messages),
                available_environments=available_tools,
                statistical_context=self._latest_statistical_summary,
                active_pattern_count=self._active_pattern_count,
                statistical_suggestions=self._statistical_suggestions,
                knowledge_context=self._build_knowledge_context(),
                root_goal=self.ROOT_GOAL,
                working_notes=self._read_working_notes(),
                workspace_files=self._scan_workspace_files(),
                plan_progress=self._build_plan_progress(),
            )

            # Persist snapshot to shared outputs if requested
            if persist_snapshot and self._output_manager is not None:
                try:
                    context_dict = {
                        "timestamp": context.timestamp,
                        "active_goal": context.active_goal,
                        "mode": context.mode,
                        "detected_speech": context.detected_speech,
                        "cli_inputs": context.cli_inputs,
                        "memory_count": len(context.relevant_memories),
                        "object_count": len(context.detected_objects),
                        "people_count": len(context.detected_people),
                    }
                    self._write_context(context_dict, share=True)
                except Exception as e:
                    log_swallowed_exception(e, operation="write_context")

            return context

    def _entries_to_memory_items(
        self, entries: list[WorkingMemoryEntry]
    ) -> list[MemoryItem]:
        """Convert WorkingMemoryEntry list to MemoryItem list for backward compat.

        StructuredContext.relevant_memories still expects list[MemoryItem].
        This bridge maintains compatibility until downstream consumers are updated.
        """
        items: list[MemoryItem] = []
        for entry in entries:
            items.append(
                MemoryItem(
                    timestamp=entry.timestamp,
                    content=entry.record.to_context_dict(),
                    salience=entry.current_salience(),
                    source=entry.source,
                    decay_rate=entry.decay_rate,
                    keywords=entry.keywords,
                    tier=entry.tier if entry.tier in (MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM)
                    else MemoryTier.SHORT_TERM,
                )
            )
        return items

    def _build_plan_progress(self) -> Any:
        """Build PlanProgressContext from active plan, or None."""
        if not self._plan_manager or not self._plan_manager.active_plan:
            return None

        from maxim.agents.bus import PlanProgressContext
        from maxim.planning.plan_document import PhaseStatus

        plan = self._plan_manager.active_plan
        current = plan.get_current_phase()

        return PlanProgressContext(
            plan_id=plan.id,
            objective=plan.objective,
            status=plan.status.name,
            current_phase_index=plan.current_phase_index,
            total_phases=len(plan.phases),
            current_phase_description=current.description if current else "",
            phases_completed=sum(
                1 for p in plan.phases if p.status == PhaseStatus.COMPLETED
            ),
            phases_failed=sum(
                1 for p in plan.phases if p.status == PhaseStatus.FAILED
            ),
            energy_utilization=(
                current.energy_budget.utilization
                if current and current.energy_budget else {}
            ),
            is_replanning=plan.status.name == "REPLANNING",
            replan_count=len(plan.replan_history),
        )

    def _build_knowledge_context(self) -> list[dict]:
        """Merge knowledge from ATL + Angular Gyrus into unified context."""
        if self._memory_hub is None:
            return []

        entries: list[dict] = []

        hub = self._memory_hub
        if getattr(hub, "atl", None) is not None:
            try:
                concepts = hub.atl.recall(limit=5, min_confidence=0.5)
                for concept in concepts:
                    rels: list[dict] = []
                    try:
                        rel_pairs = hub.atl.find_by_relationship(concept.id, limit=3)
                        for target_id, r in rel_pairs:
                            rels.append({
                                "type": r.relationship_type,
                                "target": target_id,
                                "weight": r.weight,
                            })
                    except Exception:
                        pass

                    entries.append({
                        "concept_name": concept.name,
                        "definition": getattr(concept, "definition", ""),
                        "category": concept.category,
                        "confidence": concept.confidence,
                        "source_layer": "atl",
                        "provenance": getattr(concept, "provenance", "").name
                        if hasattr(getattr(concept, "provenance", None), "name")
                        else str(getattr(concept, "provenance", "")),
                        "relationships": rels,
                        "relevance": concept.confidence,
                    })
            except Exception:
                pass

        if getattr(hub, "angular_gyrus", None) is not None:
            try:
                from maxim.math.types import MathCategory

                patterns = hub.angular_gyrus.recall(
                    limit=3,
                    category=MathCategory.PATTERN,
                    min_confidence=0.5,
                )
                for record in patterns:
                    entries.append({
                        "concept_name": record.name,
                        "definition": record.verbal,
                        "category": f"math:{record.category.name}",
                        "confidence": record.confidence,
                        "source_layer": "angular_gyrus",
                        "provenance": record.source,
                        "relationships": [],
                        "relevance": record.confidence * 0.8,
                    })
            except Exception:
                pass

        entries.sort(key=lambda e: e.get("relevance", 0), reverse=True)
        return entries[:8]

    # ── Public API ────────────────────────────────────────────────────────

    def set_active_goal(self, goal_id: str | None, description: str | None = None) -> None:
        """Set the currently active goal."""
        with self._lock:
            self._active_goal = goal_id
            self._active_goal_description = description

    def set_active_sub_goals(self, sub_goals: list[str]) -> None:
        """Set the pending sub-goals for the current goal."""
        with self._lock:
            self._active_sub_goals = list(sub_goals)

    def add_cli_input(self, command: str) -> None:
        """Record a CLI input for context."""
        with self._lock:
            self._cli_inputs.append(command)

    def check_startup(self) -> bool:
        """Check if this is first run (for startup tasks)."""
        with self._lock:
            if not self._did_startup:
                self._did_startup = True
                return True
            return False

    # ── Persistence ───────────────────────────────────────────────────────

    def save_state(self, path: str | None = None) -> None:
        """Persist working memory state for session recovery (v2.0 format)."""
        save_path = path or self._persistence_path
        if not save_path:
            return

        with self._lock:
            state = {
                "version": "2.0",
                "short_term": [e.to_dict() for e in self._short_term],
                "long_term": [e.to_dict() for e in self._long_term],
                "forming_pool": {
                    run_id: e.to_dict()
                    for run_id, e in self._forming_pool.items()
                },
            }

        # Atomic write: write to temp file, then rename
        try:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            tmp_path = save_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f)
            os.replace(tmp_path, save_path)
        except (OSError, IOError) as e:
            log_swallowed_exception(e, operation="save_memory_state", context={"path": save_path})

    def load_state(self, path: str | None = None) -> None:
        """Restore working memory state from persistence."""
        load_path = path or self._persistence_path
        if not load_path or not os.path.exists(load_path):
            return

        try:
            with open(load_path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log_swallowed_exception(e, operation="load_memory_state")
            return

        version = state.get("version", "1.0")
        if version == "1.0":
            # Legacy MemoryItem format — skip (Hippocampus has canonical records)
            return

        for item in state.get("short_term", []):
            try:
                entry = WorkingMemoryEntry.from_dict(item)
                self._short_term.append(entry)
                self._association_index.add(entry)
                self._association_graph.add_memory(entry)
            except Exception as e:
                log_swallowed_exception(e, operation="load_short_term_entry")

        for item in state.get("long_term", []):
            try:
                entry = WorkingMemoryEntry.from_dict(item)
                self._long_term.append(entry)
                self._association_index.add(entry)
                self._association_graph.add_memory(entry)
            except Exception as e:
                log_swallowed_exception(e, operation="load_long_term_entry")

        # FORMING entries from a crashed session → transition to SHORT_TERM
        for run_id, forming_data in state.get("forming_pool", {}).items():
            try:
                entry = WorkingMemoryEntry.from_dict(forming_data)
                entry.tier = MemoryTier.SHORT_TERM  # Can't resume FORMING
                self._short_term.appendleft(entry)
                self._association_index.add(entry)
                self._association_graph.add_memory(entry)
            except Exception as e:
                log_swallowed_exception(e, operation="load_forming_entry")

    def _load_memories_from_disk(self) -> None:
        """Load persisted memories on startup."""
        if not self._persistence_path:
            return
        try:
            if os.path.exists(self._persistence_path):
                with open(self._persistence_path, "r") as f:
                    data = json.load(f)

                version = data.get("version", "1.0")
                if version == "2.0":
                    self.load_state(self._persistence_path)
                    return

                # Legacy v1.0 MemoryItem format
                for item in data.get("memories", []):
                    episodic = EpisodicMemory(
                        id=f"legacy-{item['timestamp']:.0f}",
                        timestamp=item["timestamp"],
                    )
                    entry = WorkingMemoryEntry(
                        record=episodic,
                        salience=item["salience"],
                        source=item["source"],
                        decay_rate=item.get("decay_rate", 0.1),
                        tier=MemoryTier(item.get("tier", "short")),
                    )
                    if entry.tier == MemoryTier.LONG_TERM:
                        self._long_term.append(entry)
                    else:
                        self._short_term.append(entry)
                    self._association_index.add(entry)
                    self._association_graph.add_memory(entry)
        except Exception as e:
            log_swallowed_exception(e, operation="load_persisted_memories")

    def _persist_memories_to_disk(self, share_to_outputs: bool = False) -> None:
        """Persist memories to disk using v2.0 format."""
        self.save_state()

        # Also write to output manager if available
        if share_to_outputs and self._output_manager is not None:
            try:
                to_persist = []
                for entry in list(self._short_term) + self._long_term:
                    if entry.salience > self._salience_threshold * 1.5 or entry.tier == MemoryTier.LONG_TERM:
                        to_persist.append(entry.record.to_context_dict())
                self._write_memory(to_persist, share=True)
            except Exception as e:
                log_swallowed_exception(e, operation="persist_memory_to_output_manager")

    def propose_intent(self, state: Any, memory: Any, **kwargs: Any) -> dict[str, Any] | None:
        """MemoryAgent doesn't propose intents directly."""
        return None

    def on_stop(self, **kwargs: Any) -> None:
        """Clean up and persist on stop."""
        self._persist_memories_to_disk()

        self._bus.unsubscribe(Percept, self._on_percept)
        self._bus.unsubscribe(ToolResult, self._on_tool_result)
        self._bus.unsubscribe(GoalCompleted, self._on_goal_completed)
        self._bus.unsubscribe(ProposedGoal, self._on_goal_proposed)
        self._bus.unsubscribe(GoalAccepted, self._on_goal_accepted)
        self._bus.unsubscribe(StatisticalSummary, self._on_statistical_summary)
