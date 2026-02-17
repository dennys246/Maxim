"""MemoryAgent: Core memory system with salience, associations, and persistence.

THIS IS THE CENTRAL RESEARCH FOCUS OF THE PROJECT.

The MemoryAgent maintains salient memories, builds associations via similarity,
and provides structured context for goal proposal.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any

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
)
from maxim.agents.output_mixin import AgentOutputMixin
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
        self._memories: dict[str, MemoryItem] = {}
        self._embedding_model = embedding_model
        self._embedder = None  # Lazy init
        self._lock = threading.Lock()

    def add(self, memory: MemoryItem) -> None:
        """Add memory to index."""
        mid = memory.memory_id
        with self._lock:
            self._memories[mid] = memory

            # Extract and index keywords
            keywords = self._extract_keywords(memory.content)
            memory.keywords = keywords
            for kw in keywords:
                self._keyword_index[kw].add(mid)

    def remove(self, memory_id: str) -> None:
        """Remove memory from index."""
        with self._lock:
            if memory_id not in self._memories:
                return
            mem = self._memories[memory_id]
            for kw in mem.keywords:
                self._keyword_index[kw].discard(memory_id)
            del self._memories[memory_id]

    def find_similar(
        self,
        query: str | MemoryItem,
        top_k: int = 5,
        use_embeddings: bool = False,
    ) -> list[tuple[MemoryItem, float]]:
        """Find memories similar to query."""
        with self._lock:
            if isinstance(query, MemoryItem):
                query_keywords = query.keywords
                query_text = str(query.content)
            else:
                query_keywords = self._extract_keywords(query)
                query_text = query

            # Stage 1: Keyword overlap (Jaccard similarity)
            candidates: dict[str, float] = {}
            for kw in query_keywords:
                for mid in self._keyword_index.get(kw, set()):
                    if mid not in candidates:
                        candidates[mid] = 0.0
                    mem = self._memories.get(mid)
                    if mem:
                        intersection = len(query_keywords & mem.keywords)
                        union = len(query_keywords | mem.keywords)
                        if union > 0:
                            candidates[mid] = max(candidates[mid], intersection / union)

            # Stage 2: Optional embedding refinement
            if use_embeddings and self._embedding_model and candidates:
                candidates = self._refine_with_embeddings(query_text, candidates)

            # Sort and return top_k
            sorted_results = sorted(
                [
                    (self._memories[mid], score)
                    for mid, score in candidates.items()
                    if mid in self._memories
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            return sorted_results[:top_k]

    def build_associations(self, memory: MemoryItem, threshold: float = 0.3) -> None:
        """Automatically build associations based on similarity."""
        similar = self.find_similar(memory, top_k=5)
        for other_mem, score in similar:
            if score >= threshold and other_mem.memory_id != memory.memory_id:
                # Bidirectional association
                if other_mem.memory_id not in memory.associations:
                    memory.associations.append(other_mem.memory_id)
                if memory.memory_id not in other_mem.associations:
                    other_mem.associations.append(memory.memory_id)

    def _extract_keywords(self, content: Any) -> set[str]:
        """Extract keywords from content for indexing."""
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
                mem = self._memories.get(mid)
                if not mem:
                    continue
                if mem.embedding is None:
                    mem_text = (
                        str(mem.content)
                        if not isinstance(mem.content, str)
                        else mem.content
                    )
                    mem.embedding = self._embedder.encode(
                        mem_text, convert_to_numpy=True
                    ).tolist()

                mem_emb = np.array(mem.embedding)
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
        self._graph: DependencyGraph[MemoryItem] = DependencyGraph()
        self._temporal_index: dict[int, list[str]] = defaultdict(list)

    def add_memory(self, memory: MemoryItem) -> None:
        """Add memory to the association graph."""
        self._graph.add_node(memory.memory_id, memory)

        hour_bucket = int(memory.timestamp // 3600)
        self._temporal_index[hour_bucket].append(memory.memory_id)

        self._build_temporal_associations(memory)

    def associate(
        self,
        memory_a: MemoryItem,
        memory_b: MemoryItem,
        weight: float = 1.0,
        edge_type: EdgeType = EdgeType.ASSOCIATES,
    ) -> None:
        """Create bidirectional association."""
        self._graph.add_bidirectional(
            memory_a.memory_id, memory_b.memory_id, edge_type, weight
        )
        if memory_b.memory_id not in memory_a.associations:
            memory_a.associations.append(memory_b.memory_id)
        if memory_a.memory_id not in memory_b.associations:
            memory_b.associations.append(memory_a.memory_id)

    def add_causal_link(
        self,
        cause: MemoryItem,
        effect: MemoryItem,
        weight: float = 1.0,
    ) -> None:
        """Record causal relationship."""
        self._graph.add_edge(
            cause.memory_id, effect.memory_id, EdgeType.CAUSES, weight
        )

    def _build_temporal_associations(
        self,
        memory: MemoryItem,
        window_hours: int = 1,
        max_associations: int = 5,
    ) -> None:
        """Associate with memories close in time."""
        hour_bucket = int(memory.timestamp // 3600)

        nearby_ids: list[str] = []
        for offset in range(-window_hours, window_hours + 1):
            nearby_ids.extend(self._temporal_index.get(hour_bucket + offset, []))

        nearby_ids = [
            mid
            for mid in nearby_ids
            if mid != memory.memory_id and self._graph.get_node(mid) is not None
        ]

        for mid in nearby_ids[:max_associations]:
            other = self._graph.get_node(mid)
            if other:
                time_diff = abs(memory.timestamp - other.timestamp)
                weight = 1.0 / (1.0 + time_diff / 3600)
                self._graph.add_bidirectional(
                    memory.memory_id, mid, EdgeType.ASSOCIATES, weight
                )

    def get_related_memories(
        self,
        query_memories: list[MemoryItem],
        top_k: int = 10,
        activation_decay: float = 0.5,
    ) -> list[tuple[MemoryItem, float]]:
        """Get related memories via spreading activation."""
        source_ids = [m.memory_id for m in query_memories]

        activations = self._graph.spreading_activation(
            source_ids,
            initial_activation=1.0,
            decay=activation_decay,
            threshold=0.05,
            max_depth=4,
        )

        results: list[tuple[MemoryItem, float]] = []
        for mid, activation in activations.items():
            if mid not in source_ids:
                mem = self._graph.get_node(mid)
                if mem:
                    results.append((mem, activation))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class MemoryAgent(Agent, AgentOutputMixin):
    """
    Maintains salient memory and builds structured context.

    THIS IS THE CORE EXPLORATION FOCUS OF THE PROJECT.

    Responsibilities:
    - Preserve salient moments (uses PerceptionAgent salience/novelty)
    - Recall similar states to enrich StructuredContext
    - Apply salience decay over time
    - Inhibit low-salience memories
    - Track goal outcomes for learning
    - Build StructuredContext for ExecAgent
    - Passively observe ALL agent states and actions
    - Persist memories across sessions
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

        # Two-tier memory stores
        self._short_term: deque[MemoryItem] = deque(maxlen=max_short_term)
        self._long_term: list[MemoryItem] = []
        self._recent_percepts: deque[Percept] = deque(maxlen=context_window)
        self._recent_outcomes: deque[dict] = deque(maxlen=context_window)
        self._cli_inputs: deque[str] = deque(maxlen=20)

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

        # Load persisted memories
        if not self._reset_on_startup and self._persistence_path:
            self._load_memories_from_disk()

        # Statistical context (from StatisticianAgent via bus)
        self._latest_statistical_summary: str = ""
        self._active_pattern_count: int = 0
        self._statistical_suggestions: list[dict] = []

        # Optional MemoryHub for multi-layer knowledge queries
        self._memory_hub = memory_hub

        # Subscribe to messages
        self._bus.subscribe(Percept, self._on_percept)
        self._bus.subscribe(ToolResult, self._on_tool_result)
        self._bus.subscribe(GoalCompleted, self._on_goal_completed)
        self._bus.subscribe(ProposedGoal, self._on_goal_proposed)
        self._bus.subscribe(GoalAccepted, self._on_goal_accepted)
        self._bus.subscribe(StatisticalSummary, self._on_statistical_summary)

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

            # Store as memory if salient
            if percept.salience > self._salience_threshold or percept.has_maxim_keyword:
                memory = MemoryItem(
                    timestamp=percept.timestamp,
                    content={
                        "source": percept.source,
                        "transcript": percept.raw_transcript_text,
                        "has_maxim": percept.has_maxim_keyword,
                        "detections_count": len(percept.detections),
                    },
                    salience=self._compute_memory_salience(percept),
                    source="percept",
                    decay_rate=0.05 if percept.has_maxim_keyword else 0.1,
                )
                self._add_memory(memory)

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

            # Store significant outcomes
            if not result.success:
                memory = MemoryItem(
                    timestamp=time.time(),
                    content=outcome,
                    salience=0.8,
                    source="goal_outcome",
                    decay_rate=0.02,
                )
                self._add_memory(memory)

    def _on_goal_completed(self, completed: GoalCompleted) -> None:
        """Track goal completion."""
        with self._lock:
            if self._active_goal == completed.goal_id:
                self._active_goal = None
                self._active_goal_description = None
                self._active_sub_goals = []

            outcome = {
                "timestamp": time.time(),
                "goal_id": completed.goal_id,
                "success": completed.success,
                "error": completed.error,
            }
            self._recent_outcomes.append(outcome)

            # Store as memory
            memory = MemoryItem(
                timestamp=time.time(),
                content=outcome,
                salience=0.7 if completed.success else 0.9,
                source="goal_outcome",
                decay_rate=0.03,
            )
            self._add_memory(memory)

    def _on_goal_proposed(self, goal: ProposedGoal) -> None:
        """Observe all proposed goals."""
        memory = MemoryItem(
            timestamp=time.time(),
            content={
                "goal_id": goal.id,
                "description": goal.description,
                "priority": goal.priority.name,
            },
            salience=0.6,
            source="goal_proposed",
            decay_rate=0.08,
        )
        with self._lock:
            self._add_memory(memory)

    def _on_goal_accepted(self, accepted: GoalAccepted) -> None:
        """Track when goals are accepted."""
        with self._lock:
            self._active_goal = accepted.goal_id

    def _on_statistical_summary(self, summary: StatisticalSummary) -> None:
        """Receive statistical summary from StatisticianAgent via bus."""
        self._latest_statistical_summary = summary.summary
        self._active_pattern_count = summary.active_patterns

        # Capture analysis suggestions (backward-compat with older StatisticalSummary)
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

    def _add_memory(self, memory: MemoryItem) -> None:
        """Add memory to stores and indexes (must hold lock)."""
        self._short_term.append(memory)
        self._association_index.add(memory)
        self._association_graph.add_memory(memory)

        # Build keyword-based associations
        similar = self._association_index.find_similar(memory, top_k=3)
        for other_mem, score in similar:
            if score > 0.3 and other_mem.memory_id != memory.memory_id:
                self._association_graph.associate(memory, other_mem, weight=score)

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
        surviving_short: deque[MemoryItem] = deque(maxlen=self._max_short_term)
        for mem in self._short_term:
            mem.salience -= mem.decay_rate * elapsed
            if mem.salience > self._salience_threshold:
                surviving_short.append(mem)
            else:
                self._association_index.remove(mem.memory_id)
        self._short_term = surviving_short

        # Evict old long-term memories
        surviving_long: list[MemoryItem] = []
        for mem in self._long_term:
            if not mem.should_evict_long_term(self._long_term_max_age):
                surviving_long.append(mem)
            else:
                self._association_index.remove(mem.memory_id)
        self._long_term = surviving_long

    def _check_promotions(self) -> None:
        """Check for memories to promote (must hold lock)."""
        now = time.time()
        if now - self._last_promotion_time < 10.0:
            return
        self._last_promotion_time = now

        for mem in list(self._short_term):
            if mem.should_promote():
                mem.tier = MemoryTier.LONG_TERM
                mem.promoted_at = time.time()
                self._long_term.append(mem)
                self._association_index.build_associations(mem)

                # Limit long-term size
                if len(self._long_term) > self._max_long_term:
                    oldest = min(self._long_term, key=lambda m: m.last_accessed)
                    self._long_term.remove(oldest)
                    self._association_index.remove(oldest.memory_id)

    def _get_relevant_memories(self, current: Percept | None) -> list[MemoryItem]:
        """Get memories relevant to current context using both systems."""
        all_memories = list(self._short_term) + self._long_term

        if current is None:
            sorted_mems = sorted(all_memories, key=lambda m: m.salience, reverse=True)
            return list(sorted_mems[: self._context_window])

        # Stage 1: Keyword similarity
        query = current.raw_transcript_text or str(current.detections)
        keyword_similar = self._association_index.find_similar(
            query, top_k=self._context_window
        )

        # Stage 2: Spreading activation from keyword matches
        seed_memories = [mem for mem, _ in keyword_similar[:3]]
        if seed_memories:
            graph_related = self._association_graph.get_related_memories(
                seed_memories, top_k=self._context_window
            )
        else:
            graph_related = []

        # Combine and deduplicate
        seen: set[str] = set()
        combined: list[tuple[MemoryItem, float]] = []

        for mem, score in keyword_similar:
            if mem.memory_id not in seen:
                seen.add(mem.memory_id)
                combined.append((mem, score * 0.5 + mem.salience * 0.5))

        for mem, activation in graph_related:
            if mem.memory_id not in seen:
                seen.add(mem.memory_id)
                combined.append((mem, activation * 0.4 + mem.salience * 0.6))

        combined.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in combined[: self._context_window]]

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
                    # Include untracked detections (limited to avoid duplicates)
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

    def build_context(self, persist_snapshot: bool = False) -> StructuredContext:
        """Build structured context for goal proposal.

        Args:
            persist_snapshot: If True, write context snapshot to shared outputs
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

            context = StructuredContext(
                timestamp=time.time(),
                current_percept=current,
                active_goal=self._active_goal_description,
                active_goal_sub_goals=list(self._active_sub_goals),
                mode=self._mode,
                recent_percepts=recent[-self._context_window :],
                recent_outcomes=list(self._recent_outcomes),
                relevant_memories=self._get_relevant_memories(current),
                detected_objects=self._extract_detected_objects(recent),
                detected_people=self._extract_detected_people(recent),
                detected_speech=self._extract_speech(recent),
                recent_logs=self._abstraction.get_recent(n=15),
                goal_history=self._abstraction.get_by_event("goal_proposed", n=5),
                cli_inputs=list(self._cli_inputs),
                available_environments=available_tools,
                statistical_context=self._latest_statistical_summary,
                active_pattern_count=self._active_pattern_count,
                statistical_suggestions=self._statistical_suggestions,
                knowledge_context=self._build_knowledge_context(),
                root_goal=self.ROOT_GOAL,
                plan_progress=self._build_plan_progress(),
            )

            # Persist snapshot to shared outputs if requested
            if persist_snapshot and self._output_manager is not None:
                try:
                    # Convert to serializable dict (exclude non-serializable percepts)
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
        """Merge knowledge from ATL + Angular Gyrus into unified context.

        Returns a list of knowledge entries ranked by relevance, capped at 8.
        Each entry has: concept_name, definition, category, confidence,
        source_layer, provenance, relationships.
        """
        if self._memory_hub is None:
            return []

        entries: list[dict] = []

        # 1. ATL semantic concepts (if available)
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

        # 2. Angular Gyrus pattern memories (relevant learned knowledge)
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

        # 3. Rank by relevance, cap at 8 entries total
        entries.sort(key=lambda e: e.get("relevance", 0), reverse=True)
        return entries[:8]

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

    def _load_memories_from_disk(self) -> None:
        """Load persisted memories on startup."""
        if not self._persistence_path:
            return
        try:
            if os.path.exists(self._persistence_path):
                with open(self._persistence_path, "r") as f:
                    data = json.load(f)
                for item in data.get("memories", []):
                    mem = MemoryItem(
                        timestamp=item["timestamp"],
                        content=item["content"],
                        salience=item["salience"],
                        source=item["source"],
                        decay_rate=item.get("decay_rate", 0.1),
                        associations=item.get("associations", []),
                        tier=MemoryTier(item.get("tier", "short")),
                    )
                    if mem.tier == MemoryTier.LONG_TERM:
                        self._long_term.append(mem)
                    else:
                        self._short_term.append(mem)
                    self._association_index.add(mem)
                    self._association_graph.add_memory(mem)
        except Exception as e:
            log_swallowed_exception(e, operation="load_persisted_memories")

    def _persist_memories_to_disk(self, share_to_outputs: bool = False) -> None:
        """Persist high-salience memories to disk.

        Args:
            share_to_outputs: If True, also write to shared outputs for cross-instance visibility
        """
        # Build list of memories to persist
        to_persist = []
        threshold = self._salience_threshold * 1.5

        for m in list(self._short_term) + self._long_term:
            if m.salience > threshold or m.tier == MemoryTier.LONG_TERM:
                content = m.content
                if not isinstance(content, (dict, str, list, int, float, bool, type(None))):
                    content = str(content)
                to_persist.append(
                    {
                        "timestamp": m.timestamp,
                        "content": content,
                        "salience": m.salience,
                        "source": m.source,
                        "decay_rate": m.decay_rate,
                        "associations": m.associations,
                        "tier": m.tier.value,
                    }
                )

        # Write via output manager if available (preferred path)
        if self._output_manager is not None:
            try:
                self._write_memory(to_persist, share=share_to_outputs)
            except Exception as e:
                log_swallowed_exception(e, operation="persist_memory_to_output_manager")

        # Also write to legacy persistence path if specified
        if self._persistence_path:
            try:
                os.makedirs(os.path.dirname(self._persistence_path) or ".", exist_ok=True)
                with open(self._persistence_path, "w") as f:
                    json.dump({"memories": to_persist}, f, indent=2)
            except (OSError, IOError) as e:
                log_swallowed_exception(e, operation="persist_memory_to_disk", context={"path": self._persistence_path})

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
