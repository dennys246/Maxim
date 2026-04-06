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

import os
import threading
import time
from collections import deque
from typing import Any, Callable

from maxim.agents.base import Agent
from maxim.agents.bus import (
    AgentBus,
    GoalAccepted,
    GoalCompleted,
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
    Outcome,
    Perception,
    PredictedOutcome,
)
from maxim.memory.association_index import AssociationIndex
from maxim.memory.text import normalize_tokens
from maxim.utils.logging import log_swallowed_exception
from maxim.utils.structured_logging import get_abstraction_buffer


class MemoryAgent(Agent, AgentOutputMixin):
    """Orchestrates memory retrieval and salience scoring.

    THIS IS THE CORE EXPLORATION FOCUS OF THE PROJECT.

    After Phase 0 unification, Hippocampus owns all memory storage and
    associative graphs. MemoryAgent owns salience scoring, relevance
    ranking, and context building. They are complementary roles.

    Staged memory formation (FORMING → WORKING → SHORT_TERM → LONG_TERM)
    is managed via WorkingMemoryEntry wrappers around the records that
    Hippocampus stores. Pattern completion can attach predicted outcomes
    during FORMING via an optional hook.

    Responsibilities:
    - Score percept salience and track salience decay per memory
    - Capture memories via Hippocampus (single store)
    - Staged memory formation with WorkingMemoryEntry wrappers
    - Recall relevant memories for StructuredContext
    - Track goal outcomes for learning
    - Build StructuredContext for ExecAgent
    - Passively observe ALL agent states and actions
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
        self._salience_threshold = salience_threshold
        self._decay_interval = decay_interval
        self._context_window = context_window
        self._persistence_path = persistence_path
        self._long_term_max_age = long_term_max_age or self.DEFAULT_LONG_TERM_MAX_AGE
        self._reset_on_startup = reset_on_startup

        # Hippocampus reference (injected via connect_hippocampus)
        self._hippocampus: Any | None = None

        # Salience tracking — lightweight metadata for hippocampus memories
        self._salience: dict[str, float] = {}  # memory_id → current salience
        self._decay_rates: dict[str, float] = {}  # memory_id → decay rate
        self._recent_ids: deque[str] = deque(maxlen=max_short_term)  # Bounded window

        # Staged formation pool (keyed by run_id)
        self._forming_pool: dict[str, WorkingMemoryEntry] = {}

        # Pattern completion hook (set by ATL/MemoryHub wiring)
        self._pattern_completion_fn: Callable[[EpisodicMemory], list[PredictedOutcome]] | None = None

        # Recency buffers (not memory storage — just prompt context)
        self._recent_percepts: deque[Percept] = deque(maxlen=context_window)
        self._recent_outcomes: deque[dict] = deque(maxlen=context_window)
        self._cli_inputs: deque[str] = deque(maxlen=20)
        self._comms_messages: deque[dict] = deque(maxlen=20)

        # Association index (keyword-based, operates on hippocampus memory IDs)
        self._association_index = AssociationIndex(embedding_model="all-MiniLM-L6-v2" if enable_embeddings else None)

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

        # Statistical context (from StatisticianAgent via bus)
        self._latest_statistical_summary: str = ""
        self._active_pattern_count: int = 0
        self._statistical_suggestions: list[dict] = []

        # Optional MemoryHub for multi-layer knowledge queries
        self._memory_hub = memory_hub

        # Provenance collector (wired by MaximAgent.wire_provenance)
        self._collector: Any = None

        # Workspace path for reading working notes
        self._workspace_path = workspace_path

        # Subscribe to messages
        self._bus.subscribe(Percept, self._on_percept)
        self._bus.subscribe(ToolResult, self._on_tool_result)
        self._bus.subscribe(GoalCompleted, self._on_goal_completed)
        self._bus.subscribe(ProposedGoal, self._on_goal_proposed)
        self._bus.subscribe(GoalAccepted, self._on_goal_accepted)
        self._bus.subscribe(StatisticalSummary, self._on_statistical_summary)

    # ── Factory class methods ─────────────────────────────────────────────

    @classmethod
    def standalone(cls, config: dict[str, Any] | None = None) -> "MemoryAgent":
        """Create standalone MemoryAgent without external wiring."""
        bus = AgentBus()
        kwargs = dict(config or {})
        return cls(bus, **kwargs)

    # ── Wiring ────────────────────────────────────────────────────────────

    def connect_hippocampus(self, hippocampus: Any) -> None:
        """Wire Hippocampus reference for unified memory storage."""
        self._hippocampus = hippocampus

    def set_pattern_completion_fn(self, fn: Callable[[EpisodicMemory], list[PredictedOutcome]]) -> None:
        """Wire pattern completion hook (implemented in ATL concept plan)."""
        self._pattern_completion_fn = fn

    # ── Staged Memory Formation ───────────────────────────────────────────

    def _begin_memory_formation(self, percept: Percept, run_id: str) -> WorkingMemoryEntry:
        """Create a FORMING EpisodicMemory from percept + current agentic state.

        The entry is tracked in _forming_pool and also captured via Hippocampus.
        Pattern completion hook is invoked if wired.
        """
        from maxim.agents.bus import MemoryTier

        # Begin provenance trace for this cycle
        if self._collector:
            self._collector.begin_trace(run_id)

        now = time.time()

        # Sweep WORKING entries from pool
        self._flush_working_to_captured()

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
                    det.get("label", "person") for det in (percept.detections or []) if det.get("class_id") == 0
                ],
                salience=percept.salience,
                novelty=percept.novelty,
                cli_input=percept.cli_input or percept.raw_transcript_text,
                observations=percept.metadata or {},
            ),
            context=context,
            # Decision, Action, Outcome left as defaults (empty)
        )

        # Inject active skill name from protocol context (A7.1)
        protocol_registry = getattr(self, "_protocol_registry", None)
        if protocol_registry is not None:
            for proto in protocol_registry.get_active():
                skill_name = proto._context.get("_active_skill_name")
                if skill_name:
                    episodic.perception.observations["_skill_name"] = skill_name
                    break

        salience = self._compute_memory_salience(percept)
        decay_rate = 0.05 if percept.has_maxim_keyword else 0.1

        entry = WorkingMemoryEntry(
            record=episodic,
            salience=salience,
            source="percept",
            tier=MemoryTier.FORMING,
            decay_rate=decay_rate,
        )

        # Pattern completion hook
        if self._pattern_completion_fn:
            try:
                entry.predicted_outcomes = self._pattern_completion_fn(episodic)
                if entry.predicted_outcomes:
                    entry.prediction_confidence = self._compute_prediction_confidence(entry.predicted_outcomes)
            except Exception as e:
                log_swallowed_exception(e, operation="pattern_completion")

        self._forming_pool[run_id] = entry

        # Record pattern completion predictions (P3a)
        if self._collector and self._collector.verbosity >= 1:
            if entry.predicted_outcomes:
                from maxim.provenance.types import PipelineStage, ProvenanceRef

                refs = [
                    ProvenanceRef("hippocampus", p.source_episode_id, f"{p.tool} (success={p.success})", p.confidence)
                    for p in entry.predicted_outcomes
                    if p.source_episode_id
                ]
                self._collector.record(
                    run_id,
                    PipelineStage.RECALL,
                    "pattern_completer",
                    f"{len(refs)} predicted outcomes",
                    sources=refs,
                )

        # Also capture via Hippocampus
        content = {
            "source": percept.source,
            "transcript": percept.raw_transcript_text or percept.content,
            "has_maxim": percept.has_maxim_keyword,
            "detections_count": len(percept.detections),
        }
        memory_id = self._capture_to_hippocampus(content, salience, decay_rate, "percept", percept)
        if memory_id:
            entry._hippocampus_id = memory_id  # Cross-reference

        return entry

    def _update_forming_decision(self, run_id: str, decision: Decision) -> None:
        """Fill in the decision on a FORMING episodic memory."""
        from maxim.agents.bus import MemoryTier

        entry = self._forming_pool.get(run_id)
        if entry is None or entry.tier != MemoryTier.FORMING:
            return
        assert isinstance(entry.record, EpisodicMemory)
        entry.record.decision = decision
        entry.invalidate_keywords()

        # Record decision provenance (P3b)
        if self._collector and self._collector.verbosity >= 1:
            from maxim.provenance.types import PipelineStage

            action_name = decision.intent.get("action", "none") if decision.intent else "none"
            metadata: dict[str, Any] = {"reasoning": decision.reasoning}
            if self._collector.verbosity >= 2:
                metadata["alternatives"] = decision.alternatives_considered or []
            self._collector.record(
                run_id,
                PipelineStage.DECISION,
                "memory_agent",
                f"Action: {action_name} (confidence: {decision.confidence:.2f})",
                confidence=decision.confidence,
                **metadata,
            )

    def _update_forming_action(self, run_id: str, action: Action) -> None:
        """Fill in the action on a FORMING episodic memory."""
        from maxim.agents.bus import MemoryTier

        entry = self._forming_pool.get(run_id)
        if entry is None or entry.tier != MemoryTier.FORMING:
            return
        assert isinstance(entry.record, EpisodicMemory)
        entry.record.action = action
        entry.invalidate_keywords()

    def _complete_forming_memory(self, run_id: str, outcome: Outcome) -> None:
        """Fill in the outcome and transition FORMING → WORKING.

        Entry stays in _forming_pool as WORKING until next cycle sweeps it
        out via _flush_working_to_captured().
        """
        from maxim.agents.bus import MemoryTier

        entry = self._forming_pool.get(run_id)
        if entry is None or entry.tier != MemoryTier.FORMING:
            return
        assert isinstance(entry.record, EpisodicMemory)
        entry.record.outcome = outcome
        entry.tier = MemoryTier.WORKING
        entry.invalidate_keywords()

        # Record outcome provenance (P3c) then complete trace
        if self._collector and self._collector.verbosity >= 1:
            from maxim.provenance.types import PipelineStage

            duration = entry.record.action.execution_time_ms
            self._collector.record(
                run_id,
                PipelineStage.OUTCOME,
                "memory_agent",
                f"Success={outcome.success}, duration={duration:.0f}ms",
            )
        if self._collector:
            self._collector.complete_trace(run_id)

    def _flush_working_to_captured(self) -> None:
        """Transition WORKING entries in the pool — remove from forming pool.

        After Phase 0, Hippocampus owns the canonical record. We just clean up
        the local forming pool reference once the entry is no longer FORMING.
        """
        from maxim.agents.bus import MemoryTier

        to_remove = []
        for run_id, entry in self._forming_pool.items():
            if entry.tier == MemoryTier.WORKING:
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
                self._comms_messages.append(
                    {
                        "direction": "inbound",
                        "content": percept.content,
                        "channel": (percept.metadata or {}).get("channel", ""),
                        "sender": (percept.metadata or {}).get("sender", ""),
                        "timestamp": percept.timestamp,
                    }
                )

            # Capture via Hippocampus if salient — use staged formation
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

            # Store significant outcomes
            if not result.success:
                self._add_memory(outcome, 0.8, 0.02, "goal_outcome")

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

            salience = 0.7 if completed.success else 0.9
            self._add_memory(outcome, salience, 0.03, "goal_outcome")

    def _on_goal_proposed(self, goal: ProposedGoal) -> None:
        """Observe all proposed goals."""
        content = {
            "goal_id": goal.id,
            "description": goal.description,
            "priority": goal.priority.name,
        }
        with self._lock:
            self._add_memory(content, 0.6, 0.08, "goal_proposed")

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
                self._statistical_suggestions.append(
                    {
                        "metric": s.metric,
                        "tool_call": s.tool_call,
                        "operation": s.operation,
                        "rationale": s.rationale,
                        "priority": s.priority,
                        "data_type": s.data_type,
                        "fsm_state": s.fsm_state,
                    }
                )
            except AttributeError:
                pass

    # ── Memory Management ─────────────────────────────────────────────────

    def _capture_to_hippocampus(
        self,
        content: Any,
        salience: float,
        decay_rate: float,
        source: str,
        percept: Percept | None = None,
    ) -> str | None:
        """Capture a memory via Hippocampus and track its salience locally.

        Returns the memory ID if captured, None otherwise.
        """
        if not self._hippocampus:
            return None

        # Build observation dict from content
        observation = {}
        if percept is not None:
            observation = {
                "detections": [d for d in percept.detections],
                "transcript": percept.raw_transcript_text,
                "cli_input": percept.cli_input,
            }
        elif isinstance(content, dict):
            observation = content

        memory_id = self._hippocampus.capture_from_loop(
            observation=observation,
            state={"salience": salience, "novelty": 0.5, "source": source},
            intent={},
            decision={},
            action=content.get("tool_name", "") if isinstance(content, dict) else {},
            result=content if isinstance(content, dict) and "success" in content else {},
        )

        if memory_id:
            self._salience[memory_id] = salience
            self._decay_rates[memory_id] = decay_rate
            self._recent_ids.append(memory_id)

            # Index for keyword lookup
            self._association_index.add_by_id(memory_id, content)

            # Check for promotion
            self._check_promotions()

        return memory_id

    def _add_memory(
        self,
        content: Any,
        salience: float,
        decay_rate: float,
        source: str,
        percept: Percept | None = None,
    ) -> str | None:
        """Capture a memory via Hippocampus (must hold lock).

        MemoryAgent no longer creates MemoryItem objects. Hippocampus owns
        storage; MemoryAgent tracks the ID and salience metadata.
        """
        return self._capture_to_hippocampus(content, salience, decay_rate, source, percept)

    def _on_memory_deleted(self, memory_id: str) -> None:
        """Hippocampus pruned this memory — clean up local tracking."""
        self._salience.pop(memory_id, None)
        self._decay_rates.pop(memory_id, None)
        self._association_index.remove(memory_id)

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
        """Decay salience scores for tracked memories (must hold lock)."""
        now = time.time()
        if now - self._last_decay_time < self._decay_interval:
            return

        elapsed = now - self._last_decay_time
        self._last_decay_time = now

        # Decay salience metadata — does NOT remove from hippocampus
        to_remove = []
        for mid in list(self._salience):
            rate = self._decay_rates.get(mid, 0.1)
            self._salience[mid] -= rate * elapsed
            if self._salience[mid] <= self._salience_threshold:
                to_remove.append(mid)

        for mid in to_remove:
            self._salience.pop(mid, None)
            self._decay_rates.pop(mid, None)
            self._association_index.remove(mid)

    def _check_promotions(self) -> None:
        """Promote high-salience memories to long-term in Hippocampus."""
        now = time.time()
        if now - self._last_promotion_time < 10.0:
            return
        self._last_promotion_time = now

        if not self._hippocampus:
            return

        for mid, sal in list(self._salience.items()):
            if sal < 0.7:
                continue
            mem = self._hippocampus.get(mid)
            if mem and hasattr(mem, "long_term") and not mem.long_term:
                mem.long_term = True
                mem.consolidated_at = now

    def _get_relevant_memories(self, current: Percept | None) -> list[dict]:
        """Get memories relevant to current context.

        Returns list of dicts with 'source', 'salience', 'content' keys
        for consumption by StructuredContext and ExecAgent formatting.
        Includes prediction context from FORMING entries.
        """
        if not self._hippocampus:
            return []

        if current is None:
            # No context — return most salient tracked memories
            sorted_ids = sorted(self._salience.items(), key=lambda x: x[1], reverse=True)
            results = []
            for mid, sal in sorted_ids[: self._context_window]:
                mem = self._hippocampus.get(mid)
                if mem:
                    results.append(self._memory_to_context_item(mem, sal))
            return results

        # Stage 1: Keyword similarity via AssociationIndex
        # Lemmatize query to improve matching (e.g. "grasping" → "grasp")
        raw_query = current.raw_transcript_text or str(current.detections)
        tokens = normalize_tokens(raw_query)
        query = " ".join(tokens) if tokens else raw_query
        keyword_similar = self._association_index.find_similar(query, top_k=self._context_window)

        # Stage 2: Spreading activation from top keyword matches
        seed_ids = [mid for mid, _ in keyword_similar[:3]]
        graph_related: dict[str, float] = {}
        if seed_ids and self._hippocampus:
            try:
                associated = self._hippocampus.recall_associated(
                    seed_ids=seed_ids,
                    max_depth=3,
                    decay=0.5,
                )
                for mem, activation in associated:
                    graph_related[mem.id] = activation
            except Exception:
                pass

        # Combine and deduplicate
        seen: set[str] = set()
        combined: list[tuple[str, float]] = []

        # Include forming pool entries (current episode context) with boost
        for _run_id, entry in self._forming_pool.items():
            hid = getattr(entry, "_hippocampus_id", None)
            if hid and hid not in seen:
                seen.add(hid)
                combined.append((hid, self._salience.get(hid, 0.5) + 1.0))

        for mid, score in keyword_similar:
            if mid not in seen:
                seen.add(mid)
                sal = self._salience.get(mid, 0.5)
                combined.append((mid, score * 0.5 + sal * 0.5))

        for mid, activation in graph_related.items():
            if mid not in seen:
                seen.add(mid)
                sal = self._salience.get(mid, 0.5)
                combined.append((mid, activation * 0.4 + sal * 0.6))

        combined.sort(key=lambda x: x[1], reverse=True)

        results = []
        for mid, score in combined[: self._context_window]:
            mem = self._hippocampus.get(mid)
            if mem:
                item = self._memory_to_context_item(mem, score)
                # Attach prediction context from forming pool if applicable
                for entry in self._forming_pool.values():
                    if getattr(entry, "_hippocampus_id", None) == mid and entry.predicted_outcomes:
                        item["predictions"] = [
                            {
                                "tool": p.tool,
                                "success": p.success,
                                "goal": p.goal,
                                "confidence": p.confidence,
                                "math_context": ([m.to_dict() for m in p.math_context] if p.math_context else None),
                            }
                            for p in entry.predicted_outcomes
                        ]
                        item["prediction_confidence"] = entry.prediction_confidence
                        break
                results.append(item)
        return results

    @staticmethod
    def _memory_to_context_item(record: Any, salience: float) -> dict:
        """Convert a hippocampus record to the dict format ExecAgent expects."""
        source = "episodic"
        content = {}
        try:
            if hasattr(record, "context") and record.context:
                content["goal"] = getattr(record.context, "active_goal", None)
            if hasattr(record, "action") and record.action:
                content["action"] = getattr(record.action, "tool_name", None)
                source = "action"
            if hasattr(record, "outcome") and record.outcome:
                content["success"] = getattr(record.outcome, "success", None)
                source = "goal_outcome" if content.get("success") is not None else source
            if hasattr(record, "perception") and record.perception:
                transcript = getattr(record.perception, "transcript", None)
                if transcript:
                    content["transcript"] = transcript
                    source = "percept"
        except Exception:
            content = {"summary": str(record)}
        return {"source": source, "salience": salience, "content": content}

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

    # ── Working notes ──────────────────────────────────────────────────────

    WORKING_NOTES_MAX_CHARS = 2000

    def _read_working_notes(self) -> str:
        """Read .maxim_workspace/notes/context.md if it exists.

        Returns the file contents (capped at WORKING_NOTES_MAX_CHARS).
        Appends a truncation warning if the file exceeds the cap so the
        LLM knows to prune.
        """
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
    _WORKSPACE_EXCLUDE = frozenset(
        {
            os.path.join("plans", "ACTIVE_PLAN.md"),
            os.path.join("plans", "history.md"),
            os.path.join("notes", "context.md"),
        }
    )

    def _scan_workspace_files(self) -> list[dict]:
        """Scan .maxim_workspace/ for user-facing artifacts.

        Excludes plan system files (ACTIVE_PLAN.md, history.md) and
        notes/context.md (already injected verbatim via working notes).
        Returns up to WORKSPACE_FILE_CAP most recently modified files.
        """
        if not self._workspace_path or not os.path.isdir(self._workspace_path):
            return []

        entries: list[dict] = []
        for dirpath, _dirs, filenames in os.walk(self._workspace_path):
            # Skip archive directories
            if "history_archive" in dirpath:
                continue
            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, self._workspace_path)

                # Skip excluded files
                if rel_path in self._WORKSPACE_EXCLUDE:
                    continue
                # Skip hidden and temp files
                if fname.startswith(".") or fname.endswith(".tmp"):
                    continue

                try:
                    stat = os.stat(full_path)
                    entries.append(
                        {
                            "path": rel_path,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                        }
                    )
                except OSError:
                    continue

        # Sort by most recently modified, cap
        entries.sort(key=lambda e: e["modified"], reverse=True)
        return entries[: self.WORKSPACE_FILE_CAP]

    # Shared thread pool for parallel memory queries (4 workers = 4 query types)
    _context_pool: Any = None

    def build_context(self, persist_snapshot: bool = False) -> StructuredContext:
        """Build structured context for goal proposal.

        Memory-intensive queries (hippocampus recall, knowledge context,
        concept context, causal predictions) run in parallel threads for
        ~3-4x speedup over sequential execution.

        Args:
            persist_snapshot: If True, write context snapshot to shared outputs
        """
        from concurrent.futures import ThreadPoolExecutor

        # Snapshot mutable state under the lock, then release for parallel queries
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
                available_tools.extend(
                    [
                        "read_data_file",
                        "read_sandbox_file",
                        "write_sandbox_file",
                        "execute_sandbox_script",
                        "list_other_outputs",
                        "write_shared_output",
                    ]
                )

            det_objects = self._extract_detected_objects(recent)
            det_people = self._extract_detected_people(recent)

            # Snapshot cheap/fast fields synchronously
            sync_fields = dict(
                timestamp=time.time(),
                current_percept=current,
                active_goal=self._active_goal_description,
                active_goal_sub_goals=list(self._active_sub_goals),
                mode=self._mode,
                recent_percepts=recent[-self._context_window :],
                recent_outcomes=list(self._recent_outcomes),
                detected_objects=det_objects,
                detected_people=det_people,
                detected_speech=self._extract_speech(recent),
                recent_logs=self._abstraction.get_recent(n=15),
                goal_history=self._abstraction.get_by_event("goal_proposed", n=5),
                cli_inputs=list(self._cli_inputs),
                comms_messages=list(self._comms_messages),
                available_environments=available_tools,
                statistical_context=self._latest_statistical_summary,
                active_pattern_count=self._active_pattern_count,
                statistical_suggestions=self._statistical_suggestions,
                root_goal=self.ROOT_GOAL,
                working_notes=self._read_working_notes(),
                workspace_files=self._scan_workspace_files(),
                plan_progress=self._build_plan_progress(),
            )

        # --- Parallel memory queries (outside self._lock) ---
        # Each method acquires its own subsystem's lock internally.
        if MemoryAgent._context_pool is None:
            MemoryAgent._context_pool = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="memctx",
            )
        pool = MemoryAgent._context_pool

        fut_memories = pool.submit(self._get_relevant_memories, current)
        fut_knowledge = pool.submit(self._build_knowledge_context)
        fut_concepts = pool.submit(self._build_concept_context, det_objects, det_people)
        fut_causal = pool.submit(self._build_causal_context)

        # Collect results (each ~5-15ms, running in parallel → ~15ms total)
        sync_fields["relevant_memories"] = fut_memories.result(timeout=2.0)
        sync_fields["knowledge_context"] = fut_knowledge.result(timeout=2.0)
        sync_fields["concept_context"] = fut_concepts.result(timeout=2.0)
        sync_fields["causal_context"] = fut_causal.result(timeout=2.0)

        context = StructuredContext(**sync_fields)

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

        # Inject provenance context (P7)
        if self._collector and self._collector.verbosity >= 1:
            try:
                from maxim.provenance.render import render_trace
                from maxim.provenance.types import ProvenanceVerbosity

                recent_traces = self._collector.recent_traces(limit=3)
                if recent_traces:
                    context.provenance_context = "\n".join(
                        render_trace(t, verbosity=ProvenanceVerbosity.COMPACT) for t in recent_traces
                    )
            except Exception as e:
                log_swallowed_exception(e, operation="provenance_context")

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
            phases_completed=sum(1 for p in plan.phases if p.status == PhaseStatus.COMPLETED),
            phases_failed=sum(1 for p in plan.phases if p.status == PhaseStatus.FAILED),
            energy_utilization=(current.energy_budget.utilization if current and current.energy_budget else {}),
            is_replanning=plan.status.name == "REPLANNING",
            replan_count=len(plan.replan_history),
        )

    def _build_knowledge_context(self) -> list[dict]:
        """Merge knowledge from ATL + Angular Gyrus into unified context.

        Uses cross-layer spreading activation (via MemoryHub.recall_with_knowledge)
        when recent episodic seeds are available, so returned knowledge is
        contextually relevant rather than just highest-confidence.  Falls back
        to independent top-N queries when no seeds exist.

        Returns a list of knowledge entries ranked by relevance, capped at 8.
        Each entry has: concept_name, definition, category, confidence,
        source_layer, provenance, relationships.
        """
        if self._memory_hub is None:
            return []

        hub = self._memory_hub
        entries: list[dict] = []

        # Collect episodic seeds from forming pool for cross-layer activation
        seed_ids = [hid for entry in self._forming_pool.values() if (hid := getattr(entry, "_hippocampus_id", None))]

        # --- Cross-layer path: spread from episodic seeds across all layers ---
        if seed_ids:
            try:
                cross_results = hub.recall_with_knowledge(
                    seed_ids=seed_ids,
                    start_layer="hippocampus",
                    limit=10,
                )
                # Process ATL hits from cross-layer activation
                atl = getattr(hub, "atl", None)
                for record_id, activation in cross_results.get("atl", []):
                    if atl is None:
                        continue
                    try:
                        concepts = atl.recall(name=record_id, limit=1)
                        if not concepts:
                            continue
                        concept = concepts[0]
                        rels = self._get_concept_relationships(atl, concept.id)
                        entries.append(
                            {
                                "concept_name": concept.name,
                                "definition": getattr(concept, "definition", ""),
                                "category": concept.category,
                                "confidence": concept.confidence,
                                "source_layer": "atl",
                                "provenance": getattr(concept, "provenance", "").name
                                if hasattr(getattr(concept, "provenance", None), "name")
                                else str(getattr(concept, "provenance", "")),
                                "relationships": rels,
                                "relevance": activation * 0.6 + concept.confidence * 0.4,
                            }
                        )
                    except Exception:
                        pass

                # Process Angular Gyrus hits from cross-layer activation
                ag = getattr(hub, "angular_gyrus", None)
                for record_id, activation in cross_results.get("angular_gyrus", []):
                    if ag is None:
                        continue
                    try:
                        records = ag.recall(name=record_id, limit=1)
                        if not records:
                            continue
                        record = records[0]
                        entries.append(
                            {
                                "concept_name": record.name,
                                "definition": record.verbal,
                                "category": f"math:{record.category.name}",
                                "confidence": record.confidence,
                                "source_layer": "angular_gyrus",
                                "provenance": record.source,
                                "relationships": [],
                                "relevance": activation * 0.6 + record.confidence * 0.4,
                            }
                        )
                    except Exception:
                        pass
            except Exception:
                pass  # Fall through to independent queries below

        # --- Fallback: independent top-N queries when no seeds or cross-layer empty ---
        if not entries:
            if getattr(hub, "atl", None) is not None:
                try:
                    concepts = hub.atl.recall(limit=5, min_confidence=0.5)
                    for concept in concepts:
                        rels = self._get_concept_relationships(hub.atl, concept.id)
                        entries.append(
                            {
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
                            }
                        )
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
                        entries.append(
                            {
                                "concept_name": record.name,
                                "definition": record.verbal,
                                "category": f"math:{record.category.name}",
                                "confidence": record.confidence,
                                "source_layer": "angular_gyrus",
                                "provenance": record.source,
                                "relationships": [],
                                "relevance": record.confidence * 0.8,
                            }
                        )
                except Exception:
                    pass

        # Rank by relevance, cap at 8 entries total
        entries.sort(key=lambda e: e.get("relevance", 0), reverse=True)
        return entries[:8]

    @staticmethod
    def _get_concept_relationships(atl: Any, concept_id: str) -> list[dict]:
        """Extract relationships for a concept from ATL."""
        rels: list[dict] = []
        try:
            rel_pairs = atl.find_by_relationship(concept_id, limit=3)
            for target_id, r in rel_pairs:
                rels.append(
                    {
                        "type": r.relationship_type,
                        "target": target_id,
                        "weight": r.weight,
                    }
                )
        except Exception:
            pass
        return rels

    def _build_causal_context(self) -> list[dict]:
        """Build causal prediction context from NAc for recent tools/actions.

        Queries NAc.predict() for each recent tool outcome to surface
        learned expectations like "last time you ran X, outcome was Y".
        """
        if self._memory_hub is None:
            return []
        nac = getattr(self._memory_hub, "nac", None)
        if nac is None:
            return []

        entries: list[dict] = []
        seen_sigs: set[str] = set()

        # Query predictions for recent tool outcomes + active goal
        for outcome in list(self._recent_outcomes)[-5:]:
            tool = outcome.get("tool_name") or outcome.get("goal_id")
            if not tool or tool in seen_sigs:
                continue
            seen_sigs.add(tool)

            ctx = {"goal": self._active_goal_description or ""}
            try:
                prediction = nac.predict(
                    event_type="tool" if "tool_name" in outcome else "goal",
                    event_signature=tool,
                    context=ctx,
                )
                if prediction and prediction.confidence >= 0.3:
                    entries.append(
                        {
                            "event": prediction.event_signature,
                            "outcome": prediction.predicted_outcome,
                            "valence": prediction.predicted_valence.value,
                            "confidence": round(prediction.confidence, 2),
                            "context_match": round(prediction.context_match, 2),
                        }
                    )
            except Exception:
                pass

        # Sort by confidence descending, cap at 5
        entries.sort(key=lambda e: e["confidence"], reverse=True)
        return entries[:5]

    def _build_concept_context(
        self,
        det_objects: list[dict],
        det_people: list[dict],
    ) -> list[dict]:
        """Build concept context from current percept via ConceptContextBuilder.

        Extracts label strings from detection dicts and delegates to
        MemoryHub.build_concept_context().
        """
        if self._memory_hub is None:
            return []
        # Extract label strings from detection dicts
        object_labels = [
            d.get("label", d.get("class_name", "")) for d in det_objects if d.get("label") or d.get("class_name")
        ]
        people_labels = [d.get("label", d.get("class_name", "person")) for d in det_people]
        return self._memory_hub.build_concept_context(
            detected_objects=object_labels,
            detected_people=people_labels,
            active_goal=self._active_goal_description,
        )

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

    def _persist_memories_to_disk(self, share_to_outputs: bool = False) -> None:
        """Persist salience metadata to disk.

        After Phase 0, Hippocampus handles its own persistence.
        MemoryAgent only persists lightweight salience/decay metadata
        which is rebuilt each session from percepts. This is optional.
        """
        # Salience metadata is ephemeral — rebuilt from percepts each session.
        # Hippocampus persistence handles the actual memory data.
        pass

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
