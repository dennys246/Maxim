"""MemoryAgent: Core memory system with salience, associations, and persistence.

THIS IS THE CENTRAL RESEARCH FOCUS OF THE PROJECT.

The MemoryAgent maintains salient memories, builds associations via similarity,
and provides structured context for goal proposal.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Any

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
)
from maxim.agents.output_mixin import AgentOutputMixin
from maxim.utils.logging import log_swallowed_exception
from maxim.utils.structured_logging import get_abstraction_buffer


class AssociationIndex:
    """Index for fast similarity-based memory retrieval.

    After Phase 0 unification, this index stores memory IDs and keywords
    only — full records are resolved from Hippocampus on demand.

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
        self._memory_keywords: dict[str, set[str]] = {}  # mid → keywords
        self._embedding_model = embedding_model
        self._embedder = None  # Lazy init
        self._lock = threading.Lock()
        # Phase 3e: LSH-backed context similarity index
        self._context_index: Any = None  # SimilarityIndex, set via set_context_index()

    def set_context_index(self, index: Any) -> None:
        """Wire a SimilarityIndex for LSH-backed recall (Phase 3e)."""
        self._context_index = index

    def add_by_id(self, memory_id: str, content: Any) -> None:
        """Index a memory by keywords extracted from content."""
        with self._lock:
            keywords = self._extract_keywords(content)
            self._memory_keywords[memory_id] = keywords
            for kw in keywords:
                self._keyword_index[kw].add(memory_id)
            # Also register in LSH index if available
            if self._context_index is not None:
                text = content if isinstance(content, str) else " ".join(
                    str(v) for v in (content.values() if isinstance(content, dict) else [str(content)])
                    if v
                )
                self._context_index.register(memory_id, text)

    def remove(self, memory_id: str) -> None:
        """Remove memory from index."""
        with self._lock:
            kws = self._memory_keywords.pop(memory_id, set())
            for kw in kws:
                self._keyword_index[kw].discard(memory_id)
            if self._context_index is not None:
                self._context_index.remove(memory_id)

    def find_similar(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find similar memory IDs by similarity.

        Uses LSH (Phase 3e) when available, falls back to keyword Jaccard.
        Returns (memory_id, score) tuples sorted by score descending.
        """
        with self._lock:
            # Phase 3e: Prefer LSH similarity when index is populated
            if (
                self._context_index is not None
                and self._context_index.signatures
            ):
                query_text = str(query) if not isinstance(query, str) else query
                lsh_results = self._context_index.query_similar(
                    query_text, min_similarity=0.3
                )
                if lsh_results:
                    return lsh_results[:top_k]

            # Fallback: keyword Jaccard similarity
            query_keywords = self._extract_keywords(query)

            candidates: dict[str, float] = {}
            for kw in query_keywords:
                for mid in self._keyword_index.get(kw, set()):
                    if mid not in candidates:
                        candidates[mid] = 0.0
                    mem_kw = self._memory_keywords.get(mid, set())
                    if mem_kw:
                        intersection = len(query_keywords & mem_kw)
                        union = len(query_keywords | mem_kw)
                        if union > 0:
                            candidates[mid] = max(candidates[mid], intersection / union)

            sorted_results = sorted(
                candidates.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            return sorted_results[:top_k]

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


class MemoryAgent(Agent, AgentOutputMixin):
    """Orchestrates memory retrieval and salience scoring.

    THIS IS THE CORE EXPLORATION FOCUS OF THE PROJECT.

    After Phase 0 unification, Hippocampus owns all memory storage and
    associative graphs. MemoryAgent owns salience scoring, relevance
    ranking, and context building. They are complementary roles.

    Responsibilities:
    - Score percept salience and track salience decay per memory
    - Capture memories via Hippocampus (single store)
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

        # Hippocampus reference (injected via wire_memory_hub)
        self._hippocampus: Any | None = None

        # Salience tracking — lightweight metadata for hippocampus memories
        self._salience: dict[str, float] = {}  # memory_id → current salience
        self._decay_rates: dict[str, float] = {}  # memory_id → decay rate
        self._recent_ids: deque[str] = deque(maxlen=max_short_term)  # Bounded window

        # Recency buffers (not memory storage — just prompt context)
        self._recent_percepts: deque[Percept] = deque(maxlen=context_window)
        self._recent_outcomes: deque[dict] = deque(maxlen=context_window)
        self._cli_inputs: deque[str] = deque(maxlen=20)
        self._comms_messages: deque[dict] = deque(maxlen=20)

        # Association index (keyword-based, operates on hippocampus memory IDs)
        self._association_index = AssociationIndex(
            embedding_model="all-MiniLM-L6-v2" if enable_embeddings else None
        )

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

        # Workspace path for reading working notes
        self._workspace_path = workspace_path

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

            # Track comms messages (SMS, voice, etc.)
            if percept.source.startswith("comms:") and percept.content:
                self._comms_messages.append({
                    "direction": "inbound",
                    "content": percept.content,
                    "channel": (percept.metadata or {}).get("channel", ""),
                    "sender": (percept.metadata or {}).get("sender", ""),
                    "timestamp": percept.timestamp,
                })

            # Capture via Hippocampus if salient
            if percept.salience > self._salience_threshold or percept.has_maxim_keyword:
                salience = self._compute_memory_salience(percept)
                decay_rate = 0.05 if percept.has_maxim_keyword else 0.1
                content = {
                    "source": percept.source,
                    "transcript": percept.raw_transcript_text or percept.content,
                    "has_maxim": percept.has_maxim_keyword,
                    "detections_count": len(percept.detections),
                }
                self._add_memory(content, salience, decay_rate, "percept", percept)

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
        """
        if not self._hippocampus:
            return []

        if current is None:
            # No context — return most salient tracked memories
            sorted_ids = sorted(
                self._salience.items(), key=lambda x: x[1], reverse=True
            )
            results = []
            for mid, sal in sorted_ids[: self._context_window]:
                mem = self._hippocampus.get(mid)
                if mem:
                    results.append(self._memory_to_context_item(mem, sal))
            return results

        # Stage 1: Keyword similarity via AssociationIndex
        query = current.raw_transcript_text or str(current.detections)
        keyword_similar = self._association_index.find_similar(
            query, top_k=self._context_window
        )

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
                results.append(self._memory_to_context_item(mem, score))
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
    _WORKSPACE_EXCLUDE = frozenset({
        os.path.join("plans", "ACTIVE_PLAN.md"),
        os.path.join("plans", "history.md"),
        os.path.join("notes", "context.md"),
    })

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
                    entries.append({
                        "path": rel_path,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    })
                except OSError:
                    continue

        # Sort by most recently modified, cap
        entries.sort(key=lambda e: e["modified"], reverse=True)
        return entries[: self.WORKSPACE_FILE_CAP]

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
