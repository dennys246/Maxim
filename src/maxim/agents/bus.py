"""Message bus and data types for agent communication.

Defines the message types exchanged between agents and the pub/sub bus
for decoupled communication.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, TypeVar

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
    """Memory storage tier."""

    SHORT_TERM = "short"  # Fast decay, recent context
    LONG_TERM = "long"  # Slow decay, consolidated knowledge


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

    # Generic content and metadata (for comms, archives, preemption context, etc.)
    content: str | None = None
    metadata: dict[str, Any] | None = None

    # Raw data for downstream
    raw_transcript_text: str | None = None
    maxim_runtime: dict[str, Any] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes - Memory
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MemoryItem:
    """A single item in salient memory."""

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
            self._cached_memory_id = hashlib.sha256(
                f"{self.timestamp}:{content_str}".encode()
            ).hexdigest()[:16]
        return self._cached_memory_id

    def access(self) -> None:
        """Mark memory as accessed (refreshes last_accessed)."""
        self.last_accessed = time.time()
        self.access_count += 1

    def should_promote(
        self, access_threshold: int = 3, salience_threshold: float = 0.7
    ) -> bool:
        """Check if short-term memory should be promoted to long-term."""
        if self.tier != MemoryTier.SHORT_TERM:
            return False
        return self.access_count >= access_threshold or self.salience >= salience_threshold

    def should_evict_long_term(self, max_age_seconds: float) -> bool:
        """Check if long-term memory should be evicted."""
        if self.tier != MemoryTier.LONG_TERM:
            return False
        return (time.time() - self.last_accessed) > max_age_seconds


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

    # Salient memories (filtered by relevance)
    recent_percepts: list[Percept] = field(default_factory=list)
    recent_outcomes: list[dict] = field(default_factory=list)
    relevant_memories: list[MemoryItem] = field(default_factory=list)

    # Detected patterns (from YOLO, NOT raw images)
    detected_objects: list[dict] = field(default_factory=list)
    detected_people: list[dict] = field(default_factory=list)
    detected_speech: list[str] = field(default_factory=list)

    # Abstraction stream data (for LLM context)
    recent_logs: list[dict] = field(default_factory=list)
    goal_history: list[dict] = field(default_factory=list)
    cli_inputs: list[str] = field(default_factory=list)
    available_environments: list[str] = field(default_factory=list)

    # Conversation history (user input + LLM response pairs)
    # Each entry: {"user": str, "assistant": str, "timestamp": float}
    conversation_history: list[dict] = field(default_factory=list)

    # Statistical context (from StatisticianAgent via bus)
    statistical_context: str = ""
    active_pattern_count: int = 0

    # Knowledge context (from ATL semantic memory + AG pattern memory)
    # Each entry: {"concept_name", "definition", "category", "confidence",
    #              "source_layer", "provenance", "relationships": [...]}
    knowledge_context: list[dict] = field(default_factory=list)

    # Root goal reminder
    root_goal: str = "Understand reality and help people."

    # Plan progress (from PlanManager when a long-horizon plan is active)
    plan_progress: PlanProgressContext | None = None


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
            sg.id
            for sg in self.sub_goals
            if sg.status in (SubGoalStatus.COMPLETED, SubGoalStatus.SKIPPED)
        }

        # Find all executable sub-goals
        executable = [
            sg
            for sg in self.sub_goals
            if sg.status in (SubGoalStatus.PENDING, SubGoalStatus.BLOCKED)
            and sg.can_execute(completed_ids)
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
        return all(
            sg.status in (SubGoalStatus.COMPLETED, SubGoalStatus.SKIPPED)
            for sg in self.sub_goals
        )

    def all_failed(self) -> bool:
        """Check if goal failed due to sub-goal failures."""
        return any(
            sg.status == SubGoalStatus.FAILED and not sg.should_retry()
            for sg in self.sub_goals
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
class StatisticalSummary:
    """Periodic summary from StatisticianAgent for context building."""

    timestamp: float
    summary: str  # Actionable natural language (for StructuredContext)
    active_patterns: int  # Count of CONFIRMED_PATTERN metrics
    metrics_monitored: int  # Total metrics being tracked


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
                self._incoming[edge.target] = [
                    e for e in self._incoming[edge.target] if e.source != node_id
                ]
            del self._outgoing[node_id]

            for edge in self._incoming[node_id]:
                self._outgoing[edge.source] = [
                    e for e in self._outgoing[edge.source] if e.target != node_id
                ]
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

    def spreading_activation(
        self,
        source_ids: list[str],
        initial_activation: float = 1.0,
        decay: float = 0.5,
        threshold: float = 0.1,
        max_depth: int = 3,
    ) -> dict[str, float]:
        """Spread activation from source nodes through association edges."""
        activations: dict[str, float] = {}
        visited_at_depth: dict[str, int] = {}

        queue: deque[tuple[str, float, int]] = deque()
        with self._lock:
            for source in source_ids:
                if source in self._nodes:
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
                    new_activation = activation * decay * weight

                    if new_activation < threshold:
                        continue

                    if target not in activations or new_activation > activations[target]:
                        activations[target] = new_activation
                        visited_at_depth[target] = depth + 1
                        queue.append((target, new_activation, depth + 1))

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
                handler(message)
            except Exception:
                pass  # Handlers should not block the bus

    def clear(self) -> None:
        """Clear all subscriptions."""
        with self._lock:
            self._subscribers.clear()
