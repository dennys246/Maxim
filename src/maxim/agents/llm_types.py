"""Data types and protocols for the LLM worker subsystem.

Pure data classes with zero behavioral coupling. Other modules can import
types without pulling in the full LLM machinery.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from maxim.agents.autonomy import AutonomyLevel

if TYPE_CHECKING:
    from maxim.agents.bus import StructuredContext


# ─────────────────────────────────────────────────────────────────────────────
# Protocols
# ─────────────────────────────────────────────────────────────────────────────


class LLMBackend(Protocol):
    """Protocol for LLM backends that can generate JSON responses."""

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        *,
        provider_hint: str | None = None,
        request_context: dict[str, Any] | None = None,
        system_override: str | None = None,
    ) -> dict[str, Any] | None:
        """Generate a JSON response from the LLM."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────


class LLMAttemptState(str, Enum):
    """Lifecycle of the most recently submitted LLM worker job.

    This state is scoped to one ``LLMWorker``. It remains ``RUNNING`` through
    response parsing, then ``COMPLETED`` from result publication until the
    loop consumes that result. It is therefore safe for deciding whether one
    planning turn is still alive; the process-global provider-call registry
    is not.
    """

    NONE = "none"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CONSUMED = "consumed"
    MISSING = "missing"


@dataclass
class ModeInfo:
    """Minimal mode information for LLM prompts."""

    name: str
    goal: str
    context_prompt: str
    forbidden_tools: set[str] = field(default_factory=set)
    allowed_tools: set[str] = field(default_factory=set)  # Empty = all non-forbidden
    max_initiative: float = 1.0

    # Capability flags
    can_access_filesystem: bool = True
    can_access_network: bool = True

    # Response configuration for dynamic context windows
    max_response_tokens: int = 512
    context_window_tokens: int = 2048
    response_format: str = "conversational"

    # Prompt-assembly hint copied from ModeDefinition. When False, prompt
    # assembly emits the full tool manifest with descriptions (autonomous
    # modes). When True, the learned-relevance filter trims the manifest
    # (interactive passive mode).
    uses_tool_relevance_filter: bool = False

    def get_available_tools(self, all_tools: set[str]) -> set[str]:
        """Get tools available in this mode."""
        if self.allowed_tools:
            available = self.allowed_tools - self.forbidden_tools
        else:
            available = all_tools - self.forbidden_tools

        if not self.can_access_filesystem:
            available -= {"read_file", "write_file", "list_directory", "execute_file"}
        if not self.can_access_network:
            available -= {"internet_search", "http_fetch"}

        return available


@dataclass(order=True)
class LLMRequest:
    """Request for LLM processing."""

    # Fields used for ordering (priority queue)
    sort_index: tuple = field(init=False, repr=False)

    request_id: str = field(compare=False)
    context: StructuredContext = field(compare=False)
    mode: ModeInfo = field(compare=False)
    autonomy_level: AutonomyLevel = field(compare=False)
    internet_access: bool = field(compare=False)
    internet_policy_summary: str = field(compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)
    priority: int = field(default=0, compare=False)  # Higher = more urgent

    # Tool information for tool-aware prompts
    available_tools: set[str] = field(default_factory=set, compare=False)
    tool_descriptions: dict[str, str] = field(default_factory=dict, compare=False)
    surfaced_tools: list[str] = field(
        default_factory=list, compare=False
    )  # Tools shown in prompt (for learned index decay signal)

    # Context pool summary for accumulated observations
    context_pool_text: str = field(default="", compare=False)

    # Agent states for multi-agent awareness
    agent_states: list[dict[str, Any]] = field(default_factory=list, compare=False)

    # Recent action outcomes for learning
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list, compare=False)

    # Whether to use tool-aware prompting (vs simple answer-only)
    use_tool_prompting: bool = field(default=False, compare=False)

    # The user input that triggered this request (for conversation history)
    triggering_input: str = field(default="", compare=False)

    # Lane hint for WorkerPool routing
    lane: str = field(default="", compare=False)

    # Conversation history text for context
    conversation_history_text: str = field(default="", compare=False)

    # Pending modification request (action that user wants to revise)
    pending_modification: dict[str, Any] | None = field(default=None, compare=False)

    # Processing state (awake or sleep)
    is_sleeping: bool = field(default=False, compare=False)

    # Pre-fetched context from speculative pre-fetcher
    prefetch_context: str = field(default="", compare=False)
    skip_exploration: bool = field(default=False, compare=False)

    # Per-request timeout override (seconds); None means use worker default
    timeout_override: float | None = field(default=None, compare=False)

    # Active protocol context (re-injected each submission, never summarized)
    protocol_context: str = field(default="", compare=False)

    # Acting Coach config for affordance exploration meta-prompting (B3)
    acting_coach: Any | None = field(default=None, compare=False)

    # SEM entity spec dict for entity context injection (E2)
    entity_spec: dict[str, Any] | None = field(default=None, compare=False)

    # When True, the agent owns a SEM body and is running in an embodied arc.
    # Gates prompt sections that would otherwise tell the LLM to call the
    # conversational ``respond`` / ``speak`` tools — those are deregistered
    # for embodied arcs (the body exposes its own ``<body>_respond`` /
    # ``<body>_use`` tools), so emitting the conversational guidance produces
    # the silent `Tool not registered: 'respond'` loop documented in
    # docs/plans/archive/cradle_activation_fixes.md (Finding B). Set by the producer
    # (LLMWorker) at construction; consumers (prompt_builder) read it.
    is_embodied: bool = field(default=False, compare=False)

    # EXPERIMENTAL OPT-IN — pretrained-LLM hallucination mitigation. Names of
    # tools the model previously called that don't exist for this agent.
    # Surfaced as a negative-instruction prompt section. Gated by env
    # MAXIM_TOOL_FAILURE_HINTS (default OFF after E4 validation 2026-05-09
    # showed no benefit and possible backfire on qwen2.5-14B; n=6 per arm).
    # Set MAXIM_TOOL_FAILURE_HINTS=1 to enable for further experimentation.
    # Must remain OFF for grounded language acquisition Phase 0/1 — see
    # docs/plans/grounded_language_acquisition.md.
    failed_tools: list[str] = field(default_factory=list, compare=False)

    def __post_init__(self):
        # Sort by negative priority (higher priority first), then by timestamp
        self.sort_index = (-self.priority, self.timestamp)


@dataclass(frozen=True)
class LLMProposal:
    """Result from LLM processing."""

    request_id: str
    action: dict[str, Any] | None
    reasoning: str
    strategy_used: str | None
    confidence: float
    mode_goal_achieved: bool
    citations: list[dict[str, str]] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    error: str | None = None

    # Multi-step action support (sequential - one after another)
    next_actions: list[dict[str, Any]] = field(default_factory=list)

    # Parallel actions - execute ALL together before next LLM call
    # Used for batched exploration: glob + read_file calls run together
    parallel_actions: list[dict[str, Any]] = field(default_factory=list)

    # The user input that triggered this response (for conversation history)
    triggering_input: str = ""

    # Planning liveness (bugs ledger D13): the original LLMRequest so the
    # agent loop can retry the exact turn after any non-executable outcome.
    # Parse failures are resubmitted byte-identically; a well-formed proposal
    # naming an unavailable tool adds explicit corrective feedback first.
    # Runtime-ephemeral — never persisted, never crosses a wire.
    original_request: Any | None = None

    # Planning mode: human-readable plan (shown to user for approval)
    plan_text: str | None = None
    # Whether this proposal requires user approval before execution
    requires_approval: bool = False

    # PFC deliberation: whether the LLM is ready to act.
    # True (default) = one-shot / backward compatible.
    # False = LLM wants more context; cycle feeds reasoning back through
    # bio-enrichment for another round.
    ready_to_act: bool = True

    # Substrate-primary: EC interoception cluster id active at proposal
    # time. Captured by ``propose_via_substrate`` (Phase 0 / Track 2 of
    # grounded_language_acquisition.md) so the outcome-recording path
    # can credit/penalise the right ``(agent, cluster, tool)`` triple
    # in ``NAc._cluster_reward_bias`` after the action executes.
    # ``None`` for every LLM-primary proposal and for substrate-primary
    # ticks where no cluster was active (no sensors / no encoder).
    # Closes G4 from Roy-0: Track 2 wired ``current_cluster_id`` into
    # ``recommend_action`` for *selection* but deliberately deferred the
    # ``record_outcome`` plumbing for *learning*. Without this field the
    # cluster_id is captured at proposal time and lost before outcome.
    # LEGACY single-cluster alias since the extero/intero seam: equals
    # ``clusters["interoception"]`` when that channel encoded. Kept
    # through 1.x for single-cluster consumers.
    cluster_id: str | None = None

    # Extero/intero seam: the FULL per-modality active-cluster set at
    # proposal time, ``{modality_tag: ec_cluster_id}`` (e.g.
    # ``{"interoception": I, "audio": A}``) — one entry per ModalityChannel
    # that encoded this tick. The outcome path routes credit by source
    # across this set (drive-relief → interoception, operant/direction →
    # the exteroceptive cluster, generic tool-success → interoception
    # ONLY). ``None`` for LLM-primary proposals and pre-seam producers —
    # consumers fold ``cluster_id`` in as the interoception entry.
    clusters: dict[str, str] | None = None

    def get_all_actions(self) -> list[dict[str, Any]]:
        """Get the primary action followed by any next_actions."""
        actions = []
        if self.action:
            actions.append(self.action)
        actions.extend(self.next_actions)
        return actions

    def get_parallel_actions(self) -> list[dict[str, Any]]:
        """Get all parallel actions to execute together."""
        actions = []
        if self.action:
            actions.append(self.action)
        actions.extend(self.parallel_actions)
        return actions
