"""Data types and protocols for the LLM worker subsystem.

Pure data classes with zero behavioral coupling. Other modules can import
types without pulling in the full LLM machinery.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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


@dataclass
class StrategyInfo:
    """Minimal strategy information for LLM prompts."""

    name: str
    description: str
    approach_prompt: str


@dataclass(order=True)
class LLMRequest:
    """Request for LLM processing."""

    # Fields used for ordering (priority queue)
    sort_index: tuple = field(init=False, repr=False)

    request_id: str = field(compare=False)
    context: StructuredContext = field(compare=False)
    mode: ModeInfo = field(compare=False)
    autonomy_level: AutonomyLevel = field(compare=False)
    strategies: list[StrategyInfo] = field(compare=False)
    internet_access: bool = field(compare=False)
    internet_policy_summary: str = field(compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)
    priority: int = field(default=0, compare=False)  # Higher = more urgent

    # Tool information for tool-aware prompts
    available_tools: set[str] = field(default_factory=set, compare=False)
    tool_descriptions: dict[str, str] = field(default_factory=dict, compare=False)

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

    # Current strategy name (from new architecture: observe, explore, research, assist, reflect, learn)
    current_strategy: str = field(default="", compare=False)

    # Processing state (awake or sleep)
    is_sleeping: bool = field(default=False, compare=False)

    # Pre-fetched context from speculative pre-fetcher
    prefetch_context: str = field(default="", compare=False)
    skip_exploration: bool = field(default=False, compare=False)

    # Active protocol context (re-injected each submission, never summarized)
    protocol_context: str = field(default="", compare=False)

    def __post_init__(self):
        # Sort by negative priority (higher priority first), then by timestamp
        self.sort_index = (-self.priority, self.timestamp)


@dataclass
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

    # Planning mode: human-readable plan (shown to user for approval)
    plan_text: str | None = None
    # Whether this proposal requires user approval before execution
    requires_approval: bool = False

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