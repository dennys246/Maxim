"""Dedicated LLM worker thread for non-blocking inference.

The LLMWorker processes LLM requests asynchronously, ensuring the main control
loop never blocks on LLM latency. Includes fallback behaviors for when the
LLM is slow or unavailable.
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, Protocol

from maxim.agents.autonomy import AutonomyLevel
from maxim.models.language.router import CharEstimateCounter
from maxim.utils.coding_guidelines import build_coding_context, is_file_search_request
from maxim.utils.prompts import get_fallback_responses

# Optional energy tracking import
try:
    from maxim.energy.llm_tracker import LLMEnergyTracker
    _HAS_ENERGY_TRACKING = True
except ImportError:
    _HAS_ENERGY_TRACKING = False
    LLMEnergyTracker = None  # type: ignore


# Performance: Cache compiled regex patterns for phrase matching
@functools.lru_cache(maxsize=256)
def _compile_phrase_pattern(phrase: str) -> re.Pattern:
    """Compile and cache regex pattern for phrase matching."""
    return re.compile(rf"\b{re.escape(phrase)}\b")


# Simple arithmetic: "number operator number" (supports negatives, decimals, 'x' for multiply)
_ARITHMETIC_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/x^%])\s*(-?\d+(?:\.\d+)?)"
)

_SIMPLE_OPS: dict[str, Any] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "x": lambda a, b: a * b,
    "/": lambda a, b: a / b if b != 0 else float("nan"),
    "^": lambda a, b: a ** b,
    "%": lambda a, b: a % b if b != 0 else float("nan"),
}

# Unary math: "square root of 25", "sqrt 25", "cube root of 8", "25 squared"
_UNARY_MATH_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?:square\s*root\s*(?:of\s*)?|sqrt\s*|√\s*)(-?\d+(?:\.\d+)?)"), "sqrt"),
    (re.compile(r"(?:cube\s*root\s*(?:of\s*)?|cbrt\s*|∛\s*)(-?\d+(?:\.\d+)?)"), "cbrt"),
    (re.compile(r"(-?\d+(?:\.\d+)?)\s*squared"), "squared"),
    (re.compile(r"(-?\d+(?:\.\d+)?)\s*cubed"), "cubed"),
]

# Trailing binary op after unary: "... plus 3", "... + 3", "... minus 2", "... times 4"
_TRAILING_OP_PATTERN = re.compile(
    r"(?:plus|\+)\s*(-?\d+(?:\.\d+)?)|"
    r"(?:minus|-)\s*(-?\d+(?:\.\d+)?)|"
    r"(?:times|multiplied\s*by|\*|x)\s*(-?\d+(?:\.\d+)?)|"
    r"(?:divided\s*by|/)\s*(-?\d+(?:\.\d+)?)"
)

if TYPE_CHECKING:
    from maxim.agents.bus import Percept, StructuredContext
    from maxim.runtime.worker_pool import WorkerPool


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Foundational Documents (Constitution & Agent Rules)
# ─────────────────────────────────────────────────────────────────────────────

_foundational_context_cache: str | None = None


def _load_foundational_context() -> str:
    """Load CONSTITUTION.md and AGENTS.md as foundational LLM context.

    Caches the result to avoid repeated file I/O.
    Returns a condensed version suitable for prompts.
    """
    global _foundational_context_cache
    if _foundational_context_cache is not None:
        return _foundational_context_cache

    from pathlib import Path

    parts: list[str] = []

    # Find repo root (look for CONSTITUTION.md or AGENTS.md)
    current = Path(__file__).resolve()
    repo_root = None
    for parent in current.parents:
        if (parent / "CONSTITUTION.md").exists() or (parent / "AGENTS.md").exists():
            repo_root = parent
            break

    if repo_root is None:
        logger.debug("Could not find repo root for foundational documents")
        _foundational_context_cache = ""
        return ""

    # Load CONSTITUTION.md - extract key principles (condensed for prompt)
    constitution_path = repo_root / "CONSTITUTION.md"
    if constitution_path.exists():
        try:
            # Extract key constitutional principles (condensed for prompt efficiency)
            parts.append("=== CORE PRINCIPLES (from Constitution) ===")
            parts.append("Priority Order: 1) Physical Safety 2) Ethics 3) Guidelines 4) Helpfulness")
            parts.append("")
            parts.append("Hard Constraints (NEVER violate):")
            parts.append("- Never move toward a person who said 'stop' or shows distress")
            parts.append("- Never continue movement after unexpected collision")
            parts.append("- Never attempt to prevent being powered off")
            parts.append("- Never fabricate information or claim false certainty")
            parts.append("")
            parts.append("Core Values: Honesty, transparency, respect for persons, avoiding harm")
            parts.append("When uncertain: Ask rather than assume. Halt rather than proceed blindly.")
            logger.debug("Loaded constitution principles")
        except Exception as e:
            logger.warning("Failed to load CONSTITUTION.md: %s", e)

    # Load AGENTS.md - extract agent behavior rules
    agents_path = repo_root / "AGENTS.md"
    if agents_path.exists():
        try:
            # Extract key agent behavior rules (condensed for prompt efficiency)
            parts.append("")
            parts.append("=== AGENT BEHAVIOR RULES (from AGENTS.md) ===")
            parts.append("Agents THINK but do not ACT directly.")
            parts.append("")
            parts.append("You MAY: Read state, query memory, propose intents, evaluate outcomes")
            parts.append("You MAY NOT: Execute tools directly, mutate state, control execution loops")
            parts.append("")
            parts.append("Output: Structured intent (JSON), never imperative commands")
            parts.append("Coordination: Through state and decision engine, not direct agent calls")
            logger.debug("Loaded agent rules")
        except Exception as e:
            logger.warning("Failed to load AGENTS.md: %s", e)

    _foundational_context_cache = "\n".join(parts) if parts else ""
    return _foundational_context_cache


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


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Budgeter
# ─────────────────────────────────────────────────────────────────────────────


class SectionPriority(IntEnum):
    """Priority tiers for prompt sections. Lower = higher priority."""

    MANDATORY = 0   # Never dropped (instructions, user request)
    CRITICAL = 1    # Dropped only under extreme pressure (identity, tools, planning)
    IMPORTANT = 2   # Truncatable first, then dropped (conversation, context pool)
    NICE_TO_HAVE = 3  # Dropped first (foundational, mode context, speech)


@dataclass
class PromptSection:
    """A single section of the assembled prompt."""

    name: str
    content: str
    priority: SectionPriority
    token_count: int
    insertion_order: int
    truncatable: bool = False
    min_tokens: int = 0
    truncate_fn: Callable[[str, int], str] | None = None


class PromptBudgeter:
    """Assembles prompt sections within a token budget.

    Processes sections by priority tier (MANDATORY first, NICE_TO_HAVE last).
    Truncatable sections are shortened before being dropped entirely.
    Final output preserves the original insertion order of included sections.
    """

    def __init__(
        self,
        total_budget: int,
        response_reserve: int,
        token_counter: Any,
        template_overhead: int = 100,
    ) -> None:
        self._total_budget = total_budget
        self._response_reserve = response_reserve
        self._template_overhead = template_overhead
        self._counter = token_counter
        self._sections: list[PromptSection] = []
        self._insertion_idx = 0

    @property
    def prompt_budget(self) -> int:
        """Tokens available for the actual prompt content."""
        return max(0, self._total_budget - self._response_reserve - self._template_overhead)

    def add(
        self,
        name: str,
        content: str,
        priority: SectionPriority,
        truncatable: bool = False,
        min_tokens: int = 0,
        truncate_fn: Callable[[str, int], str] | None = None,
    ) -> None:
        """Add a section to the budget. Empty content is silently ignored."""
        if not content or not content.strip():
            return
        token_count = self._counter.count_tokens(content)
        self._sections.append(PromptSection(
            name=name,
            content=content,
            priority=priority,
            token_count=token_count,
            insertion_order=self._insertion_idx,
            truncatable=truncatable,
            min_tokens=min_tokens,
            truncate_fn=truncate_fn,
        ))
        self._insertion_idx += 1

    def build(self) -> tuple[str, list[str]]:
        """Assemble the prompt within budget.

        Returns:
            (prompt_text, list of dropped section names)
        """
        budget = self.prompt_budget
        included: list[PromptSection] = []
        dropped: list[str] = []
        used = 0

        # Process by priority tier
        by_tier: dict[SectionPriority, list[PromptSection]] = {}
        for s in self._sections:
            by_tier.setdefault(s.priority, []).append(s)

        for tier in sorted(by_tier.keys()):
            for section in by_tier[tier]:
                remaining = budget - used
                if section.token_count <= remaining:
                    # Fits entirely
                    included.append(section)
                    used += section.token_count
                elif (
                    section.truncatable
                    and section.truncate_fn is not None
                    and remaining >= section.min_tokens
                    and remaining > 0
                ):
                    # Truncate to fit
                    truncated = section.truncate_fn(section.content, remaining)
                    new_count = self._counter.count_tokens(truncated)
                    if new_count <= remaining and truncated.strip():
                        included.append(PromptSection(
                            name=section.name,
                            content=truncated,
                            priority=section.priority,
                            token_count=new_count,
                            insertion_order=section.insertion_order,
                            truncatable=section.truncatable,
                            min_tokens=section.min_tokens,
                            truncate_fn=section.truncate_fn,
                        ))
                        used += new_count
                        logger.debug("Truncated section '%s': %d→%d tokens",
                                     section.name, section.token_count, new_count)
                    else:
                        dropped.append(section.name)
                else:
                    dropped.append(section.name)

        if dropped:
            logger.info("Prompt budget %d/%d — dropped sections: %s",
                        used, budget, ", ".join(dropped))

        # Re-sort by original insertion order to preserve prompt flow
        included.sort(key=lambda s: s.insertion_order)
        prompt_text = "\n\n".join(s.content for s in included)
        return prompt_text, dropped


# ── Truncation Helpers ──────────────────────────────────────────────────────


def _truncate_conversation(content: str, max_tokens: int, counter: Any) -> str:
    """Drop oldest User+Maxim turn pairs from the front."""
    lines = content.split("\n")
    # Find turn boundaries (lines starting with "User:" or "Maxim:")
    turn_starts: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("User:") or stripped.startswith("Maxim:"):
            turn_starts.append(i)

    if len(turn_starts) < 2:
        return content

    # Try removing from the front, one turn boundary at a time
    for cut_idx in range(1, len(turn_starts)):
        candidate = "\n".join(lines[turn_starts[cut_idx]:])
        if counter.count_tokens(candidate) <= max_tokens:
            return candidate
    # Last resort: return just the last turn
    return "\n".join(lines[turn_starts[-1]:])


def _truncate_context_pool(content: str, max_tokens: int, counter: Any) -> str:
    """Drop oldest lines from the front of the context pool."""
    lines = content.split("\n")
    for start in range(1, len(lines)):
        candidate = "\n".join(lines[start:])
        if counter.count_tokens(candidate) <= max_tokens:
            return candidate
    return lines[-1] if lines else ""


def _truncate_tool_guidance(content: str, max_tokens: int, counter: Any) -> str:
    """Remove Example lines, then indented detail lines."""
    lines = content.split("\n")
    # First pass: remove lines containing "Example" or "e.g."
    filtered = [l for l in lines if "Example" not in l and "e.g." not in l]
    candidate = "\n".join(filtered)
    if counter.count_tokens(candidate) <= max_tokens:
        return candidate
    # Second pass: remove indented detail lines (4+ spaces or tab)
    filtered = [l for l in filtered if not l.startswith("    ") and not l.startswith("\t")]
    return "\n".join(filtered)


def _truncate_reasoning_carryover(content: str, max_tokens: int, counter: Any) -> str:
    """Drop oldest entries from the reasoning carryover."""
    lines = content.split("\n")
    # Keep header line, truncate entry lines from front
    header_lines: list[str] = []
    entry_lines: list[str] = []
    for line in lines:
        if line.startswith("- "):
            entry_lines.append(line)
        else:
            header_lines.append(line)

    header = "\n".join(header_lines)
    for start in range(1, len(entry_lines)):
        candidate = header + "\n" + "\n".join(entry_lines[start:])
        if counter.count_tokens(candidate) <= max_tokens:
            return candidate
    return header


def _truncate_manifest(content: str, max_tokens: int, counter: Any) -> str:
    """Truncate a manifest section by removing file entries from the end."""
    lines = content.split("\n")
    # Separate header/rule lines from indented file entries
    header_lines: list[str] = []
    entry_lines: list[str] = []
    for line in lines:
        if line.startswith("  ") and not line.startswith("  ...") and not line.startswith("  1.") and not line.startswith("  2.") and not line.startswith("  3."):
            entry_lines.append(line)
        else:
            header_lines.append(line)

    # Remove entries from the end until it fits
    while entry_lines and counter.count_tokens(
        "\n".join(header_lines + entry_lines)
    ) > max_tokens:
        entry_lines.pop()

    if entry_lines:
        # Reconstruct: headers before entries, then remaining headers after
        return "\n".join(header_lines[:2] + entry_lines + header_lines[2:])
    return "\n".join(header_lines)


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning Carryover
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ReasoningEntry:
    """A single decision+outcome record for carryover."""

    action_tool: str
    reasoning: str
    success: bool
    result_summary: str
    timestamp: float = field(default_factory=time.time)

    def to_prompt_line(self) -> str:
        """Compact one-liner for prompt injection."""
        status = "OK" if self.success else "FAIL"
        reason = self.reasoning[:80] if self.reasoning else ""
        summary = self.result_summary[:100] if self.result_summary else ""
        return f"- {self.action_tool}: {reason} [{status}: {summary}]"


class ReasoningCarryover:
    """Rolling buffer of recent decision+outcome summaries.

    Thread-safe. Injected into the next LLM prompt as working memory
    so the model can reason about its own prior decisions.
    """

    def __init__(self, max_entries: int = 5) -> None:
        self._max_entries = max_entries
        self._entries: list[ReasoningEntry] = []
        self._lock = threading.Lock()

    def record(
        self,
        tool_name: str,
        reasoning: str,
        success: bool,
        result_summary: str,
    ) -> None:
        """Record a decision+outcome. Evicts oldest if over max."""
        entry = ReasoningEntry(
            action_tool=tool_name,
            reasoning=reasoning,
            success=success,
            result_summary=result_summary,
        )
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]

    def get_prompt_text(self) -> str:
        """Format entries as prompt text for injection."""
        with self._lock:
            if not self._entries:
                return ""
            lines = ["=== Recent Decisions (Working Memory) ==="]
            for entry in self._entries:
                lines.append(entry.to_prompt_line())
            return "\n".join(lines)

    def clear(self) -> None:
        """Wipe the buffer."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Worker
# ─────────────────────────────────────────────────────────────────────────────


class LLMWorker:
    """
    Dedicated thread for LLM inference.

    Design principles:
    - Main loop NEVER waits on LLM
    - Stale contexts are dropped (only latest matters)
    - Results are non-blocking to consume
    - Graceful degradation if LLM is slow/unavailable
    """

    def __init__(
        self,
        llm: LLMBackend,
        max_queue_size: int = 5,
        stale_threshold_s: float = 2.0,
        llm_timeout_s: float = 60.0,
        energy_tracker: "LLMEnergyTracker | None" = None,
        n_ctx: int = 4096,
        token_counter: Any | None = None,
        pool: "WorkerPool | None" = None,
    ):
        self._llm = llm
        self._stale_threshold = stale_threshold_s
        self._llm_timeout = llm_timeout_s
        self._energy_tracker = energy_tracker
        self._n_ctx = n_ctx
        self._token_counter = token_counter or CharEstimateCounter()
        self._reasoning_carryover = ReasoningCarryover(max_entries=5)

        # WorkerPool integration: infer lane replaces internal threading
        self._pool: WorkerPool | None = pool
        self._owns_pool = pool is None  # create internal pool in start()

        # Legacy queues (used when no pool is provided)
        self._request_queue: queue.PriorityQueue[tuple[int, LLMRequest | None]] = (
            queue.PriorityQueue(maxsize=max_queue_size)
        )
        self._proposal_queue: queue.Queue[LLMProposal] = queue.Queue()
        self._latest_proposal: LLMProposal | None = None
        self._proposal_lock = threading.Lock()

        # Worker thread (legacy, unused when pool is active)
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        # Thread pool for timeout-wrapped LLM calls (used in both modes)
        self._llm_executor: concurrent.futures.ThreadPoolExecutor | None = None

        # Metrics
        self._requests_processed = 0
        self._requests_dropped = 0
        self._avg_latency_ms = 0.0

    def record_outcome(
        self,
        tool_name: str,
        reasoning: str,
        success: bool,
        result_summary: str,
    ) -> None:
        """Record a decision+outcome into the reasoning carryover buffer."""
        self._reasoning_carryover.record(tool_name, reasoning, success, result_summary)

    def retry_with_timeout(self, request: LLMRequest, timeout_s: float) -> bool:
        """Resubmit a request with a temporarily increased timeout.

        Used when the user asks for more processing time after a timeout.
        The timeout increases permanently (ratchets up) since this model
        is consistently slow.

        Returns True if queued, False if queue full.
        """
        old_timeout = self._llm_timeout
        self._llm_timeout = timeout_s
        # Refresh timestamp so staleness checks don't drop it
        request.timestamp = time.time()
        request.sort_index = (-request.priority, request.timestamp)
        logger.info("Retrying LLM request with timeout=%.0fs (was %.0fs)", timeout_s, old_timeout)

        if self._pool is not None:
            def _retry_fn(prefetched=None):
                if self._stop_event.is_set():
                    return None
                proposal = self._process_request(request)
                self._requests_processed += 1
                return proposal

            try:
                self._pool.submit(
                    lane="infer",
                    job_id=f"{request.request_id}-retry",
                    fn=_retry_fn,
                    priority=-request.priority,
                )
                return True
            except queue.Full:
                self._requests_dropped += 1
                return False
        else:
            # Legacy
            try:
                self._request_queue.put_nowait((-request.priority, request))
                return True
            except queue.Full:
                self._requests_dropped += 1
                return False

    def start(self) -> None:
        """Start the worker thread (or infer lane via WorkerPool)."""
        self._stop_event.clear()

        # Always create the LLM executor (needed for _call_llm_with_timeout)
        if self._llm_executor is None:
            self._llm_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="LLMCall"
            )

        # WorkerPool mode: use infer lane instead of manual thread
        if self._owns_pool:
            from maxim.runtime.worker_pool import LaneConfig, WorkerPool

            self._pool = WorkerPool(lane_configs={
                "infer": LaneConfig(name="infer", max_workers=1, requires_gpu=True),
            })

        if self._pool is not None:
            self._pool.start()
            logger.info("LLM worker started (WorkerPool infer lane)")
        else:
            # Legacy: start internal worker thread
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._worker_loop,
                    daemon=True,
                    name="LLMWorker",
                )
                self._worker.start()
                logger.info("LLM worker thread started (legacy)")

    def stop(self) -> None:
        """Stop the worker thread (or infer lane via WorkerPool)."""
        self._stop_event.set()

        # Stop WorkerPool if we own it
        if self._pool is not None and self._owns_pool:
            self._pool.stop()
            logger.info("LLM worker stopped (WorkerPool infer lane)")
        elif self._pool is None:
            # Legacy: unblock the queue with poison pill
            try:
                self._request_queue.put_nowait((0, None))
            except queue.Full:
                pass
            if self._worker:
                self._worker.join(timeout=2.0)
                if self._worker.is_alive():
                    logger.warning("LLM worker thread did not stop in time")
                else:
                    logger.info("LLM worker thread stopped (legacy)")

        # Always shutdown the LLM executor
        if self._llm_executor is not None:
            try:
                self._llm_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python < 3.9 doesn't have cancel_futures
                self._llm_executor.shutdown(wait=False)
            self._llm_executor = None

    def _call_llm_with_timeout(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any] | None:
        """Call LLM with timeout to allow graceful shutdown.

        Returns:
            LLM response dict or None if timeout/error/shutdown.
        """
        if self._stop_event.is_set():
            return None

        executor = self._llm_executor
        if executor is None:
            return None

        try:
            future = executor.submit(
                self._llm.generate_json,
                prompt,
                temperature,
                max_tokens,
            )
            # Wait with timeout, checking stop_event periodically
            timeout_remaining = self._llm_timeout
            poll_interval = 0.5
            while timeout_remaining > 0:
                if self._stop_event.is_set():
                    future.cancel()
                    return None
                try:
                    result = future.result(timeout=min(poll_interval, timeout_remaining))
                    return result
                except concurrent.futures.TimeoutError:
                    timeout_remaining -= poll_interval
                    continue
            # Final timeout exceeded
            logger.warning("LLM call timed out after %.1fs", self._llm_timeout)
            future.cancel()
            # Replace the executor so the orphaned thread doesn't block
            # future LLM calls (max_workers=1 means next submit() queues
            # behind the still-running orphan, causing cascading timeouts).
            try:
                old_executor = self._llm_executor
                self._llm_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="LLMCall"
                )
                if old_executor is not None:
                    old_executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.error("Failed to replace LLM executor after timeout: %s", e)
            return {"_timeout": True, "_timeout_s": self._llm_timeout}
        except concurrent.futures.CancelledError:
            return None
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return None

    def submit_context(
        self,
        context: StructuredContext,
        mode: ModeInfo,
        autonomy_level: AutonomyLevel,
        strategies: list[StrategyInfo],
        internet_access: bool,
        internet_policy_summary: str,
        priority: int = 0,
        *,
        available_tools: set[str] | None = None,
        tool_descriptions: dict[str, str] | None = None,
        context_pool_text: str = "",
        agent_states: list[dict[str, Any]] | None = None,
        recent_outcomes: list[dict[str, Any]] | None = None,
        use_tool_prompting: bool = False,
        triggering_input: str = "",
        conversation_history_text: str = "",
        pending_modification: dict[str, Any] | None = None,
        prefetch_context: str = "",
        skip_exploration: bool = False,
        current_strategy: str = "",
        is_sleeping: bool = False,
    ) -> bool:
        """
        Submit context for LLM processing (non-blocking).

        Returns True if queued, False if dropped (queue full).
        Main loop should call this frequently; stale requests are pruned.

        Args:
            context: Structured context from memory agent
            mode: Current mode information
            autonomy_level: Current autonomy level
            strategies: Available strategies
            internet_access: Whether internet is available
            internet_policy_summary: Summary of internet policy
            priority: Request priority (higher = more urgent)
            available_tools: Set of tool names available in current mode
            tool_descriptions: Dict of tool name -> description for prompts
            context_pool_text: Accumulated context/observations summary
            agent_states: List of agent state snapshots
            recent_outcomes: List of recent action outcomes for learning
            use_tool_prompting: Whether to use full tool-aware prompts
            triggering_input: The user input that triggered this request
            conversation_history_text: Formatted conversation history for context
        """
        request = LLMRequest(
            request_id=f"req-{time.time_ns()}",
            context=context,
            mode=mode,
            autonomy_level=autonomy_level,
            strategies=strategies,
            internet_access=internet_access,
            internet_policy_summary=internet_policy_summary,
            priority=priority,
            available_tools=available_tools or set(),
            tool_descriptions=tool_descriptions or {},
            context_pool_text=context_pool_text,
            agent_states=agent_states or [],
            recent_outcomes=recent_outcomes or [],
            use_tool_prompting=use_tool_prompting,
            triggering_input=triggering_input,
            conversation_history_text=conversation_history_text,
            pending_modification=pending_modification,
            prefetch_context=prefetch_context,
            skip_exploration=skip_exploration,
            current_strategy=current_strategy,
            is_sleeping=is_sleeping,
        )

        if self._pool is not None:
            # WorkerPool mode: wrap _process_request in a job
            def _infer_job(prefetched=None):
                # Staleness guard inside the job
                age = time.time() - request.timestamp
                if age > self._stale_threshold:
                    self._requests_dropped += 1
                    logger.debug("Dropped stale LLM request (age=%.2fs)", age)
                    return None
                if self._stop_event.is_set():
                    return None
                proposal = self._process_request(request)
                self._requests_processed += 1
                return proposal

            try:
                # Negate priority: higher caller priority → lower queue value
                self._pool.submit(
                    lane="infer",
                    job_id=request.request_id,
                    fn=_infer_job,
                    priority=-priority,
                )
                return True
            except queue.Full:
                self._requests_dropped += 1
                return False
        else:
            # Legacy: internal priority queue
            try:
                self._request_queue.put_nowait((-priority, request))
                return True
            except queue.Full:
                self._requests_dropped += 1
                return False

    def get_latest_proposal(self) -> LLMProposal | None:
        """
        Get the most recent proposal (non-blocking).

        Main loop calls this each iteration to check for LLM output.
        Returns None if no proposal available.
        """
        if self._pool is not None:
            # WorkerPool mode: poll completed infer jobs
            completed = self._pool.get_completed("infer")
            if completed is not None and completed.result is not None:
                return completed.result  # LLMProposal
            return None
        else:
            # Legacy: consume-once latest proposal
            with self._proposal_lock:
                proposal = self._latest_proposal
                self._latest_proposal = None
                return proposal

    def get_all_proposals(self) -> list[LLMProposal]:
        """Get all pending proposals (non-blocking)."""
        if self._pool is not None:
            # Drain all completed infer jobs
            proposals = []
            while True:
                completed = self._pool.get_completed("infer")
                if completed is None or completed.result is None:
                    break
                proposals.append(completed.result)
            return proposals
        else:
            # Legacy: drain proposal queue
            proposals = []
            while True:
                try:
                    proposals.append(self._proposal_queue.get_nowait())
                except queue.Empty:
                    break
            return proposals

    def _worker_loop(self) -> None:
        """Background worker that processes LLM requests."""
        while not self._stop_event.is_set():
            try:
                # Block waiting for request (with timeout for stop check)
                try:
                    _, request = self._request_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if request is None:  # Poison pill
                    break

                # Check if request is stale
                age = time.time() - request.timestamp
                if age > self._stale_threshold:
                    self._requests_dropped += 1
                    logger.debug(f"Dropped stale LLM request (age={age:.2f}s)")
                    continue

                # Process the request
                proposal = self._process_request(request)

                # Store result
                with self._proposal_lock:
                    self._latest_proposal = proposal
                self._proposal_queue.put(proposal)

                self._requests_processed += 1

            except Exception as e:
                # Log but don't crash the worker
                logger.error(f"LLMWorker error: {e}")

    def _process_request(self, request: LLMRequest) -> LLMProposal:
        """Process a single LLM request."""
        start_time = time.time()

        try:
            prompt = self._build_prompt(request)

            # Skip LLM call if no meaningful prompt (idle observation)
            if not prompt or not prompt.strip():
                return LLMProposal(
                    request_id=request.request_id,
                    action=None,
                    reasoning="No user input to respond to",
                    strategy_used=None,
                    confidence=0.0,
                    mode_goal_achieved=False,
                    citations=[],
                    latency_ms=0.0,
                    triggering_input=request.triggering_input,
                )

            # Check if this is a pre-built JSON response (simple answer)
            # These don't need LLM - just parse and return
            if prompt.startswith('{"action":'):
                latency_ms = (time.time() - start_time) * 1000
                try:
                    import json
                    response = json.loads(prompt)
                    return LLMProposal(
                        request_id=request.request_id,
                        action=response.get("action"),
                        reasoning=response.get("reasoning", "direct_answer"),
                        strategy_used="direct_answer",
                        confidence=response.get("confidence", 0.95),
                        mode_goal_achieved=response.get("mode_goal_achieved", False),
                        citations=[],
                        latency_ms=latency_ms,
                        triggering_input=request.triggering_input,
                    )
                except Exception:
                    pass  # Fall through to LLM if parse fails

            # Check for shutdown before LLM call
            if self._stop_event.is_set():
                return LLMProposal(
                    request_id=request.request_id,
                    action=None,
                    reasoning="Shutdown requested",
                    strategy_used=None,
                    confidence=0.0,
                    mode_goal_achieved=False,
                    citations=[],
                    latency_ms=(time.time() - start_time) * 1000,
                    error="shutdown",
                    triggering_input=request.triggering_input,
                )

            # Use mode-specific max tokens for dynamic response length
            max_tokens = request.mode.max_response_tokens
            response = self._call_llm_with_timeout(prompt, temperature=0.3, max_tokens=max_tokens)

            # Check for timeout — ask user if they want to wait longer
            if isinstance(response, dict) and response.get("_timeout"):
                timeout_s = response.get("_timeout_s", self._llm_timeout)
                question = (request.triggering_input or "your request")[:50]
                timeout_msg = (
                    f"I ran out of time processing '{question}' "
                    f"(took longer than {int(timeout_s)}s). "
                    f"Would you like me to try again with more time? "
                    f"Say 'yes' to double my time limit, a number of minutes "
                    f"(e.g. '2'), or 'no' to skip."
                )
                return LLMProposal(
                    request_id=request.request_id,
                    action={
                        "tool_name": "respond",
                        "params": {"message": timeout_msg},
                        "_timeout_retry": True,
                        "_original_request": request,
                        "_timeout_s": timeout_s,
                    },
                    reasoning="llm_timeout",
                    strategy_used="fallback",
                    confidence=0.7,
                    mode_goal_achieved=False,
                    citations=[],
                    latency_ms=(time.time() - start_time) * 1000,
                    triggering_input=request.triggering_input,
                )

            # Check for shutdown after LLM call
            if self._stop_event.is_set():
                return LLMProposal(
                    request_id=request.request_id,
                    action=None,
                    reasoning="Shutdown requested",
                    strategy_used=None,
                    confidence=0.0,
                    mode_goal_achieved=False,
                    citations=[],
                    latency_ms=(time.time() - start_time) * 1000,
                    error="shutdown",
                    triggering_input=request.triggering_input,
                )

            latency_ms = (time.time() - start_time) * 1000
            self._update_avg_latency(latency_ms)

            # Record energy usage if tracker is available
            if self._energy_tracker is not None and response and isinstance(response, dict):
                # Extract token counts from response if available
                # Models often include usage info in the response
                usage = response.get("usage", {})
                input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
                output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))

                # If no token counts in response, estimate from prompt/response length
                if input_tokens == 0:
                    # Rough estimate: ~4 chars per token
                    input_tokens = len(prompt) // 4
                if output_tokens == 0:
                    import json
                    try:
                        response_str = json.dumps(response)
                        output_tokens = len(response_str) // 4
                    except Exception:
                        output_tokens = 50  # Fallback estimate

                try:
                    self._energy_tracker.record(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model=getattr(self._llm, "model_name", "unknown"),
                        latency_ms=latency_ms,
                        context={
                            "request_id": request.request_id,
                            "mode": request.mode.name if request.mode else "unknown",
                        },
                    )
                except Exception as e:
                    logger.debug("Failed to record LLM energy: %s", e)

            if not response or not isinstance(response, dict):
                # LLM failed - generate a fallback response for the user
                fallback = self._generate_llm_fallback(request)
                if fallback:
                    return LLMProposal(
                        request_id=request.request_id,
                        action=fallback,
                        reasoning="llm_fallback",
                        strategy_used="fallback",
                        confidence=0.7,
                        mode_goal_achieved=False,
                        citations=[],
                        latency_ms=latency_ms,
                        triggering_input=request.triggering_input,
                    )

                return LLMProposal(
                    request_id=request.request_id,
                    action=None,
                    reasoning="LLM returned invalid response",
                    strategy_used=None,
                    confidence=0.0,
                    mode_goal_achieved=False,
                    citations=[],
                    latency_ms=latency_ms,
                    error="Invalid LLM response",
                    triggering_input=request.triggering_input,
                )

            # Extract next_actions if present (sequential execution)
            next_actions = response.get("next_actions", [])
            if not isinstance(next_actions, list):
                next_actions = []

            # Extract parallel_actions if present (batched execution)
            # These execute together before the next LLM call
            parallel_actions = response.get("parallel_actions", [])
            if not isinstance(parallel_actions, list):
                parallel_actions = []

            # Extract planning mode fields (prefixed with _)
            plan_text = response.pop("_plan_text", None)
            requires_approval = response.pop("_requires_approval", False)

            return LLMProposal(
                request_id=request.request_id,
                action=response.get("action"),
                reasoning=response.get("reasoning", ""),
                strategy_used=response.get("strategy"),
                confidence=response.get("confidence", 0.5),
                mode_goal_achieved=response.get("mode_goal_achieved", False),
                citations=response.get("citations", []),
                latency_ms=latency_ms,
                next_actions=next_actions,
                parallel_actions=parallel_actions,
                triggering_input=request.triggering_input,
                plan_text=plan_text,
                requires_approval=requires_approval,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return LLMProposal(
                request_id=request.request_id,
                action=None,
                reasoning="",
                strategy_used=None,
                confidence=0.0,
                mode_goal_achieved=False,
                latency_ms=latency_ms,
                error=str(e),
                triggering_input=request.triggering_input,
            )

    def _generate_llm_fallback(self, request: LLMRequest) -> dict[str, Any] | None:
        """Generate a fallback response when LLM fails to produce valid JSON.

        Returns an action dict or None if no fallback is appropriate.
        """
        # Use triggering_input if provided (already validated by agent loop)
        # Otherwise fall back to cli_inputs with maxim/reachy keyword check
        latest_input = None
        if request.triggering_input:
            latest_input = request.triggering_input
        elif request.context.cli_inputs:
            for inp in request.context.cli_inputs:
                if inp and ("maxim" in inp.lower() or "reachy" in inp.lower()):
                    latest_input = inp
                    break

        if not latest_input:
            return None

        # Extract the question (strip wake words if present)
        question = latest_input.lower()
        for wake_word in ["maxim", "reachy"]:
            question = question.replace(wake_word, "")
        question = question.strip().lstrip(",").lstrip(":").strip()

        # Generate a humble fallback response
        default_responses = {
            "what": "I'm not sure how to answer '{question}'. Could you rephrase that or ask something else?",
            "how": "I'd need to think more about '{question}'. Is there something specific I can help with?",
            "why": "That's an interesting question about '{question}'. I'm still learning about many things.",
            "who": "I don't have enough information to answer '{question}' right now.",
            "where": "I'm not certain about '{question}'. Can I help with something else?",
            "when": "I don't have that information about '{question}'.",
        }
        generic_default = (
            "I heard your question about '{question}', but I'm having trouble processing it right now. "
            "Could you try rephrasing?"
        )

        fallback_config = get_fallback_responses()
        if not isinstance(fallback_config, dict):
            fallback_config = {}
        question_templates = fallback_config.get("question_types")
        generic_template = fallback_config.get("generic")

        fallback_responses = dict(default_responses)
        if isinstance(question_templates, dict):
            for key, template in question_templates.items():
                if template:
                    fallback_responses[str(key)] = str(template)

        # Try simple arithmetic before generic question-type templates
        arithmetic_result = self._evaluate_simple_arithmetic(question)
        if arithmetic_result:
            return {
                "tool_name": "respond",
                "params": {"message": arithmetic_result},
            }

        # Try unary math (square root, cube root, squared, cubed)
        unary_result = self._evaluate_unary_math(question)
        if unary_result:
            return {
                "tool_name": "respond",
                "params": {"message": unary_result},
            }

        # Find matching question type
        for q_type, response in fallback_responses.items():
            if question.startswith(q_type):
                try:
                    response_text = response.format(question=question)
                except Exception:
                    response_text = response
                return {
                    "tool_name": "respond",
                    "params": {"message": response_text},
                }

        # Generic fallback
        if not isinstance(generic_template, str) or not generic_template:
            generic_template = generic_default
        try:
            generic_text = generic_template.format(question=question[:50])
        except Exception:
            generic_text = generic_template
        return {
            "tool_name": "respond",
            "params": {
                "message": generic_text,
            },
        }

    def _build_prompt(self, request: LLMRequest) -> str:
        """Build prompt from request.

        This method constructs either:
        1. A simple ANSWER_ONLY prompt for basic questions
        2. A tool-aware prompt with full context for complex requests

        Note: This prompt is wrapped by the LLMRouter.generate_json() method
        with the appropriate chat format (ChatML, Mistral, etc.) based on
        the model's prompt_style configuration.
        """
        context = request.context

        # Check for action followup that needs processing
        # Format: [ACTION_FOLLOWUP type=X tool=Y mode=Z query='Q']: result
        action_followup_input = ""
        if context.cli_inputs:
            for cli_input in context.cli_inputs:
                if cli_input and cli_input.startswith("[ACTION_FOLLOWUP"):
                    action_followup_input = cli_input
                    break
                # Legacy format support
                if cli_input and cli_input.startswith("[SEARCH RESULT"):
                    action_followup_input = cli_input
                    break

        # If we have an action followup, generate appropriate prompt based on type
        if action_followup_input:
            return self._build_followup_prompt(action_followup_input)

        # Check for pending user input that needs a response
        # Use triggering_input if provided (agent loop already validated it should be processed)
        # Otherwise fall back to checking cli_inputs for "maxim" keyword
        user_question = ""
        if request.triggering_input:
            # Agent loop already determined this input should be processed
            user_question = request.triggering_input
        elif context.cli_inputs:
            latest_input = context.cli_inputs[-1] if context.cli_inputs else None
            if latest_input and ("maxim" in latest_input.lower() or "reachy" in latest_input.lower()):
                user_question = latest_input

        if not user_question:
            # No user input - check if we should still process (exploration, etc.)
            if not request.use_tool_prompting:
                return ""
            # For tool prompting without user input, continue to build context-only prompt

        # Extract question text (strip wake words if present)
        question_text = ""
        if user_question:
            question_text = user_question.lower()
            for wake_word in ["maxim", "reachy"]:
                question_text = question_text.replace(wake_word, "")
            question_text = question_text.strip().lstrip(",").lstrip(":").strip()
            if question_text.endswith("?"):
                question_text = question_text[:-1].strip()

        # Get current date/time for context
        now = datetime.now()
        date_str = now.strftime("%A, %B %d, %Y")
        time_str = now.strftime("%I:%M %p")

        # Try simple answers first (no LLM needed)
        if question_text:
            answer = self._generate_simple_answer(question_text, date_str, time_str)
            if answer:
                # Use json.dumps to properly escape the answer string
                import json
                escaped_answer = json.dumps(answer)  # Returns quoted, escaped string
                # Field order: action → confidence → reasoning (most important first)
                return f'{{"action":{{"tool_name":"respond","params":{{"message":{escaped_answer}}}}},"confidence":0.95,"reasoning":"direct_answer"}}'

        # If not using tool prompting, fall back to ANSWER_ONLY
        if not request.use_tool_prompting or not request.available_tools:
            if question_text:
                return f"""ANSWER_ONLY|{question_text}"""
            return ""

        # Build full tool-aware prompt
        return self._build_tool_aware_prompt(request, question_text, date_str, time_str)

    # ── Section Builders ──────────────────────────────────────────────────

    @staticmethod
    def _build_planning_banner(autonomy_level: AutonomyLevel) -> str:
        """Build the planning mode banner text."""
        if autonomy_level != AutonomyLevel.PLANNING:
            return ""
        lines = [
            "!" * 60,
            "!!! CRITICAL: PLANNING MODE - YOU MUST ASK PERMISSION FIRST !!!",
            "!" * 60,
            "",
            "DO NOT output raw JSON. You MUST follow this EXACT format:",
            "",
            "STEP 1: Write a proposal IN PLAIN ENGLISH asking for permission",
            "STEP 2: Write the EXACT delimiter: <|action_json|>",
            "STEP 3: Write the JSON object",
            "",
            "=== CORRECT FORMAT EXAMPLE 1 ===",
            "I'd like to search the internet for the current Broncos vs Patriots score.",
            "May I proceed with this search?",
            "<|action_json|>",
            '{"action": {"tool_name": "internet_search", "params": {"query": "Broncos vs Patriots score today"}}, "confidence": 0.9}',
            "",
            "=== CORRECT FORMAT EXAMPLE 2 ===",
            "To answer your question about the weather, I need to search online.",
            "Should I look up the current weather for you?",
            "<|action_json|>",
            '{"action": {"tool_name": "internet_search", "params": {"query": "current weather"}}, "confidence": 0.9}',
            "",
            "=== WRONG (DO NOT DO THIS) ===",
            '{"action": {"tool_name": "internet_search", ...}}  <-- NO! Missing proposal text!',
            "",
            "YOUR RESPONSE MUST START WITH PLAIN TEXT, NOT JSON!",
            "!" * 60,
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_modification_section(mod: dict[str, Any]) -> str:
        """Build the modification request section."""
        import json
        lines = [
            "=" * 60,
            ">>> ACTION MODIFICATION REQUESTED <<<",
            "=" * 60,
            "",
            "The user requested a modification to the following proposed action:",
            "",
            "ORIGINAL ACTION:",
            f"  Tool: {mod.get('original_tool_name', 'unknown')}",
            f"  Parameters: {json.dumps(mod.get('original_action', {}).get('params', {}), indent=4)}",
            f"  Original reasoning: {mod.get('original_reasoning', 'not provided')}",
            "",
            "USER'S MODIFICATION REQUEST:",
            f'  "{mod.get("user_modification", "")}"',
            "",
            "YOUR TASK: Revise the action based on the user's feedback.",
            "- Interpret what change the user wants",
            "- Keep the same tool if appropriate, or choose a different one",
            "- Update the parameters according to the user's request",
            "- Provide updated reasoning",
            "",
            "Respond with a REVISED action that incorporates the user's modification.",
            "=" * 60,
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_identity_section(mode: ModeInfo, request: LLMRequest, date_str: str, time_str: str) -> str:
        """Build the system identity and operational state section."""
        lines = [
            "You are Maxim, a robot assistant.",
            "",
            "=== OPERATIONAL STATE ===",
            f"Mode: {mode.name.upper()}",
            f"Mode goal: {mode.goal}",
        ]

        if request.current_strategy:
            from maxim.modes.definitions import STRATEGIES
            strategy = STRATEGIES.get(request.current_strategy)
            if strategy:
                lines.append(f"Strategy: {strategy.name.upper()} - {strategy.description}")
                lines.append(f"Strategy focus: {strategy.focus}")

        lines.append(f"Autonomy level: {request.autonomy_level.value if request.autonomy_level else 'unknown'}")

        if request.is_sleeping:
            lines.append("Processing state: SLEEP (minimal processing, monitoring for wake keywords)")
        else:
            lines.append("Processing state: AWAKE")

        return "\n".join(lines)

    @staticmethod
    def _build_datetime_section(date_str: str, time_str: str) -> str:
        """Build the date/time section."""
        lines = [
            "=== CURRENT DATE/TIME (IMPORTANT) ===",
            f"Today is {date_str}",
            f"Current time: {time_str}",
            "When searching for current events, scores, or news, ALWAYS include the date in your query.",
            f"Example: Instead of 'Broncos game score', search 'Broncos game score {date_str}'",
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_tools_section(request: LLMRequest, mode_name: str = "passive") -> str:
        """Build the available tools section, adapted to operational mode."""
        import json as _json
        lines = ["=== Available Tools ==="]
        tool_list = sorted(request.available_tools)
        for tool_name in tool_list:
            tool_info = request.tool_descriptions.get(tool_name, {})
            if isinstance(tool_info, dict) and tool_info:
                desc = tool_info.get("description", "")
                params = tool_info.get("params", {})
                example = tool_info.get("example", {})
                lines.append(f"- {tool_name}: {desc}")
                if params:
                    param_strs = [f"{k}={v}" for k, v in params.items()]
                    lines.append(f"    REQUIRED params: {', '.join(param_strs)}")
                if example:
                    lines.append(f"    Example: {_json.dumps(example)}")
            elif isinstance(tool_info, str) and tool_info:
                lines.append(f"- {tool_name}: {tool_info}")
            else:
                lines.append(f"- {tool_name}")

        result = "\n".join(lines)

        # In singularity mode, remove .maxim_workspace/ prefix from examples
        if mode_name == "singularity":
            result = result.replace(".maxim_workspace/", "")

        return result

    @staticmethod
    def _scan_workspace_entries(workspace_path: str) -> list[str]:
        """Scan .maxim_workspace/ and return formatted file entries."""
        import os
        if not os.path.isdir(workspace_path):
            return []
        entries: list[str] = []
        try:
            for root, _dirs, files in os.walk(workspace_path):
                for fname in sorted(files):
                    fpath = os.path.join(root, fname)
                    rel = os.path.relpath(fpath, os.path.dirname(workspace_path))
                    try:
                        size = os.path.getsize(fpath)
                        size_str = f"{size}B" if size < 1024 else f"{size // 1024}KB"
                        entries.append(f"  {rel} ({size_str})")
                    except OSError:
                        entries.append(f"  {rel}")
        except OSError:
            return []
        return entries

    @staticmethod
    def _build_workspace_manifest(mode_name: str = "passive", cwd: str | None = None) -> str:
        """Build a file manifest that adapts to the operational mode.

        - **singularity**: Full CWD tree (depth 2) — LLM has full read/write.
        - **active**: Workspace listing + CWD top-level (read-only, suggest edits).
        - **passive**: Workspace listing + CWD top-level (read-only, propose plans).
        """
        import os
        from maxim.utils.filesystem_policy import (
            get_effective_cwd,
            get_workspace_path,
            scan_cwd_tree,
        )

        if cwd is None:
            cwd = get_effective_cwd()
        workspace = get_workspace_path(cwd)
        cwd_name = os.path.basename(cwd) or cwd
        sections: list[str] = []

        if mode_name == "singularity":
            # Singularity: CWD is the primary view
            cwd_entries = scan_cwd_tree(cwd, max_depth=2, max_entries=30)
            n_entries = len(cwd_entries)
            sections.append(
                f"=== PROJECT DIRECTORY ({n_entries} entries, CWD: {cwd_name}) ==="
            )
            sections.append(
                "You have FULL read/write access to all files below."
            )
            sections.append(
                "!! IMPORTANT: These files ALREADY EXIST. Do NOT recreate them. !!"
            )
            sections.extend(cwd_entries)
            # Mention workspace as secondary scratch space
            ws_entries = LLMWorker._scan_workspace_entries(workspace)
            if ws_entries:
                sections.append(
                    f"\n.maxim_workspace/ also available for scratch/drafts ({len(ws_entries)} files)."
                )
            sections.append("\nRULES:")
            sections.append("  1. Use read_file FIRST to see current contents before editing")
            sections.append("  2. Then write_file with overwrite=true to update")
            sections.append("  3. NEVER write a brand-new file that duplicates an existing one")
            if n_entries >= 5:
                sections.append(
                    "PLAN FIRST: With many project files, use 'respond' to outline "
                    "your approach before making changes."
                )
        else:
            # Passive/Active: Workspace is primary
            ws_entries = LLMWorker._scan_workspace_entries(workspace)
            if ws_entries:
                n_files = len(ws_entries)
                sections.append(
                    f"=== EXISTING WORKSPACE ({n_files} file{'s' if n_files != 1 else ''}) ==="
                )
                sections.append(
                    "!! IMPORTANT: These files ALREADY EXIST. Do NOT recreate them. !!"
                )
                sections.extend(ws_entries[:15])
                if n_files > 15:
                    sections.append(f"  ... and {n_files - 15} more files")
                sections.append("RULES for existing files:")
                sections.append("  1. Use read_file FIRST to see current contents")
                sections.append("  2. Then write_file with overwrite=true to update")
                sections.append("  3. NEVER write a brand-new file that duplicates an existing one")
                if n_files >= 3:
                    sections.append(
                        "PLAN FIRST: With multiple existing files, use 'respond' to outline "
                        "your approach before making changes."
                    )

            # Add CWD read-only context
            cwd_entries = scan_cwd_tree(cwd, max_depth=1, max_entries=10)
            if cwd_entries:
                sections.append(
                    f"\n=== CWD Context (read-only: {cwd_name}) ==="
                )
                if mode_name == "active":
                    sections.append(
                        "You can READ these files and SUGGEST edits (requires approval)."
                    )
                else:
                    sections.append(
                        "You can READ these files. CWD edits must be proposed as plans."
                    )
                sections.extend(cwd_entries)

        return "\n".join(sections) if sections else ""

    @staticmethod
    def _build_tool_guidance_core(mode_name: str = "passive") -> str:
        """Build compact essential tool guidance, adapted to operational mode."""
        lines = [
            "=== Tool Parameters ===",
            '- respond: {"message": "your answer"}',
            '- speak: {"text": "text to speak"}',
        ]

        if mode_name == "singularity":
            lines.extend([
                '- write_file: {"path": "src/main.py", "content": "code", "overwrite": true}',
                '- read_file: {"path": "src/main.py"}',
                '- internet_search: {"query": "search query"}',
                "",
                "=== File Operation Rules ===",
                "You can read and write ANY file within the project directory.",
                "EXISTING files: read_file FIRST, then write_file with overwrite=true.",
                "NEW files: write_file (no overwrite needed).",
                "NEVER blindly overwrite — always read first to understand current contents.",
            ])
        else:
            lines.extend([
                '- write_file: {"path": ".maxim_workspace/f.py", "content": "code", "overwrite": true}',
                '- read_file: {"path": ".maxim_workspace/f.py"}',
                '- internet_search: {"query": "search query"}',
                "",
                "=== File Operation Rules ===",
                "All file writes MUST use '.maxim_workspace/' prefix.",
                "EXISTING files: read_file FIRST, then write_file with overwrite=true.",
                "NEW files: write_file (no overwrite needed).",
                "NEVER blindly overwrite — always read first to understand current contents.",
            ])
            if mode_name == "active":
                lines.append("CWD files: You can read freely and suggest edits (requires approval).")
            else:
                lines.append("CWD files: You can read freely. Edits must be proposed as plans.")

        lines.extend([
            "",
            "=== Planning Rule ===",
            "For changes that touch multiple files or significantly alter existing code:",
            "  1. Use 'respond' to outline your plan FIRST",
            "  2. Wait for user confirmation before executing",
            "  3. Then read_file → modify → write_file for each file",
            "If request is unclear, use 'respond' to ask for clarification.",
        ])
        return "\n".join(lines)

    @staticmethod
    def _build_tool_guidance_extended(mode_name: str = "passive") -> str:
        """Build extended tool selection guidance, adapted to operational mode."""
        lines = [
            "=== Tool Selection ===",
            "- REAL-TIME DATA (scores, weather, news, prices): Use 'internet_search'",
            "- KNOWLEDGE from memory (what is X, explain Y): Use 'respond'",
            "- CREATE/WRITE FILES (create script, make file): Use 'write_file'",
            "- VISUAL COMMANDS (look at, focus on, track): Use 'focus_interests', 'track_target'",
            "- MATH CALCULATIONS: Use 'math' with sqrt/factorial/compute",
            "- MATH MEMORY: Use 'math' with recall_memory/store_memory",
            "",
            "=== FILE WORKSPACE ===",
        ]
        if mode_name == "singularity":
            lines.append(
                "Project directory: full read/write access to all files."
            )
            lines.append(
                "Workspace (.maxim_workspace/): drafts/ (code), notes/ (thinking), "
                "plans/ (proposals), scratch/ (temp)"
            )
        else:
            lines.append(
                "Workspace: drafts/ (code), notes/ (thinking), plans/ (proposals), scratch/ (temp)"
            )
        lines.extend([
            "",
            "=== BATCHED TOOL CALLS ===",
            "Batch exploration with parallel_actions for efficiency.",
        ])
        return "\n".join(lines)

    @staticmethod
    def _build_observation_section(percept: Any) -> str:
        """Build the current observation section."""
        lines = ["=== Current Observation ==="]
        if percept.transcript_chunk:
            lines.append(f'Heard: "{percept.transcript_chunk[:200]}"')
        if percept.detections:
            objects = [d.get("label", "?") for d in percept.detections[:5]]
            lines.append(f"Visible objects: {', '.join(objects)}")
        if percept.cli_input:
            lines.append(f'User input: "{percept.cli_input}"')
        return "\n".join(lines) if len(lines) > 1 else ""

    @staticmethod
    def _build_instructions_section(request: LLMRequest) -> str:
        """Build the response format instructions section."""
        lines = ["=== Instructions ==="]
        if request.autonomy_level == AutonomyLevel.PLANNING:
            lines.extend([
                "!" * 40,
                "FINAL REMINDER: PLANNING MODE ACTIVE!",
                "Your response MUST be:",
                "1. Plain text proposal asking permission (NOT JSON!)",
                "2. The delimiter <|action_json|>",
                "3. Then the JSON",
                "START YOUR RESPONSE WITH PLAIN ENGLISH TEXT!",
                "!" * 40,
            ])
        else:
            lines.extend([
                "Respond with a compact JSON object. IMPORTANT: Put fields in this order:",
                '  "action": {"tool_name": "<tool>", "params": {...}}',
                '  "confidence": 0.0-1.0',
                '  "reasoning": "Brief explanation (1 sentence)"',
                "Keep response compact. Do not include optional fields.",
            ])
        return "\n".join(lines)

    @staticmethod
    def _is_realtime_request(question_text: str) -> bool:
        """Check if the question is asking for real-time data."""
        realtime_keywords = [
            "score", "game", "match", "playing", "vs", "versus",
            "weather", "temperature", "forecast",
            "news", "latest", "current", "today", "now", "live",
            "price", "cost", "stock", "bitcoin", "crypto",
            "broncos", "patriots", "lakers", "yankees",
        ]
        q_lower = question_text.lower() if question_text else ""
        return any(kw in q_lower for kw in realtime_keywords)

    def _build_tool_aware_prompt(
        self,
        request: LLMRequest,
        question_text: str,
        date_str: str,
        time_str: str,
    ) -> str:
        """Build a comprehensive tool-aware prompt for complex reasoning.

        Uses PromptBudgeter to assemble sections within the model's token
        budget, dropping lower-priority sections when space is tight.
        """
        context = request.context
        mode = request.mode
        counter = self._token_counter
        response_reserve = mode.max_response_tokens
        mode_name = mode.name  # "passive", "active", or "singularity"

        from maxim.utils.filesystem_policy import get_effective_cwd
        effective_cwd = get_effective_cwd()

        budgeter = PromptBudgeter(
            total_budget=self._n_ctx,
            response_reserve=response_reserve,
            token_counter=counter,
        )

        is_realtime = self._is_realtime_request(question_text)

        # ── MANDATORY sections (never dropped) ──

        # Response format instructions
        budgeter.add(
            "instructions",
            self._build_instructions_section(request),
            SectionPriority.MANDATORY,
        )

        # User request
        if question_text:
            budgeter.add(
                "user_request",
                f'=== User Request ===\n"{question_text}"',
                SectionPriority.MANDATORY,
            )

        # ── CRITICAL sections ──

        # Planning mode banner
        budgeter.add(
            "planning_banner",
            self._build_planning_banner(request.autonomy_level),
            SectionPriority.CRITICAL,
        )

        # Modification request
        if request.pending_modification:
            budgeter.add(
                "modification",
                self._build_modification_section(request.pending_modification),
                SectionPriority.CRITICAL,
            )

        # System identity + operational state
        budgeter.add(
            "identity",
            self._build_identity_section(mode, request, date_str, time_str),
            SectionPriority.CRITICAL,
        )

        # Available tools (truncatable — drop examples first)
        budgeter.add(
            "tools",
            self._build_tools_section(request, mode_name=mode_name),
            SectionPriority.CRITICAL,
            truncatable=True,
            min_tokens=50,
            truncate_fn=lambda c, m: _truncate_tool_guidance(c, m, counter),
        )

        # Workspace manifest — LLM MUST know what files exist
        workspace_manifest = self._build_workspace_manifest(
            mode_name=mode_name, cwd=effective_cwd,
        )
        if workspace_manifest:
            is_singularity = (mode_name == "singularity")
            budgeter.add(
                "workspace_manifest",
                workspace_manifest,
                SectionPriority.CRITICAL,
                truncatable=is_singularity,
                min_tokens=80 if is_singularity else 0,
                truncate_fn=(
                    (lambda c, m: _truncate_manifest(c, m, counter))
                    if is_singularity else None
                ),
            )

        # Real-time data hint
        if is_realtime:
            hint = ">>> REAL-TIME DATA NEEDED <<<\n"
            if request.autonomy_level == AutonomyLevel.PLANNING:
                hint += "Propose using 'internet_search' and ask for approval."
            else:
                hint += "Use 'internet_search' tool directly."
            budgeter.add("realtime_hint", hint, SectionPriority.CRITICAL)

        # ── IMPORTANT sections (truncatable) ──

        # Tool guidance — compact core (params + workspace rules)
        budgeter.add(
            "tool_guidance_core",
            self._build_tool_guidance_core(mode_name=mode_name),
            SectionPriority.IMPORTANT,
        )

        # Tool guidance — extended (selection tips, batching examples)
        budgeter.add(
            "tool_guidance_extended",
            self._build_tool_guidance_extended(mode_name=mode_name),
            SectionPriority.NICE_TO_HAVE,
        )

        # Date/time
        budgeter.add(
            "datetime",
            self._build_datetime_section(date_str, time_str),
            SectionPriority.IMPORTANT,
        )

        # Conversation history
        if request.conversation_history_text:
            budgeter.add(
                "conversation",
                "=== Conversation History ===\n" + request.conversation_history_text,
                SectionPriority.IMPORTANT,
                truncatable=True,
                min_tokens=50,
                truncate_fn=lambda c, m: _truncate_conversation(c, m, counter),
            )

        # Context pool
        if request.context_pool_text:
            budgeter.add(
                "context_pool",
                "=== Context ===\n" + request.context_pool_text,
                SectionPriority.IMPORTANT,
                truncatable=True,
                min_tokens=30,
                truncate_fn=lambda c, m: _truncate_context_pool(c, m, counter),
            )

        # Reasoning carryover (NEW)
        carryover_text = self._reasoning_carryover.get_prompt_text()
        if carryover_text:
            budgeter.add(
                "reasoning_carryover",
                carryover_text,
                SectionPriority.IMPORTANT,
                truncatable=True,
                min_tokens=30,
                truncate_fn=lambda c, m: _truncate_reasoning_carryover(c, m, counter),
            )

        # Pre-fetched context
        if request.prefetch_context:
            prefetch = request.prefetch_context
            if request.skip_exploration:
                prefetch += (
                    "\n\n" + "!" * 50
                    + "\n!!! ONE-CALL MODE: WRITE DIRECTLY !!!\n"
                    + "!" * 50
                    + "\n\nFile discovery is COMPLETE. Do NOT use glob or read_file."
                    + "\nProceed DIRECTLY to your action:"
                    + "\n- For EXISTING file: write_file with overwrite=True"
                    + "\n- For NEW file: write_file (no overwrite needed)"
                    + "\n\nYour response should be the write_file action, NOT exploration."
                )
            # Discovery plans are higher priority — they contain structured
            # file context that the LLM needs for planning.
            has_discovery = "DISCOVERY PLAN" in prefetch or "FILE DISCOVERY" in prefetch
            priority = SectionPriority.CRITICAL if has_discovery else SectionPriority.IMPORTANT
            budgeter.add(
                "prefetch_context",
                prefetch,
                priority,
                truncatable=True,
                min_tokens=50,
                truncate_fn=lambda c, m: _truncate_context_pool(c, m, counter),
            )

        # Coding guidelines
        guidelines_text = question_text
        if not guidelines_text and request.pending_modification:
            guidelines_text = request.pending_modification.get("user_modification", "")
        if guidelines_text:
            coding_context = build_coding_context(
                guidelines_text,
                include_sandbox_reminder=True,
                max_guidelines=2,
            )
            if coding_context:
                budgeter.add(
                    "coding_guidelines",
                    coding_context,
                    SectionPriority.NICE_TO_HAVE,
                )

        # ── NICE_TO_HAVE sections ──

        # Foundational context (Constitution & Agent Rules)
        budgeter.add(
            "foundational",
            _load_foundational_context(),
            SectionPriority.NICE_TO_HAVE,
        )

        # Mode context prompt
        if mode.context_prompt:
            budgeter.add(
                "mode_context",
                mode.context_prompt,
                SectionPriority.NICE_TO_HAVE,
            )

        # Strategy context prompt
        if request.current_strategy:
            from maxim.modes.definitions import STRATEGIES
            strategy = STRATEGIES.get(request.current_strategy)
            if strategy and strategy.context_prompt:
                budgeter.add(
                    "strategy_context",
                    strategy.context_prompt,
                    SectionPriority.NICE_TO_HAVE,
                )

        # Current observation
        if context.current_percept:
            obs_text = self._build_observation_section(context.current_percept)
            if obs_text:
                budgeter.add("observation", obs_text, SectionPriority.NICE_TO_HAVE)

        # Recent speech
        if context.detected_speech:
            speech_lines = ["=== Recent Speech ==="]
            for speech in context.detected_speech[-3:]:
                speech_lines.append(f'- "{speech[:100]}"')
            budgeter.add("speech", "\n".join(speech_lines), SectionPriority.NICE_TO_HAVE)

        # Agent states
        if request.agent_states:
            state_lines = ["=== Agent States ==="]
            for state in request.agent_states[-5:]:
                agent_name = state.get("agent", "unknown")
                agent_state = state.get("state", "unknown")
                goal = state.get("goal", "")
                state_lines.append(f"- {agent_name}: {agent_state}" + (f" (goal: {goal})" if goal else ""))
            budgeter.add("agent_states", "\n".join(state_lines), SectionPriority.NICE_TO_HAVE)

        # Recent outcomes
        if request.recent_outcomes:
            outcome_lines = ["=== Recent Action Outcomes ==="]
            for outcome in request.recent_outcomes[-3:]:
                tool = outcome.get("tool", "?")
                success = "succeeded" if outcome.get("success") else "failed"
                result = outcome.get("result", "")[:50] if outcome.get("result") else ""
                outcome_lines.append(f"- {tool}: {success}" + (f" ({result})" if result else ""))
            budgeter.add("recent_outcomes", "\n".join(outcome_lines), SectionPriority.NICE_TO_HAVE)

        # Statistical patterns
        if context.statistical_context and context.active_pattern_count > 0:
            stat_lines = [
                f"=== Statistical Patterns ({context.active_pattern_count} active) ===",
                context.statistical_context,
                "Use 'math' tool to investigate patterns (assess_randomness, analyze, recall_memory).",
                "Use 'internet_search' to research unfamiliar patterns or analysis techniques.",
            ]
            budgeter.add("statistical_patterns", "\n".join(stat_lines), SectionPriority.NICE_TO_HAVE)

        # ── Build final prompt ──
        prompt_text, dropped = budgeter.build()

        if dropped:
            logger.info("Prompt budget: dropped %d sections for %s (n_ctx=%d, reserve=%d)",
                        len(dropped), mode.name, self._n_ctx, response_reserve)

        return f"TOOL_PROMPT|{prompt_text}"

    def _build_followup_prompt(self, followup_input: str) -> str:
        """Build a prompt to handle action followups based on followup_type.

        Followup types:
          - "process": LLM processes results for next action (coding agent, grep)
          - "respond": LLM synthesizes results into user response
          - "engage":  LLM responds AND offers proactive follow-up options

        Format: [ACTION_FOLLOWUP type=X tool=Y mode=Z query='Q']: result
        Legacy: [SEARCH RESULT for 'query']: result
        """
        import re

        # Try new format first - more flexible regex that handles special chars in query
        # Format: [ACTION_FOLLOWUP type=X tool=Y mode=Z query='Q']: result
        new_format = re.match(
            r"\[ACTION_FOLLOWUP type=(\w+) tool=([\w_-]+) mode=([\w_-]+) query='(.*)'\]: (.*)",
            followup_input,
            re.DOTALL
        )

        if new_format:
            followup_type = new_format.group(1)
            tool_name = new_format.group(2)
            mode_name = new_format.group(3)
            # Query might contain trailing content, split on last ']:
            raw_query = new_format.group(4)
            # Handle case where query itself contains ']: by finding the actual split point
            if "']: " in followup_input:
                split_idx = followup_input.find("']: ")
                bracket_start = followup_input.find("query='") + 7
                original_query = followup_input[bracket_start:split_idx]
                result = followup_input[split_idx + 4:]
            else:
                original_query = raw_query
                result = new_format.group(5)
            logger.debug(f"Parsed followup: type={followup_type}, tool={tool_name}, query_len={len(original_query)}, result_len={len(result)}")
        else:
            # Legacy format: [SEARCH RESULT for 'query']: result
            legacy_format = re.match(r"\[SEARCH RESULT for '([^']+)'\]: (.*)", followup_input, re.DOTALL)
            if legacy_format:
                followup_type = "engage"  # Default to engage for legacy
                tool_name = "internet_search"
                mode_name = "live"
                original_query = legacy_format.group(1)
                result = legacy_format.group(2)
            else:
                # Fallback - log warning since we couldn't parse
                logger.warning(f"Could not parse followup input format, using fallback. Input starts with: {followup_input[:100]}")
                followup_type = "engage"  # Default to engage for search results
                tool_name = "internet_search"
                mode_name = "live"
                original_query = ""
                result = followup_input

        # Build prompt based on followup_type
        if followup_type == "process":
            return self._build_process_prompt(tool_name, original_query, result)
        elif followup_type == "respond":
            return self._build_respond_prompt(tool_name, original_query, result)
        elif followup_type == "engage":
            return self._build_engage_prompt(tool_name, original_query, result, mode_name)
        else:
            # Default to respond
            return self._build_respond_prompt(tool_name, original_query, result)

    def _build_process_prompt(self, tool_name: str, query: str, result: str) -> str:
        """Build prompt for 'process' followup - LLM decides next action based on results."""
        # Check if this is batched exploration results
        is_batched = tool_name == "batched_exploration" or "BATCHED EXPLORATION RESULTS" in result

        if is_batched:
            prompt = f"""You completed batched exploration. Now analyze ALL results and make your final action.

Original request: "{query}"

{result}

=== Instructions ===
You now have complete context from the batched exploration. Based on ALL the results above:

1. If modifying a file: Use 'write_file' with the COMPLETE updated content
   - Include overwrite=True for existing files
   - Incorporate patterns/conventions from related files you read

2. If creating a new file: Use 'write_file' with appropriate content

3. If task is already complete or you need to inform the user: Use 'respond'

4. If you need more information: Request additional tools (but batching should have provided enough)

Respond with JSON:
{{"action": {{"tool_name": "<tool>", "params": {{...}}}}, "confidence": 0.9, "reasoning": "your_reasoning"}}"""
        else:
            prompt = f"""You just executed '{tool_name}'. Analyze the results and decide your next action.

Original request: "{query}"

Results from {tool_name}:
{result}

=== Instructions ===
Based on these results, determine your next action:
- If the task is complete, use 'respond' to inform the user
- If more steps are needed, choose the appropriate tool
- Extract relevant information to inform your decision

Respond with JSON:
{{"action": {{"tool_name": "<next_tool>", "params": {{...}}}}, "confidence": 0.9, "reasoning": "your_reasoning"}}"""

        return f"TOOL_PROMPT|{prompt}"

    def _build_respond_prompt(self, tool_name: str, query: str, result: str) -> str:
        """Build prompt for 'respond' followup - synthesize results into user response."""
        prompt = f"""You completed an internet search. Synthesize the results into a helpful response.

Original user request: "{query}"

Search Results:
{result}

=== Instructions ===
Extract the most relevant information from the search results and answer the user's question clearly.

IMPORTANT: You MUST use the "respond" tool. Do NOT use any other tool name.

Example JSON format:
{{"action": {{"tool_name": "respond", "params": {{"message": "YOUR ANSWER HERE"}}}}, "confidence": 0.9, "reasoning": "synthesized_search_results"}}

Your response (use tool_name "respond"):"""

        return f"TOOL_PROMPT|{prompt}"

    def _build_engage_prompt(self, tool_name: str, query: str, result: str, mode_name: str) -> str:
        """Build prompt for 'engage' followup - respond AND offer proactive follow-ups."""
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%A, %B %d, %Y")
        time_str = now.strftime("%I:%M %p")

        # Adjust engagement level based on mode
        if mode_name == "active-assistance":
            engagement_instruction = "Be proactive and suggest 2-3 relevant follow-up options the user might find helpful."
        elif mode_name == "observe":
            engagement_instruction = "Keep your response informative but concise. Only offer one follow-up if highly relevant."
        else:
            engagement_instruction = "Optionally offer 1-2 relevant follow-up questions or related information."

        prompt = f"""You just completed an internet search. Synthesize the results into a helpful response for the user.

CURRENT DATE: {date_str} at {time_str}
(Use this date context when interpreting results about "today" or recent events)

Original user request: "{query}"

Search Results:
{result}

=== Instructions ===
1. Extract the relevant information from the search results above
2. Provide a clear, direct answer to the user's question
3. {engagement_instruction}
4. If results seem outdated or incomplete, acknowledge this

IMPORTANT: You MUST use the "respond" tool to reply to the user. Do NOT use any other tool name.

Example JSON format (COPY THIS STRUCTURE EXACTLY):
{{"action": {{"tool_name": "respond", "params": {{"message": "The Broncos beat the Bills 31-7 in the AFC Divisional playoff. Would you like details on the top performers?"}}}}, "confidence": 0.9, "reasoning": "synthesized_search_results"}}

Your response (use tool_name "respond"):"""

        return f"TOOL_PROMPT|{prompt}"

    @staticmethod
    def _matches_phrase(text: str, phrase: str) -> bool:
        phrase = phrase.strip()
        if not phrase:
            return False
        # Use cached compiled pattern for performance
        compiled = _compile_phrase_pattern(phrase)
        return compiled.search(text) is not None

    @staticmethod
    def _normalize_phrases(value: Any, default: list[str]) -> list[str]:
        if not isinstance(value, list):
            return default
        normalized = [str(item).strip() for item in value if item]
        return normalized or default

    @staticmethod
    def _evaluate_simple_arithmetic(text: str) -> str | None:
        """Evaluate a simple two-operand arithmetic expression safely.

        Handles: "1 + 1", "what is 5 * 3", "calculate 10 / 2", "1+1"
        Returns formatted answer string or None if not arithmetic.
        """
        match = _ARITHMETIC_PATTERN.search(text)
        if not match:
            return None

        left_str, op, right_str = match.groups()

        # Guard: reject chained expressions like "1 + 2 + 3"
        remainder = text[match.end():].strip()
        if remainder and re.search(r"\d", remainder):
            return None

        try:
            left = float(left_str)
            right = float(right_str)
        except ValueError:
            return None

        op_func = _SIMPLE_OPS.get(op)
        if op_func is None:
            return None

        try:
            result = op_func(left, right)
        except (OverflowError, ZeroDivisionError):
            return None

        # NaN check (division by zero)
        if result != result:
            return "That operation is undefined (division by zero)."

        def _fmt(v: float) -> str:
            if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
                return str(int(v))
            return f"{v:g}"

        display_op = "*" if op == "x" else op
        return f"{_fmt(left)} {display_op} {_fmt(right)} = {_fmt(result)}"

    @staticmethod
    def _evaluate_unary_math(text: str) -> str | None:
        """Evaluate unary math expressions, optionally followed by a binary op.

        Simple:   "square root of 25"           → "√25 = 5"
        Compound: "square root of 25 plus 3"    → "√25 + 3 = 8"
                  "5 squared minus 10"           → "5² - 10 = 15"
        Returns formatted answer string or None if not a unary math pattern.
        """
        lower = text.lower()

        def _fmt(v: float) -> str:
            if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
                return str(int(v))
            return f"{v:g}"

        for pattern, op_type in _UNARY_MATH_PATTERNS:
            match = pattern.search(lower)
            if not match:
                continue

            try:
                value = float(match.group(1))
            except ValueError:
                continue

            # Evaluate the unary operation
            if op_type == "sqrt":
                if value < 0:
                    return "The square root of a negative number is not a real number."
                unary_result = value ** 0.5
                label = f"√{_fmt(value)}"
            elif op_type == "cbrt":
                unary_result = value ** (1 / 3) if value >= 0 else -((-value) ** (1 / 3))
                label = f"∛{_fmt(value)}"
            elif op_type == "squared":
                unary_result = value ** 2
                label = f"{_fmt(value)}²"
            elif op_type == "cubed":
                unary_result = value ** 3
                label = f"{_fmt(value)}³"
            else:
                continue

            # Check for trailing binary operation: "... plus 3", "... minus 2"
            remainder = lower[match.end():].strip()
            trailing = _TRAILING_OP_PATTERN.search(remainder) if remainder else None

            if trailing:
                # Groups: (plus_val, minus_val, times_val, divide_val)
                groups = trailing.groups()
                if groups[0] is not None:
                    rhs = float(groups[0])
                    final = unary_result + rhs
                    return f"{label} + {_fmt(rhs)} = {_fmt(final)}"
                elif groups[1] is not None:
                    rhs = float(groups[1])
                    final = unary_result - rhs
                    return f"{label} - {_fmt(rhs)} = {_fmt(final)}"
                elif groups[2] is not None:
                    rhs = float(groups[2])
                    final = unary_result * rhs
                    return f"{label} * {_fmt(rhs)} = {_fmt(final)}"
                elif groups[3] is not None:
                    rhs = float(groups[3])
                    if rhs == 0:
                        return "That operation is undefined (division by zero)."
                    final = unary_result / rhs
                    return f"{label} / {_fmt(rhs)} = {_fmt(final)}"

            # Simple unary — no trailing op
            return f"{label} = {_fmt(unary_result)}"

        return None

    def _generate_simple_answer(
        self, question: str, date_str: str, time_str: str
    ) -> str | None:
        """Generate answers for simple factual questions without LLM.

        Returns the answer string, or None if LLM is needed.
        Only matches SHORT, simple questions - complex questions with
        greetings or other phrases embedded should go to the LLM.
        """
        q = question.lower().strip()

        # Skip simple answers if the question is too long or complex
        # This prevents "Hello, can you create a python script..." from
        # matching the greeting shortcut
        word_count = len(q.split())
        if word_count > 8:  # More than ~8 words = too complex for simple answer
            return None
        fallback_config = get_fallback_responses()
        if not isinstance(fallback_config, dict):
            fallback_config = {}
        simple_answers = fallback_config.get("simple_answers")
        if not isinstance(simple_answers, dict):
            simple_answers = {}

        time_default = ["what time", "what's the time", "current time", "time is it"]
        date_default = [
            "what date",
            "what's the date",
            "what day",
            "today's date",
            "current date",
        ]
        identity_default = ["who are you", "what are you", "your name"]
        greeting_default = ["hello", "hi", "hey", "greetings"]
        wellbeing_default = ["how are you"]

        time_phrases = self._normalize_phrases(
            simple_answers.get("time_phrases"), time_default
        )
        date_phrases = self._normalize_phrases(
            simple_answers.get("date_phrases"), date_default
        )
        identity_phrases = self._normalize_phrases(
            simple_answers.get("identity_phrases"), identity_default
        )
        greeting_phrases = self._normalize_phrases(
            simple_answers.get("greeting_phrases"), greeting_default
        )
        wellbeing_phrases = self._normalize_phrases(
            simple_answers.get("wellbeing_phrases"), wellbeing_default
        )

        # Time questions
        if any(self._matches_phrase(q, phrase) for phrase in time_phrases):
            return f"The current time is {time_str}."

        # Date questions
        if any(self._matches_phrase(q, phrase) for phrase in date_phrases):
            return f"Today is {date_str}."

        # Combined date/time
        if "date and time" in q or "time and date" in q:
            return f"It's {time_str} on {date_str}."

        # Identity questions
        if any(self._matches_phrase(q, phrase) for phrase in identity_phrases):
            return "I'm Maxim, a robot assistant designed to understand reality and help people."

        # Greeting responses
        if any(self._matches_phrase(q, phrase) for phrase in greeting_phrases):
            return "Hello! How can I help you?"

        # How are you
        if any(self._matches_phrase(q, phrase) for phrase in wellbeing_phrases):
            return "I'm functioning well, thank you for asking. How can I assist you?"

        # Simple arithmetic (e.g., "what is 1+1", "calculate 5*3")
        arithmetic_result = self._evaluate_simple_arithmetic(q)
        if arithmetic_result:
            return arithmetic_result

        # Unary math (e.g., "square root of 25", "5 cubed")
        unary_result = self._evaluate_unary_math(q)
        if unary_result:
            return unary_result

        # Can't answer directly - need LLM
        return None

    def _update_avg_latency(self, latency_ms: float) -> None:
        """Update rolling average latency."""
        alpha = 0.1  # Smoothing factor
        self._avg_latency_ms = alpha * latency_ms + (1 - alpha) * self._avg_latency_ms

    @property
    def stats(self) -> dict[str, Any]:
        """Get worker statistics."""
        return {
            "requests_processed": self._requests_processed,
            "requests_dropped": self._requests_dropped,
            "avg_latency_ms": self._avg_latency_ms,
            "queue_size": self._request_queue.qsize(),
            "is_running": self._worker is not None and self._worker.is_alive(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fallback Behavior
# ─────────────────────────────────────────────────────────────────────────────


class FallbackBehavior:
    """Default behaviors when LLM is unavailable."""

    @staticmethod
    def get_fallback_action(
        mode_name: str,
        percept: Percept | None,
        internet_access: bool = False,
    ) -> dict[str, Any] | None:
        """Return safe default action for mode."""
        if percept is None:
            return None

        # Hard overrides always work
        if percept.hard_override:
            return {
                "tool_name": "maxim_command",
                "params": {"command": percept.hard_override},
            }

        # Mode-specific fallbacks
        if mode_name == "observe":
            # In observe mode, just focus on interests (passive)
            return {"tool_name": "focus_interests", "params": {}}

        if not percept.has_maxim_keyword:
            return None

        fallback_config = get_fallback_responses()
        if not isinstance(fallback_config, dict):
            fallback_config = {}
        llm_unavailable = fallback_config.get("llm_unavailable")
        if not isinstance(llm_unavailable, dict):
            llm_unavailable = {}

        normalized_mode = mode_name.replace("_", "-")
        default_messages = {
            "reflection": "I heard you. I'm in reflection mode - give me a moment to gather my thoughts.",
            "active-assistance": "I'm here but processing is delayed. One moment.",
        }
        message = llm_unavailable.get(normalized_mode)
        if message is None:
            message = default_messages.get(normalized_mode)

        if message:
            return {
                "tool_name": "speak",
                "params": {"text": str(message)},
            }

        # Default: do nothing safely
        return None

    @staticmethod
    def get_fallback_proposal(
        mode_name: str,
        percept: Percept | None,
        internet_access: bool = False,
    ) -> LLMProposal:
        """Create a fallback proposal when LLM is unavailable."""
        action = FallbackBehavior.get_fallback_action(
            mode_name, percept, internet_access
        )

        return LLMProposal(
            request_id=f"fallback-{time.time_ns()}",
            action=action,
            reasoning="LLM unavailable, using fallback behavior",
            strategy_used="fallback",
            confidence=0.5 if action else 0.0,
            mode_goal_achieved=False,
            citations=[],
            latency_ms=0.0,
            error="LLM unavailable",
        )
