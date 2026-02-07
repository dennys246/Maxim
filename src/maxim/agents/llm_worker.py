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
from typing import TYPE_CHECKING, Any, Protocol

from maxim.agents.autonomy import AutonomyLevel
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

if TYPE_CHECKING:
    from maxim.agents.bus import Percept, StructuredContext


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
        llm_timeout_s: float = 30.0,
        energy_tracker: "LLMEnergyTracker | None" = None,
    ):
        self._llm = llm
        self._stale_threshold = stale_threshold_s
        self._llm_timeout = llm_timeout_s
        self._energy_tracker = energy_tracker

        # Input: contexts waiting for LLM processing
        self._request_queue: queue.PriorityQueue[tuple[int, LLMRequest | None]] = (
            queue.PriorityQueue(maxsize=max_queue_size)
        )

        # Output: proposals ready for main loop
        self._proposal_queue: queue.Queue[LLMProposal] = queue.Queue()

        # Latest proposal (main loop can always get most recent)
        self._latest_proposal: LLMProposal | None = None
        self._proposal_lock = threading.Lock()

        # Worker thread
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        # Thread pool for timeout-wrapped LLM calls
        self._llm_executor: concurrent.futures.ThreadPoolExecutor | None = None

        # Metrics
        self._requests_processed = 0
        self._requests_dropped = 0
        self._avg_latency_ms = 0.0

    def start(self) -> None:
        """Start the worker thread."""
        if self._worker is None or not self._worker.is_alive():
            self._stop_event.clear()
            # Create thread pool for LLM calls with timeout
            self._llm_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="LLMCall"
            )
            self._worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="LLMWorker",
            )
            self._worker.start()
            logger.info("LLM worker thread started")

    def stop(self) -> None:
        """Stop the worker thread."""
        self._stop_event.set()
        # Unblock the queue with poison pill
        try:
            self._request_queue.put_nowait((0, None))
        except queue.Full:
            pass
        # Shutdown the LLM executor first (don't wait for pending tasks)
        if self._llm_executor is not None:
            try:
                self._llm_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python < 3.9 doesn't have cancel_futures
                self._llm_executor.shutdown(wait=False)
            self._llm_executor = None
        if self._worker:
            self._worker.join(timeout=2.0)
            if self._worker.is_alive():
                logger.warning("LLM worker thread did not stop in time")
            else:
                logger.info("LLM worker thread stopped")

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
            return None
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

        try:
            # Priority queue: lower number = higher priority
            # Negate priority so higher values are processed first
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
        with self._proposal_lock:
            proposal = self._latest_proposal
            self._latest_proposal = None
            return proposal

    def get_all_proposals(self) -> list[LLMProposal]:
        """Get all pending proposals (non-blocking)."""
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

    def _build_tool_aware_prompt(
        self,
        request: LLMRequest,
        question_text: str,
        date_str: str,
        time_str: str,
    ) -> str:
        """Build a comprehensive tool-aware prompt for complex reasoning.

        This prompt includes:
        - Mode context and goals
        - Available tools with descriptions
        - Recent observations and context
        - Agent states
        - Recent action outcomes
        """
        context = request.context
        mode = request.mode
        parts: list[str] = []

        # Pre-detect real-time data requests
        realtime_keywords = [
            "score", "game", "match", "playing", "vs", "versus",
            "weather", "temperature", "forecast",
            "news", "latest", "current", "today", "now", "live",
            "price", "cost", "stock", "bitcoin", "crypto",
            "broncos", "patriots", "lakers", "yankees",  # Common team names
        ]
        q_lower = question_text.lower() if question_text else ""
        is_realtime_request = any(kw in q_lower for kw in realtime_keywords)

        # PLANNING MODE BANNER - must be at the very top so LLM sees it first
        if request.autonomy_level == AutonomyLevel.PLANNING:
            parts.append("!" * 60)
            parts.append("!!! CRITICAL: PLANNING MODE - YOU MUST ASK PERMISSION FIRST !!!")
            parts.append("!" * 60)
            parts.append("")
            parts.append("DO NOT output raw JSON. You MUST follow this EXACT format:")
            parts.append("")
            parts.append("STEP 1: Write a proposal IN PLAIN ENGLISH asking for permission")
            parts.append("STEP 2: Write the EXACT delimiter: <|action_json|>")
            parts.append("STEP 3: Write the JSON object")
            parts.append("")
            parts.append("=== CORRECT FORMAT EXAMPLE 1 ===")
            parts.append("I'd like to search the internet for the current Broncos vs Patriots score.")
            parts.append("May I proceed with this search?")
            parts.append("<|action_json|>")
            parts.append('{"action": {"tool_name": "internet_search", "params": {"query": "Broncos vs Patriots score today"}}, "confidence": 0.9}')
            parts.append("")
            parts.append("=== CORRECT FORMAT EXAMPLE 2 ===")
            parts.append("To answer your question about the weather, I need to search online.")
            parts.append("Should I look up the current weather for you?")
            parts.append("<|action_json|>")
            parts.append('{"action": {"tool_name": "internet_search", "params": {"query": "current weather"}}, "confidence": 0.9}')
            parts.append("")
            parts.append("=== WRONG (DO NOT DO THIS) ===")
            parts.append('{"action": {"tool_name": "internet_search", ...}}  <-- NO! Missing proposal text!')
            parts.append("")
            parts.append("YOUR RESPONSE MUST START WITH PLAIN TEXT, NOT JSON!")
            parts.append("!" * 60)
            parts.append("")

        # If this is a real-time data request, add instruction
        if is_realtime_request:
            parts.append(">>> REAL-TIME DATA NEEDED <<<")
            if request.autonomy_level == AutonomyLevel.PLANNING:
                parts.append("Propose using 'internet_search' and ask for approval.")
            else:
                parts.append("Use 'internet_search' tool directly.")
            parts.append("")

        # MODIFICATION REQUEST - User wants to revise a previous action
        if request.pending_modification:
            import json
            mod = request.pending_modification
            parts.append("=" * 60)
            parts.append(">>> ACTION MODIFICATION REQUESTED <<<")
            parts.append("=" * 60)
            parts.append("")
            parts.append("The user requested a modification to the following proposed action:")
            parts.append("")
            parts.append("ORIGINAL ACTION:")
            original_action = mod.get("original_action", {})
            parts.append(f"  Tool: {mod.get('original_tool_name', 'unknown')}")
            parts.append(f"  Parameters: {json.dumps(original_action.get('params', {}), indent=4)}")
            parts.append(f"  Original reasoning: {mod.get('original_reasoning', 'not provided')}")
            parts.append("")
            parts.append("USER'S MODIFICATION REQUEST:")
            parts.append(f'  "{mod.get("user_modification", "")}"')
            parts.append("")
            parts.append("YOUR TASK: Revise the action based on the user's feedback.")
            parts.append("- Interpret what change the user wants")
            parts.append("- Keep the same tool if appropriate, or choose a different one")
            parts.append("- Update the parameters according to the user's request")
            parts.append("- Provide updated reasoning")
            parts.append("")
            parts.append("Respond with a REVISED action that incorporates the user's modification.")
            parts.append("=" * 60)
            parts.append("")

        # Foundational context (Constitution & Agent Rules)
        foundational = _load_foundational_context()
        if foundational:
            parts.append(foundational)
            parts.append("")

        # System context with new architecture (mode + strategy)
        parts.append(f"You are Maxim, a robot assistant.")
        parts.append(f"")
        parts.append(f"=== OPERATIONAL STATE ===")
        parts.append(f"Mode: {mode.name.upper()}")
        parts.append(f"Mode goal: {mode.goal}")

        # Add current strategy if available (new architecture)
        if request.current_strategy:
            from maxim.modes.definitions import STRATEGIES
            strategy = STRATEGIES.get(request.current_strategy)
            if strategy:
                parts.append(f"Strategy: {strategy.name.upper()} - {strategy.description}")
                parts.append(f"Strategy focus: {strategy.focus}")

        parts.append(f"Autonomy level: {request.autonomy_level.value if request.autonomy_level else 'unknown'}")

        # Processing state indicator
        if request.is_sleeping:
            parts.append(f"Processing state: SLEEP (minimal processing, monitoring for wake keywords)")
        else:
            parts.append(f"Processing state: AWAKE")

        parts.append(f"")
        parts.append(f"=== CURRENT DATE/TIME (IMPORTANT) ===")
        parts.append(f"Today is {date_str}")
        parts.append(f"Current time: {time_str}")
        parts.append(f"When searching for current events, scores, or news, ALWAYS include the date in your query.")
        parts.append(f"Example: Instead of 'Broncos game score', search 'Broncos game score {date_str}'")

        # Mode context prompt if available
        if mode.context_prompt:
            parts.append(f"\n{mode.context_prompt}")

        # Strategy context prompt if available (new architecture)
        if request.current_strategy:
            from maxim.modes.definitions import STRATEGIES
            strategy = STRATEGIES.get(request.current_strategy)
            if strategy and strategy.context_prompt:
                parts.append(f"\n{strategy.context_prompt}")

        # Context pool summary (accumulated observations)
        if request.context_pool_text:
            parts.append("\n=== Context ===")
            parts.append(request.context_pool_text)

        # Pre-fetched context (speculative pre-fetching for efficiency)
        if request.prefetch_context:
            parts.append("")
            parts.append(request.prefetch_context)
            if request.skip_exploration:
                parts.append("")
                parts.append("!" * 50)
                parts.append("!!! ONE-CALL MODE: WRITE DIRECTLY !!!")
                parts.append("!" * 50)
                parts.append("")
                parts.append("File discovery is COMPLETE. Do NOT use glob or read_file.")
                parts.append("Proceed DIRECTLY to your action:")
                parts.append("- For EXISTING file: write_file with overwrite=True")
                parts.append("- For NEW file: write_file (no overwrite needed)")
                parts.append("")
                parts.append("Your response should be the write_file action, NOT exploration.")
                parts.append("")

        # Recent percepts
        if context.current_percept:
            percept = context.current_percept
            parts.append("\n=== Current Observation ===")
            if percept.transcript_chunk:
                parts.append(f"Heard: \"{percept.transcript_chunk[:200]}\"")
            if percept.detections:
                objects = [d.get("label", "?") for d in percept.detections[:5]]
                parts.append(f"Visible objects: {', '.join(objects)}")
            if percept.cli_input:
                parts.append(f"User input: \"{percept.cli_input}\"")

        # Detected speech
        if context.detected_speech:
            parts.append("\n=== Recent Speech ===")
            for speech in context.detected_speech[-3:]:
                parts.append(f"- \"{speech[:100]}\"")

        # Conversation history (past user inputs and Maxim's responses)
        if request.conversation_history_text:
            parts.append("\n=== Conversation History ===")
            parts.append(request.conversation_history_text)

        # Agent states (if provided)
        if request.agent_states:
            parts.append("\n=== Agent States ===")
            for state in request.agent_states[-5:]:
                agent_name = state.get("agent", "unknown")
                agent_state = state.get("state", "unknown")
                goal = state.get("goal", "")
                parts.append(f"- {agent_name}: {agent_state}" + (f" (goal: {goal})" if goal else ""))

        # Recent outcomes (learning from past actions)
        if request.recent_outcomes:
            parts.append("\n=== Recent Action Outcomes ===")
            for outcome in request.recent_outcomes[-3:]:
                tool = outcome.get("tool", "?")
                success = "succeeded" if outcome.get("success") else "failed"
                result = outcome.get("result", "")[:50] if outcome.get("result") else ""
                parts.append(f"- {tool}: {success}" + (f" ({result})" if result else ""))

        # Available tools with params and examples
        parts.append("\n=== Available Tools ===")
        tool_list = sorted(request.available_tools)
        for tool_name in tool_list:
            tool_info = request.tool_descriptions.get(tool_name, {})
            if isinstance(tool_info, dict) and tool_info:
                desc = tool_info.get("description", "")
                params = tool_info.get("params", {})
                example = tool_info.get("example", {})
                parts.append(f"- {tool_name}: {desc}")
                if params:
                    param_strs = [f"{k}={v}" for k, v in params.items()]
                    parts.append(f"    REQUIRED params: {', '.join(param_strs)}")
                if example:
                    import json
                    parts.append(f"    Example: {json.dumps(example)}")
            elif isinstance(tool_info, str) and tool_info:
                parts.append(f"- {tool_name}: {tool_info}")
            else:
                parts.append(f"- {tool_name}")

        # Tool selection guidance
        parts.append("\n=== Tool Selection ===")
        parts.append("Choose the right tool based on what's needed:")
        parts.append("- AMBIGUOUS REQUEST (incomplete, unclear): Use 'respond' to ASK FOR CLARIFICATION")
        parts.append("- REAL-TIME DATA (scores, weather, news, prices): Use 'internet_search'")
        parts.append("- KNOWLEDGE from memory (what is X, explain Y): Use 'respond'")
        parts.append("- CREATE/WRITE FILES (create script, make file, save code): Use 'write_file'")
        parts.append("- VISUAL COMMANDS (look at, focus on, track): Use 'focus_interests', 'track_target'")
        parts.append("")
        parts.append("=== AMBIGUOUS REQUESTS ===")
        parts.append("If the user's request is incomplete or unclear (e.g., 'look up', 'search', 'find'):")
        parts.append("- DO NOT assume what they want")
        parts.append("- DO NOT start file exploration or web searches without context")
        parts.append("- USE 'respond' to ask: 'What would you like me to look up?' or 'What should I search for?'")
        parts.append("")
        parts.append("=== CRITICAL: Use Correct Parameters ===")
        parts.append("- 'respond': params={\"message\": \"your answer\"}")
        parts.append("- 'speak': params={\"text\": \"text to speak\"}")
        parts.append("- 'write_file': params={\"path\": \".maxim_sandbox/filename.py\", \"content\": \"file content\"}")
        parts.append("- 'internet_search': params={\"query\": \"search query\"}")
        parts.append("- NEVER use wrong param names! 'write_file' uses 'path' and 'content', NOT 'message'!")
        parts.append("")
        parts.append("=== FILE SANDBOX (CRITICAL) ===")
        parts.append("All file writes MUST use '.maxim_sandbox/' prefix:")
        parts.append("- CORRECT: '.maxim_sandbox/hello.py', '.maxim_sandbox/scripts/test.py'")
        parts.append("- WRONG: 'hello.py', 'scripts/test.py' (will FAIL!)")
        parts.append("")
        parts.append("=== BATCHED TOOL CALLS (EFFICIENT) ===")
        parts.append("For file modifications, batch exploration into ONE call using parallel_actions:")
        parts.append('{"action": {"tool_name": "glob", "params": {"pattern": ".maxim_sandbox/**/*.py"}},')
        parts.append(' "parallel_actions": [')
        parts.append('   {"tool_name": "read_file", "params": {"path": ".maxim_sandbox/target.py"}},')
        parts.append('   {"tool_name": "glob", "params": {"pattern": ".maxim_sandbox/**/*.py"}}')
        parts.append(' ],')
        parts.append(' "reasoning": "Batched exploration"}')
        parts.append("All parallel_actions execute together, then results return for your next decision.")

        # Inject coding guidelines based on request type
        # Use question_text, or fall back to modification text if this is a revision request
        guidelines_text = question_text
        if not guidelines_text and request.pending_modification:
            # For modification requests, use the user's modification text for guideline detection
            guidelines_text = request.pending_modification.get("user_modification", "")

        if guidelines_text:
            coding_context = build_coding_context(
                guidelines_text,
                include_sandbox_reminder=True,
                max_guidelines=2,
            )
            if coding_context:
                parts.append("")
                parts.append(coding_context)

        # User request
        if question_text:
            parts.append(f"\n=== User Request ===")
            parts.append(f"\"{question_text}\"")

        # Response format instructions - ordered by importance (most critical first)
        # This ensures truncation loses least important fields first
        parts.append("\n=== Instructions ===")

        # In PLANNING mode, remind about the required format
        if request.autonomy_level == AutonomyLevel.PLANNING:
            parts.append("!" * 40)
            parts.append("FINAL REMINDER: PLANNING MODE ACTIVE!")
            parts.append("Your response MUST be:")
            parts.append("1. Plain text proposal asking permission (NOT JSON!)")
            parts.append("2. The delimiter <|action_json|>")
            parts.append("3. Then the JSON")
            parts.append("START YOUR RESPONSE WITH PLAIN ENGLISH TEXT!")
            parts.append("!" * 40)
        else:
            parts.append("Respond with a compact JSON object. IMPORTANT: Put fields in this order:")
            parts.append('  "action": {"tool_name": "<tool>", "params": {...}}')
            parts.append('  "confidence": 0.0-1.0')
            parts.append('  "reasoning": "Brief explanation (1 sentence)"')
            parts.append("Keep response compact. Do not include optional fields.")

        # Combine all parts
        prompt_text = "\n".join(parts)

        # Return as TOOL_PROMPT prefix so router knows to handle differently
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
