"""Dedicated LLM worker thread for non-blocking inference.

The LLMWorker processes LLM requests asynchronously, ensuring the main control
loop never blocks on LLM latency. Includes fallback behaviors for when the
LLM is slow or unavailable.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from maxim.agents.autonomy import AutonomyLevel

if TYPE_CHECKING:
    from maxim.agents.bus import Percept, StructuredContext


logger = logging.getLogger(__name__)


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
    max_initiative: float = 1.0


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
    ):
        self._llm = llm
        self._stale_threshold = stale_threshold_s

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

        # Metrics
        self._requests_processed = 0
        self._requests_dropped = 0
        self._avg_latency_ms = 0.0

    def start(self) -> None:
        """Start the worker thread."""
        if self._worker is None or not self._worker.is_alive():
            self._stop_event.clear()
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
        if self._worker:
            self._worker.join(timeout=5.0)
            logger.info("LLM worker thread stopped")

    def submit_context(
        self,
        context: StructuredContext,
        mode: ModeInfo,
        autonomy_level: AutonomyLevel,
        strategies: list[StrategyInfo],
        internet_access: bool,
        internet_policy_summary: str,
        priority: int = 0,
    ) -> bool:
        """
        Submit context for LLM processing (non-blocking).

        Returns True if queued, False if dropped (queue full).
        Main loop should call this frequently; stale requests are pruned.
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
                )

            response = self._llm.generate_json(prompt, temperature=0.3, max_tokens=2048)

            latency_ms = (time.time() - start_time) * 1000
            self._update_avg_latency(latency_ms)

            if not response or not isinstance(response, dict):
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
                )

            return LLMProposal(
                request_id=request.request_id,
                action=response.get("action"),
                reasoning=response.get("reasoning", ""),
                strategy_used=response.get("strategy"),
                confidence=response.get("confidence", 0.5),
                mode_goal_achieved=response.get("mode_goal_achieved", False),
                citations=response.get("citations", []),
                latency_ms=latency_ms,
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
            )

    def _build_prompt(self, request: LLMRequest) -> str:
        """Build prompt from request.

        Note: This prompt is wrapped by the LLMRouter.generate_json() method
        with the appropriate chat format (ChatML, Mistral, etc.) based on
        the model's prompt_style configuration.
        """
        context = request.context

        # Check for pending user input that needs a response
        user_question = ""
        if context.cli_inputs:
            latest_input = context.cli_inputs[-1] if context.cli_inputs else None
            if latest_input and "maxim" in latest_input.lower():
                # Strip "maxim" prefix for cleaner question
                user_question = latest_input

        # For simple user questions, use a very explicit JSON-only format
        if user_question:
            # Extract just the question part after "maxim"
            question_text = user_question.lower().replace("maxim", "").strip()
            question_text = question_text.lstrip(",").lstrip(":").strip()
            if question_text.endswith("?"):
                question_text = question_text[:-1].strip()

            # Get current date/time for context
            now = datetime.now()
            date_str = now.strftime("%A, %B %d, %Y")  # e.g., "Friday, January 24, 2025"
            time_str = now.strftime("%I:%M %p")  # e.g., "11:30 PM"

            # Direct format for small models - no example to copy
            # The model fills in the ANSWER placeholder
            return f"""Today is {date_str}. The time is {time_str}.

User asks: "{question_text}"

Reply with JSON only:
{{"action":{{"tool_name":"respond","params":{{"message":"ANSWER"}}}},"reasoning":"question","confidence":0.9,"mode_goal_achieved":false}}

Replace ANSWER with your response to the user's question. Output the JSON now:"""

        # For non-question contexts, return None (no action needed)
        # This prevents unnecessary LLM calls for idle observations
        return ""

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

        if mode_name == "reflection":
            # In reflection mode, acknowledge direct address but stay introspective
            if percept.has_maxim_keyword:
                return {
                    "tool_name": "speak",
                    "params": {
                        "text": "I heard you. I'm in reflection mode - give me a moment to gather my thoughts."
                    },
                }
            # Stay quiet otherwise - focus on internal processing
            return None

        if mode_name in ("active-assistance", "active_assistance"):
            # In active mode with maxim keyword, acknowledge
            if percept.has_maxim_keyword:
                return {
                    "tool_name": "speak",
                    "params": {
                        "text": "I'm here but processing is delayed. One moment."
                    },
                }
            return None

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
