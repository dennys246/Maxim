"""Dedicated LLM worker thread for non-blocking inference.

The LLMWorker processes LLM requests asynchronously, ensuring the main control
loop never blocks on LLM latency. Includes fallback behaviors for when the
LLM is slow or unavailable.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from maxim.agents.autonomy import AutonomyLevel
from maxim.utils.prompts import get_fallback_responses

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
            available -= {"web_search", "http_fetch", "internet_search"}

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

    # Multi-step action support
    next_actions: list[dict[str, Any]] = field(default_factory=list)

    def get_all_actions(self) -> list[dict[str, Any]]:
        """Get the primary action followed by any next_actions."""
        actions = []
        if self.action:
            actions.append(self.action)
        actions.extend(self.next_actions)
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
        *,
        available_tools: set[str] | None = None,
        tool_descriptions: dict[str, str] | None = None,
        context_pool_text: str = "",
        agent_states: list[dict[str, Any]] | None = None,
        recent_outcomes: list[dict[str, Any]] | None = None,
        use_tool_prompting: bool = False,
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
                    )
                except Exception:
                    pass  # Fall through to LLM if parse fails

            # Use mode-specific max tokens for dynamic response length
            max_tokens = request.mode.max_response_tokens
            response = self._llm.generate_json(prompt, temperature=0.3, max_tokens=max_tokens)

            latency_ms = (time.time() - start_time) * 1000
            self._update_avg_latency(latency_ms)

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
                )

            # Extract next_actions if present
            next_actions = response.get("next_actions", [])
            if not isinstance(next_actions, list):
                next_actions = []

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

    def _generate_llm_fallback(self, request: LLMRequest) -> dict[str, Any] | None:
        """Generate a fallback response when LLM fails to produce valid JSON.

        Returns an action dict or None if no fallback is appropriate.
        """
        context = request.context
        if not context.cli_inputs:
            return None

        latest_input = context.cli_inputs[-1] if context.cli_inputs else None
        if not latest_input or "maxim" not in latest_input.lower():
            return None

        # Extract the question
        question = latest_input.lower().replace("maxim", "").strip()
        question = question.lstrip(",").lstrip(":").strip()

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

        # Check for pending user input that needs a response
        user_question = ""
        if context.cli_inputs:
            latest_input = context.cli_inputs[-1] if context.cli_inputs else None
            if latest_input and "maxim" in latest_input.lower():
                user_question = latest_input

        if not user_question:
            # No user input - check if we should still process (exploration, etc.)
            if not request.use_tool_prompting:
                return ""
            # For tool prompting without user input, continue to build context-only prompt

        # Extract question text
        question_text = ""
        if user_question:
            question_text = user_question.lower().replace("maxim", "").strip()
            question_text = question_text.lstrip(",").lstrip(":").strip()
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
                return f'{{"action":{{"tool_name":"respond","params":{{"message":"{answer}"}}}},"reasoning":"direct_answer","confidence":0.95,"mode_goal_achieved":false}}'

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

        # System context
        parts.append(f"You are Maxim, a robot assistant. Current mode: {mode.name}")
        parts.append(f"Mode goal: {mode.goal}")
        parts.append(f"Current time: {time_str} on {date_str}")

        # Mode context prompt if available
        if mode.context_prompt:
            parts.append(f"\n{mode.context_prompt}")

        # Context pool summary (accumulated observations)
        if request.context_pool_text:
            parts.append("\n=== Context ===")
            parts.append(request.context_pool_text)

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

        # Available tools
        parts.append("\n=== Available Tools ===")
        tool_list = sorted(request.available_tools)
        for tool_name in tool_list:
            if tool_name in request.tool_descriptions:
                parts.append(f"- {tool_name}: {request.tool_descriptions[tool_name]}")
            else:
                parts.append(f"- {tool_name}")

        # User request
        if question_text:
            parts.append(f"\n=== User Request ===")
            parts.append(f"\"{question_text}\"")

        # Response format instructions
        parts.append("\n=== Instructions ===")
        parts.append("Respond with a JSON object containing:")
        parts.append('  "action": {"tool_name": "<tool>", "params": {...}} or null if no action needed')
        parts.append('  "reasoning": "Brief explanation of your decision"')
        parts.append('  "confidence": 0.0-1.0')
        parts.append('  "next_actions": [optional list of follow-up actions to propose]')

        # Add multi-step hint if appropriate
        if len(tool_list) > 3:
            parts.append("\nYou may propose multiple actions in sequence using 'next_actions'.")

        # Combine all parts
        prompt_text = "\n".join(parts)

        # Return as TOOL_PROMPT prefix so router knows to handle differently
        return f"TOOL_PROMPT|{prompt_text}"

    @staticmethod
    def _matches_phrase(text: str, phrase: str) -> bool:
        phrase = phrase.strip()
        if not phrase:
            return False
        pattern = rf"\b{re.escape(phrase)}\b"
        return re.search(pattern, text) is not None

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
        """
        q = question.lower().strip()
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
