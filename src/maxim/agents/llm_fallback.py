"""Fallback behavior and simple answer generation for the LLM worker.

Contains:
- Regex patterns and static methods for arithmetic/math evaluation
- ReasoningCarryover buffer for decision+outcome tracking
- FallbackBehavior class for when LLM is unavailable
- Standalone functions for fallback/simple answer generation (delegated from LLMWorker)
"""

from __future__ import annotations

import functools
import re
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from maxim.agents.llm_types import LLMProposal
from maxim.utils.prompts import get_fallback_responses

if TYPE_CHECKING:
    from maxim.agents.bus import Percept

logger = logging.getLogger(__name__)


# Performance: Cache compiled regex patterns for phrase matching
@functools.lru_cache(maxsize=256)
def _compile_phrase_pattern(phrase: str) -> re.Pattern:
    """Compile and cache regex pattern for phrase matching."""
    return re.compile(rf"\b{re.escape(phrase)}\b")


# Simple arithmetic: "number operator number" (supports negatives, decimals, 'x' for multiply)
_ARITHMETIC_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x^%])\s*(-?\d+(?:\.\d+)?)")

_SIMPLE_OPS: dict[str, Any] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "x": lambda a, b: a * b,
    "/": lambda a, b: a / b if b != 0 else float("nan"),
    "^": lambda a, b: a**b,
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
                self._entries = self._entries[-self._max_entries :]

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
# Static Math Helpers
# ─────────────────────────────────────────────────────────────────────────────


def matches_phrase(text: str, phrase: str) -> bool:
    phrase = phrase.strip()
    if not phrase:
        return False
    # Use cached compiled pattern for performance
    compiled = _compile_phrase_pattern(phrase)
    return compiled.search(text) is not None


def normalize_phrases(value: Any, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return default
    normalized = [str(item).strip() for item in value if item]
    return normalized or default


def evaluate_simple_arithmetic(text: str) -> str | None:
    """Evaluate a simple two-operand arithmetic expression safely.

    Handles: "1 + 1", "what is 5 * 3", "calculate 10 / 2", "1+1"
    Returns formatted answer string or None if not arithmetic.
    """
    match = _ARITHMETIC_PATTERN.search(text)
    if not match:
        return None

    left_str, op, right_str = match.groups()

    # Guard: reject chained expressions like "1 + 2 + 3"
    remainder = text[match.end() :].strip()
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


def evaluate_unary_math(text: str) -> str | None:
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
            unary_result = value**0.5
            label = f"√{_fmt(value)}"
        elif op_type == "cbrt":
            unary_result = value ** (1 / 3) if value >= 0 else -((-value) ** (1 / 3))
            label = f"∛{_fmt(value)}"
        elif op_type == "squared":
            unary_result = value**2
            label = f"{_fmt(value)}²"
        elif op_type == "cubed":
            unary_result = value**3
            label = f"{_fmt(value)}³"
        else:
            continue

        # Check for trailing binary operation: "... plus 3", "... minus 2"
        remainder = lower[match.end() :].strip()
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


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Fallback/Simple Answer Functions
# ─────────────────────────────────────────────────────────────────────────────


def generate_simple_answer(question: str, date_str: str, time_str: str) -> str | None:
    """Generate answers for simple factual questions without LLM.

    Returns the answer string, or None if LLM is needed.
    Only matches SHORT, simple questions - complex questions with
    greetings or other phrases embedded should go to the LLM.
    """
    q = question.lower().strip()

    # Skip simple answers if the question is too long or complex
    word_count = len(q.split())
    if word_count > 8:
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

    time_phrases = normalize_phrases(simple_answers.get("time_phrases"), time_default)
    date_phrases = normalize_phrases(simple_answers.get("date_phrases"), date_default)
    identity_phrases = normalize_phrases(simple_answers.get("identity_phrases"), identity_default)
    greeting_phrases = normalize_phrases(simple_answers.get("greeting_phrases"), greeting_default)
    wellbeing_phrases = normalize_phrases(simple_answers.get("wellbeing_phrases"), wellbeing_default)

    # Time questions
    if any(matches_phrase(q, phrase) for phrase in time_phrases):
        return f"The current time is {time_str}."

    # Date questions
    if any(matches_phrase(q, phrase) for phrase in date_phrases):
        return f"Today is {date_str}."

    # Combined date/time
    if "date and time" in q or "time and date" in q:
        return f"It's {time_str} on {date_str}."

    # Identity questions
    if any(matches_phrase(q, phrase) for phrase in identity_phrases):
        return "I'm Maxim, a robot assistant designed to understand reality and help people."

    # Greeting responses
    if any(matches_phrase(q, phrase) for phrase in greeting_phrases):
        return "Hello! How can I help you?"

    # How are you
    if any(matches_phrase(q, phrase) for phrase in wellbeing_phrases):
        return "I'm functioning well, thank you for asking. How can I assist you?"

    # Simple arithmetic (e.g., "what is 1+1", "calculate 5*3")
    arithmetic_result = evaluate_simple_arithmetic(q)
    if arithmetic_result:
        return arithmetic_result

    # Unary math (e.g., "square root of 25", "5 cubed")
    unary_result = evaluate_unary_math(q)
    if unary_result:
        return unary_result

    # Can't answer directly - need LLM
    return None


def generate_llm_fallback(
    request: Any,
    robot_name: str = "maxim",
) -> dict[str, Any] | None:
    """Generate a fallback response when LLM fails to produce valid JSON.

    Returns an action dict or None if no fallback is appropriate.
    """
    # Use triggering_input if provided (already validated by agent loop)
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
    arithmetic_result = evaluate_simple_arithmetic(question)
    if arithmetic_result:
        return {
            "tool_name": "respond",
            "params": {"message": arithmetic_result},
        }

    # Try unary math (square root, cube root, squared, cubed)
    unary_result = evaluate_unary_math(question)
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
        action = FallbackBehavior.get_fallback_action(mode_name, percept, internet_access)

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
