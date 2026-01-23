"""Flexible mode executor using LLM reasoning.

Instead of calling fixed functions, the mode executor uses the LLM
to reason about how to achieve the mode's goal.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from maxim.agents.autonomy import AutonomyLevel
from maxim.modes.definitions import ModeDefinition, get_mode, MODES
from maxim.modes.strategies import Strategy, StrategyLibrary, get_strategy_library

if TYPE_CHECKING:
    from maxim.agents.bus import StructuredContext
    from maxim.utils.internet_access import InternetAccessPolicy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModeStepResult:
    """Result of executing one step of mode behavior."""

    action: dict[str, Any] | None
    reasoning: str
    strategy_used: str | None = None
    confidence: float = 0.5
    mode_complete: bool = False
    citations: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LLM Protocol
# ─────────────────────────────────────────────────────────────────────────────


class LLMBackend(Protocol):
    """Protocol for LLM backends."""

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> dict[str, Any] | None:
        """Generate a JSON response."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Flexible Mode Executor
# ─────────────────────────────────────────────────────────────────────────────


class FlexibleModeExecutor:
    """Executes modes flexibly using LLM reasoning.

    Instead of fixed mode functions, this executor:
    1. Selects strategies appropriate for the mode and context
    2. Builds a prompt with mode goal, strategies, and context
    3. Uses the LLM to decide what action to take
    4. Tracks strategy outcomes for learning
    """

    def __init__(
        self,
        llm: LLMBackend,
        strategy_library: StrategyLibrary | None = None,
    ):
        self._llm = llm
        self._strategies = strategy_library or get_strategy_library()

    def execute_mode_step(
        self,
        mode: ModeDefinition | str,
        context: StructuredContext,
        autonomy_level: AutonomyLevel,
        internet_access: bool = False,
        internet_policy: InternetAccessPolicy | None = None,
    ) -> ModeStepResult:
        """Execute one step of mode behavior using LLM reasoning.

        Args:
            mode: Mode definition or mode name string
            context: Current structured context
            autonomy_level: Current autonomy level
            internet_access: Whether internet is enabled
            internet_policy: Internet access policy (for summary)

        Returns:
            ModeStepResult with action, reasoning, etc.
        """
        start_time = time.time()

        # Resolve mode if string
        if isinstance(mode, str):
            resolved_mode = get_mode(mode)
            if resolved_mode is None:
                return ModeStepResult(
                    action=None,
                    reasoning=f"Unknown mode: {mode}",
                    error=f"Unknown mode: {mode}",
                )
            mode = resolved_mode

        # Select strategies for this situation
        selected_strategies = self._strategies.select_strategies(mode, context)

        # Build prompt for LLM
        policy_summary = ""
        if internet_policy:
            policy_summary = internet_policy.summary()

        prompt = self._build_mode_prompt(
            mode,
            context,
            selected_strategies,
            autonomy_level,
            internet_access,
            policy_summary,
        )

        # Get LLM decision
        try:
            response = self._llm.generate_json(prompt, temperature=0.3)
            latency_ms = (time.time() - start_time) * 1000

            if not response or not isinstance(response, dict):
                return ModeStepResult(
                    action=None,
                    reasoning="LLM did not respond with valid JSON",
                    error="Invalid LLM response",
                    latency_ms=latency_ms,
                )

            return ModeStepResult(
                action=response.get("action"),
                reasoning=response.get("reasoning", ""),
                strategy_used=response.get("strategy"),
                confidence=float(response.get("confidence", 0.5)),
                mode_complete=bool(response.get("mode_goal_achieved", False)),
                citations=response.get("citations", []),
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Mode execution error: {e}")
            return ModeStepResult(
                action=None,
                reasoning="",
                error=str(e),
                latency_ms=latency_ms,
            )

    def record_outcome(
        self,
        strategy_name: str | None,
        success: bool,
    ) -> None:
        """Record strategy outcome for learning."""
        if strategy_name and self._strategies:
            self._strategies.record_outcome(strategy_name, success)

    def _build_mode_prompt(
        self,
        mode: ModeDefinition,
        context: StructuredContext,
        strategies: list[Strategy],
        autonomy_level: AutonomyLevel,
        internet_access: bool,
        internet_policy_summary: str,
    ) -> str:
        """Build prompt for LLM mode execution."""
        strategy_text = "\n".join(
            [
                f"- {s.name}: {s.description}\n  Approach: {s.approach_prompt}"
                for s in strategies
            ]
        )

        autonomy_text = {
            AutonomyLevel.PLANNING: "You are in PLANNING mode. Propose actions but do not execute. Explain your reasoning thoroughly.",
            AutonomyLevel.SUPERVISED: "You are in SUPERVISED mode. You may act within your allowed tools, but explain significant decisions.",
            AutonomyLevel.AUTONOMOUS: "You are in AUTONOMOUS mode. Act decisively toward the goal while respecting safety constraints.",
        }.get(autonomy_level, "Unknown autonomy level.")

        # Build context summary
        context_parts = []

        if context.active_goal:
            context_parts.append(f"- Active goal: {context.active_goal}")

        if context.detected_objects:
            obj_names = [
                obj.get("label", "unknown") for obj in context.detected_objects[:5]
            ]
            context_parts.append(f"- Detected objects: {', '.join(obj_names)}")

        if context.detected_people:
            context_parts.append(f"- People detected: {len(context.detected_people)}")

        if context.detected_speech:
            recent_speech = context.detected_speech[-1] if context.detected_speech else "(none)"
            context_parts.append(f"- Recent speech: {recent_speech}")

        if context.cli_inputs:
            recent_cli = context.cli_inputs[-1] if context.cli_inputs else "(none)"
            context_parts.append(f"- Recent CLI input: {recent_cli}")

        context_summary = "\n".join(context_parts) if context_parts else "(no significant context)"

        return f"""ROOT GOAL: {context.root_goal}

CURRENT MODE: {mode.name}
MODE GOAL: {mode.goal}
SUCCESS CRITERIA: {', '.join(mode.success_criteria)}

{mode.context_prompt}

AUTONOMY LEVEL: {autonomy_level.value}
{autonomy_text}

INTERNET ACCESS: {"enabled" if internet_access else "disabled"}
{internet_policy_summary if internet_access else "Internet tools are not available."}
CITATIONS REQUIRED WHEN USING WEB SOURCES.

AVAILABLE STRATEGIES:
{strategy_text if strategy_text else "(none available)"}

FORBIDDEN IN THIS MODE: {', '.join(mode.forbidden_tools) or '(none)'}
MAX INITIATIVE LEVEL: {mode.max_initiative}

CURRENT CONTEXT:
{context_summary}

Based on the mode goal, available strategies, and current context, decide:
1. What action (if any) should be taken?
2. Which strategy are you using?
3. What is your reasoning?
4. How confident are you? (0.0-1.0)
5. Is the mode goal achieved?

Respond with JSON:
{{
    "action": {{"tool_name": "...", "params": {{...}}}} or null,
    "strategy": "strategy_name or null",
    "reasoning": "why this action/inaction",
    "confidence": 0.0-1.0,
    "mode_goal_achieved": true/false,
    "citations": [{{"title": "...", "url": "..."}}]
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Mode Suggestion
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModeSuggestion:
    """Agent's proposal for a new or modified mode."""

    suggestion_type: str  # "new_mode", "modify_mode", "deprecate_mode"

    # For new/modified modes
    mode_definition: ModeDefinition | None = None

    # For modifications
    base_mode: str | None = None
    changes: dict[str, Any] = field(default_factory=dict)

    # Justification
    reasoning: str = ""
    observed_need: str = ""  # What situation prompted this
    expected_benefit: str = ""

    # Status
    status: str = "proposed"  # "proposed", "approved", "rejected", "testing"
    test_results: list[dict[str, Any]] = field(default_factory=list)


class ModeEvolutionSystem:
    """Manages agent-proposed mode changes.

    Allows the agent to propose new modes or modifications,
    which are then tested and approved by humans.
    """

    def __init__(self, mode_registry: dict[str, ModeDefinition] | None = None):
        self.modes = mode_registry if mode_registry is not None else dict(MODES)
        self.suggestions: list[ModeSuggestion] = []
        self.testing_modes: dict[str, ModeDefinition] = {}

    def propose_mode(
        self,
        suggestion: ModeSuggestion,
    ) -> None:
        """Agent proposes a new or modified mode."""
        suggestion.status = "proposed"
        self.suggestions.append(suggestion)

        logger.info(
            f"Mode suggestion: type={suggestion.suggestion_type}, "
            f"reasoning={suggestion.reasoning[:100]}"
        )

    def approve_for_testing(self, suggestion_index: int) -> str | None:
        """Human approves suggestion for testing.

        Returns the test mode name, or None if failed.
        """
        if suggestion_index >= len(self.suggestions):
            return None

        suggestion = self.suggestions[suggestion_index]
        suggestion.status = "testing"

        # Create test version of mode
        if suggestion.suggestion_type == "new_mode" and suggestion.mode_definition:
            test_name = f"_test_{suggestion.mode_definition.name}"
            self.testing_modes[test_name] = suggestion.mode_definition
            return test_name

        if suggestion.suggestion_type == "modify_mode" and suggestion.base_mode:
            base = self.modes.get(suggestion.base_mode)
            if base is None:
                return None
            modified = self._apply_changes(base, suggestion.changes)
            test_name = f"_test_{base.name}_v{len(self.suggestions)}"
            self.testing_modes[test_name] = modified
            return test_name

        return None

    def report_test_result(
        self,
        test_mode_name: str,
        success: bool,
        details: dict[str, Any],
    ) -> None:
        """Record results from testing a proposed mode."""
        for suggestion in self.suggestions:
            if suggestion.status == "testing":
                suggestion.test_results.append(
                    {
                        "success": success,
                        "details": details,
                        "timestamp": time.time(),
                    }
                )
                break

    def finalize_mode(self, test_mode_name: str, approve: bool) -> bool:
        """Human decision on tested mode.

        Returns True if mode was finalized successfully.
        """
        if test_mode_name not in self.testing_modes:
            return False

        mode = self.testing_modes.pop(test_mode_name)

        if approve:
            # Add to official modes
            self.modes[mode.name] = mode

            # Update suggestion status
            for suggestion in self.suggestions:
                if (
                    suggestion.mode_definition
                    and suggestion.mode_definition.name == mode.name
                ):
                    suggestion.status = "approved"
                    break

            logger.info(f"Mode approved: {mode.name}")
            return True
        else:
            for suggestion in self.suggestions:
                if (
                    suggestion.mode_definition
                    and suggestion.mode_definition.name == mode.name
                ):
                    suggestion.status = "rejected"
                    break

            logger.info(f"Mode rejected: {mode.name}")
            return False

    def _apply_changes(
        self, base: ModeDefinition, changes: dict[str, Any]
    ) -> ModeDefinition:
        """Apply changes to create a modified mode."""
        # Create a copy of the base mode
        modified = ModeDefinition(
            name=changes.get("name", base.name),
            goal=changes.get("goal", base.goal),
            success_criteria=changes.get("success_criteria", list(base.success_criteria)),
            forbidden_tools=changes.get("forbidden_tools", set(base.forbidden_tools)),
            max_initiative=changes.get("max_initiative", base.max_initiative),
            preferred_strategies=changes.get(
                "preferred_strategies", list(base.preferred_strategies)
            ),
            avoid_strategies=changes.get("avoid_strategies", list(base.avoid_strategies)),
            context_prompt=changes.get("context_prompt", base.context_prompt),
            outcome_memory_key=changes.get("outcome_memory_key", base.outcome_memory_key),
        )
        return modified

    def get_pending_suggestions(self) -> list[ModeSuggestion]:
        """Get suggestions awaiting approval."""
        return [s for s in self.suggestions if s.status == "proposed"]

    def get_testing_modes(self) -> list[str]:
        """Get modes currently being tested."""
        return list(self.testing_modes.keys())
