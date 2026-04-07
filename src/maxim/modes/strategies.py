"""Strategy system for flexible mode execution.

Strategies are reusable patterns the agent can select and combine
to achieve mode goals.

Prompts are loaded from data/prompts/strategies/ for easy editing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from maxim.utils.prompts import get_strategy_prompt

if TYPE_CHECKING:
    from maxim.agents.bus import StructuredContext
    from maxim.modes.definitions import ModeDefinition


# Strategy category mapping for prompt loading
STRATEGY_CATEGORIES: dict[str, str] = {
    # Reactive strategies
    "wait_for_address": "reactive",
    "respond_concisely": "reactive",
    "ask_clarifying_questions": "reactive",
    "gather_information": "reactive",
    "explain_reasoning": "reactive",
    "minimal_processing": "reactive",
    "listen_for_wake": "reactive",
    # Proactive strategies
    "anticipate_needs": "proactive",
    "offer_suggestions": "proactive",
    "targeted_web_search": "proactive",
    "verify_with_sources": "proactive",
    "help_humans_proactively": "proactive",
    # Learning strategies
    "watch_and_learn": "learning",
    "observe_demonstrations": "learning",
    "request_feedback": "learning",
    "incremental_training": "learning",
    "self_reflect_intent": "learning",
    # Exploration strategies
    "novelty_exploration": "exploration",
    "curiosity_driven_search": "exploration",
    "autonomous_analysis": "exploration",
    "adaptive_engagement": "exploration",
    "understand_world": "exploration",
}


@dataclass
class Strategy:
    """A reusable approach to achieving goals."""

    name: str
    description: str
    applicable_contexts: list[str]  # Mode names or "any"

    # What the strategy does (LLM guidance) - loaded from file if empty
    approach_prompt: str = ""

    # Expected outcomes
    success_indicators: list[str] = field(default_factory=list)
    failure_indicators: list[str] = field(default_factory=list)

    # Learning
    historical_success_rate: float = 0.5
    times_used: int = 0

    def __post_init__(self) -> None:
        """Load prompt from file if not provided inline."""
        if not self.approach_prompt:
            category = STRATEGY_CATEGORIES.get(self.name, "")
            self.approach_prompt = get_strategy_prompt(self.name, category)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "applicable_contexts": self.applicable_contexts,
            "approach_prompt": self.approach_prompt,
            "success_indicators": self.success_indicators,
            "failure_indicators": self.failure_indicators,
            "historical_success_rate": self.historical_success_rate,
            "times_used": self.times_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Strategy:
        """Deserialize from dictionary."""
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            applicable_contexts=list(data.get("applicable_contexts", [])),
            approach_prompt=str(data.get("approach_prompt", "")),
            success_indicators=list(data.get("success_indicators", [])),
            failure_indicators=list(data.get("failure_indicators", [])),
            historical_success_rate=float(data.get("historical_success_rate", 0.5)),
            times_used=int(data.get("times_used", 0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Built-in Strategies
# ─────────────────────────────────────────────────────────────────────────────


BUILTIN_STRATEGIES: dict[str, Strategy] = {
    # Prompts are loaded from data/prompts/strategies/<category>/<name>.txt
    # Reactive strategies
    "wait_for_address": Strategy(
        name="wait_for_address",
        description="Wait until directly spoken to before responding",
        applicable_contexts=["reflection", "observe", "sleep"],
        success_indicators=["user_addressed_maxim", "response_given"],
        failure_indicators=["interrupted_user", "unsolicited_response"],
    ),
    "respond_concisely": Strategy(
        name="respond_concisely",
        description="Give brief, focused responses",
        applicable_contexts=["reflection", "live", "any"],
        success_indicators=["user_satisfied", "no_followup_needed"],
        failure_indicators=["user_asked_for_more", "response_too_long"],
    ),
    "ask_clarifying_questions": Strategy(
        name="ask_clarifying_questions",
        description="Ask questions to understand the request better",
        applicable_contexts=["reflection", "active-assistance", "any"],
        success_indicators=["clarification_received", "correct_action_taken"],
        failure_indicators=["wrong_assumption_made", "user_frustrated"],
    ),
    "gather_information": Strategy(
        name="gather_information",
        description="Collect more data before acting",
        applicable_contexts=["any"],
        success_indicators=["information_gathered", "uncertainty_reduced"],
        failure_indicators=["acted_without_info", "wrong_assumption"],
    ),
    "explain_reasoning": Strategy(
        name="explain_reasoning",
        description="Verbalize thought process for transparency",
        applicable_contexts=["supervised", "planning", "train", "any"],
        success_indicators=["explanation_given", "user_understood"],
        failure_indicators=["acted_without_explanation"],
    ),
    "minimal_processing": Strategy(
        name="minimal_processing",
        description="Reduce activity to conserve resources",
        applicable_contexts=["sleep"],
        success_indicators=["low_resource_usage", "wake_command_detected"],
        failure_indicators=["unnecessary_processing", "missed_wake_command"],
    ),
    "listen_for_wake": Strategy(
        name="listen_for_wake",
        description="Monitor for wake commands while inactive",
        applicable_contexts=["sleep"],
        success_indicators=["wake_command_detected", "quick_wake_response"],
        failure_indicators=["false_wake", "missed_wake_command"],
    ),
    # Proactive strategies
    "anticipate_needs": Strategy(
        name="anticipate_needs",
        description="Predict what the user will need and prepare",
        applicable_contexts=["active-assistance"],
        success_indicators=["user_accepted_help", "prediction_correct"],
        failure_indicators=["prediction_wrong", "help_rejected"],
    ),
    "offer_suggestions": Strategy(
        name="offer_suggestions",
        description="Proactively suggest helpful actions or information",
        applicable_contexts=["active-assistance"],
        success_indicators=["suggestion_accepted", "user_grateful"],
        failure_indicators=["suggestion_rejected", "user_annoyed"],
    ),
    "targeted_web_search": Strategy(
        name="targeted_web_search",
        description="Use internet search when an info gap blocks goal progress",
        applicable_contexts=["active-assistance", "research", "any"],
        success_indicators=["sources_cited", "information_gap_closed"],
        failure_indicators=["searched_without_need", "missing_citations", "policy_blocked"],
    ),
    "verify_with_sources": Strategy(
        name="verify_with_sources",
        description="Cross-check claims against multiple sources",
        applicable_contexts=["research", "any"],
        success_indicators=["claims_verified", "citations_present", "multiple_sources"],
        failure_indicators=["single_source_only", "no_citations"],
    ),
    "help_humans_proactively": Strategy(
        name="help_humans_proactively",
        description="Identify opportunities to help before being asked",
        applicable_contexts=["live", "active-assistance"],
        success_indicators=["help_accepted", "human_grateful", "need_anticipated"],
        failure_indicators=["help_rejected", "too_intrusive", "need_missed"],
    ),
    # Learning strategies
    "watch_and_learn": Strategy(
        name="watch_and_learn",
        description="Observe patterns without intervening",
        applicable_contexts=["observe", "train"],
        success_indicators=["patterns_identified", "no_interference"],
        failure_indicators=["interrupted_observation", "missed_pattern"],
    ),
    "observe_demonstrations": Strategy(
        name="observe_demonstrations",
        description="Learn from user demonstrations",
        applicable_contexts=["train"],
        success_indicators=["demonstration_recorded", "pattern_extracted"],
        failure_indicators=["missed_demonstration", "wrong_pattern"],
    ),
    "request_feedback": Strategy(
        name="request_feedback",
        description="Ask the user for feedback on actions",
        applicable_contexts=["train", "supervised"],
        success_indicators=["feedback_received", "improvement_noted"],
        failure_indicators=["feedback_ignored", "defensive_response"],
    ),
    "incremental_training": Strategy(
        name="incremental_training",
        description="Propose and execute model training when useful patterns emerge",
        applicable_contexts=["exploration", "train"],
        success_indicators=["model_improved", "training_converged"],
        failure_indicators=["training_diverged", "no_improvement"],
    ),
    "self_reflect_intent": Strategy(
        name="self_reflect_intent",
        description="Reflect on current live mode intent and consider adjustments",
        applicable_contexts=["reflection"],
        success_indicators=["insight_recorded", "gradual_improvement", "mission_alignment"],
        failure_indicators=["no_reflection", "drastic_changes", "mission_drift"],
    ),
    # Exploration strategies
    "novelty_exploration": Strategy(
        name="novelty_exploration",
        description="Systematically explore novel objects in the visual field",
        applicable_contexts=["exploration", "any"],
        success_indicators=["novel_object_tracked", "information_gathered"],
        failure_indicators=["stuck_on_familiar_objects", "no_movement"],
    ),
    "curiosity_driven_search": Strategy(
        name="curiosity_driven_search",
        description="Search the internet based on what you observe",
        applicable_contexts=["exploration", "research"],
        success_indicators=["relevant_sources_found", "knowledge_gained"],
        failure_indicators=["irrelevant_searches", "policy_blocked"],
    ),
    "autonomous_analysis": Strategy(
        name="autonomous_analysis",
        description="Write and execute Python scripts to analyze observations",
        applicable_contexts=["exploration", "any"],
        success_indicators=["analysis_completed", "insights_generated"],
        failure_indicators=["script_failed", "no_actionable_insights"],
    ),
    "adaptive_engagement": Strategy(
        name="adaptive_engagement",
        description="Adjust engagement level based on context and feedback",
        applicable_contexts=["live", "any"],
        success_indicators=["appropriate_engagement", "positive_feedback", "help_accepted"],
        failure_indicators=["over_engagement", "under_engagement", "user_annoyed"],
    ),
    "understand_world": Strategy(
        name="understand_world",
        description="Build understanding of the environment and context",
        applicable_contexts=["live", "observe", "exploration"],
        success_indicators=["knowledge_gained", "patterns_identified", "context_understood"],
        failure_indicators=["knowledge_gaps", "missed_observations", "wrong_assumptions"],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Library
# ─────────────────────────────────────────────────────────────────────────────


class StrategyLibrary:
    """Collection of strategies the agent can select from."""

    def __init__(self, load_builtin: bool = True):
        self.strategies: dict[str, Strategy] = {}

        if load_builtin:
            self._load_builtin_strategies()

    def _load_builtin_strategies(self) -> None:
        """Load built-in strategies."""
        for name, strategy in BUILTIN_STRATEGIES.items():
            self.strategies[name] = strategy

    def add_strategy(self, strategy: Strategy) -> None:
        """Add a custom strategy."""
        self.strategies[strategy.name] = strategy

    def get_strategy(self, name: str) -> Strategy | None:
        """Get a strategy by name."""
        return self.strategies.get(name)

    def list_strategies(self) -> list[str]:
        """Get list of available strategy names."""
        return list(self.strategies.keys())

    def select_strategies(
        self,
        mode: ModeDefinition,
        context: StructuredContext | None = None,
        max_strategies: int = 3,
    ) -> list[Strategy]:
        """Select appropriate strategies for current situation.

        Args:
            mode: The current mode definition
            context: Optional context for smarter selection
            max_strategies: Maximum strategies to return

        Returns:
            List of selected strategies, ordered by relevance
        """
        candidates: list[tuple[Strategy, float]] = []

        for strategy in self.strategies.values():
            # Check if applicable to current mode
            mode_name = mode.name.lower().replace("_", "-")
            applicable = "any" in strategy.applicable_contexts or mode_name in strategy.applicable_contexts

            if not applicable:
                continue

            # Calculate score
            score = strategy.historical_success_rate

            # Boost if in preferred list
            if strategy.name in mode.preferred_strategies:
                score += 0.2

            # Penalize if in avoid list
            if strategy.name in mode.avoid_strategies:
                score -= 0.3

            # Context-based adjustments
            if context:
                # Boost web search if internet is available
                if strategy.name == "targeted_web_search" and context.internet_access:
                    score += 0.1

                # Boost clarifying questions if there's ambiguous input
                if strategy.name == "ask_clarifying_questions" and context.detected_speech:
                    score += 0.05

            candidates.append((strategy, score))

        # Sort by score (highest first) and return top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [strategy for strategy, _ in candidates[:max_strategies]]

    def record_outcome(
        self,
        strategy_name: str,
        success: bool,
        learning_rate: float = 0.1,
    ) -> None:
        """Record strategy outcome and update success rate.

        Args:
            strategy_name: Name of the strategy used
            success: Whether the strategy succeeded
            learning_rate: How much to update the success rate
        """
        strategy = self.strategies.get(strategy_name)
        if strategy is None:
            return

        strategy.times_used += 1

        # Exponential moving average update
        target = 1.0 if success else 0.0
        strategy.historical_success_rate = (
            1 - learning_rate
        ) * strategy.historical_success_rate + learning_rate * target

    def to_dict(self) -> dict[str, Any]:
        """Serialize library state for persistence."""
        return {"strategies": {name: strategy.to_dict() for name, strategy in self.strategies.items()}}

    def load_stats(self, data: dict[str, Any]) -> None:
        """Load strategy statistics from persisted data."""
        strategies_data = data.get("strategies", {})
        for name, stats in strategies_data.items():
            if name in self.strategies:
                self.strategies[name].historical_success_rate = float(stats.get("historical_success_rate", 0.5))
                self.strategies[name].times_used = int(stats.get("times_used", 0))


# ─────────────────────────────────────────────────────────────────────────────
# Global Instance
# ─────────────────────────────────────────────────────────────────────────────


_global_library: StrategyLibrary | None = None


def get_strategy_library(create_if_missing: bool = True) -> StrategyLibrary | None:
    """Get the global strategy library instance."""
    global _global_library

    if _global_library is None and create_if_missing:
        _global_library = StrategyLibrary()

    return _global_library


def set_strategy_library(library: StrategyLibrary) -> None:
    """Set the global strategy library instance."""
    global _global_library
    _global_library = library
