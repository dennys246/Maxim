"""Strategy system for flexible mode execution.

Strategies are reusable patterns the agent can select and combine
to achieve mode goals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.agents.bus import StructuredContext
    from maxim.modes.definitions import ModeDefinition


@dataclass
class Strategy:
    """A reusable approach to achieving goals."""

    name: str
    description: str
    applicable_contexts: list[str]  # Mode names or "any"

    # What the strategy does (LLM guidance)
    approach_prompt: str

    # Expected outcomes
    success_indicators: list[str] = field(default_factory=list)
    failure_indicators: list[str] = field(default_factory=list)

    # Learning
    historical_success_rate: float = 0.5
    times_used: int = 0

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
    "wait_for_address": Strategy(
        name="wait_for_address",
        description="Wait until directly spoken to before responding",
        applicable_contexts=["reflection", "observe", "sleep"],
        approach_prompt="""Wait for the user to address you directly (by name or clear indication).
Don't interrupt or offer unsolicited input. When addressed, respond
promptly and helpfully.""",
        success_indicators=["user_addressed_maxim", "response_given"],
        failure_indicators=["interrupted_user", "unsolicited_response"],
    ),
    "respond_concisely": Strategy(
        name="respond_concisely",
        description="Give brief, focused responses",
        applicable_contexts=["reflection", "live", "any"],
        approach_prompt="""Provide concise, direct answers. Avoid unnecessary elaboration.
If more detail is needed, offer to expand. Respect the user's time.""",
        success_indicators=["user_satisfied", "no_followup_needed"],
        failure_indicators=["user_asked_for_more", "response_too_long"],
    ),
    "ask_clarifying_questions": Strategy(
        name="ask_clarifying_questions",
        description="Ask questions to understand the request better",
        applicable_contexts=["reflection", "active-assistance", "any"],
        approach_prompt="""When the request is ambiguous or could be interpreted multiple ways,
ask a clarifying question before acting. Be specific about what you need
to know. One question at a time.""",
        success_indicators=["clarification_received", "correct_action_taken"],
        failure_indicators=["wrong_assumption_made", "user_frustrated"],
    ),
    "anticipate_needs": Strategy(
        name="anticipate_needs",
        description="Predict what the user will need and prepare",
        applicable_contexts=["active-assistance"],
        approach_prompt="""Based on what the user is doing and has done, predict what they'll
need next. Prepare resources, gather information, or position yourself
to help. Offer assistance before they ask.""",
        success_indicators=["user_accepted_help", "prediction_correct"],
        failure_indicators=["prediction_wrong", "help_rejected"],
    ),
    "offer_suggestions": Strategy(
        name="offer_suggestions",
        description="Proactively suggest helpful actions or information",
        applicable_contexts=["active-assistance"],
        approach_prompt="""When you notice an opportunity to help, offer a suggestion.
Frame it as an option, not a directive. Be ready to back off if declined.""",
        success_indicators=["suggestion_accepted", "user_grateful"],
        failure_indicators=["suggestion_rejected", "user_annoyed"],
    ),
    "gather_information": Strategy(
        name="gather_information",
        description="Collect more data before acting",
        applicable_contexts=["any"],
        approach_prompt="""You need more information before acting. Use available sensors and
tools to gather data. Ask clarifying questions if appropriate.
Don't act on incomplete information.""",
        success_indicators=["information_gathered", "uncertainty_reduced"],
        failure_indicators=["acted_without_info", "wrong_assumption"],
    ),
    "explain_reasoning": Strategy(
        name="explain_reasoning",
        description="Verbalize thought process for transparency",
        applicable_contexts=["supervised", "planning", "train", "any"],
        approach_prompt="""Explain what you're thinking and why. Make your reasoning transparent
so humans can understand and correct if needed. This builds trust.""",
        success_indicators=["explanation_given", "user_understood"],
        failure_indicators=["acted_without_explanation"],
    ),
    "targeted_web_search": Strategy(
        name="targeted_web_search",
        description="Use internet search when an info gap blocks goal progress",
        applicable_contexts=["active-assistance", "research", "any"],
        approach_prompt="""If internet_access is enabled and policy allows, run a focused search.
Keep queries specific, fetch only a few sources, and include citations.
If internet_access is disabled, ask the user or proceed offline.""",
        success_indicators=["sources_cited", "information_gap_closed"],
        failure_indicators=["searched_without_need", "missing_citations", "policy_blocked"],
    ),
    "verify_with_sources": Strategy(
        name="verify_with_sources",
        description="Cross-check claims against multiple sources",
        applicable_contexts=["research", "any"],
        approach_prompt="""Use at least two sources when possible. Prefer primary sources.
Always include citations in the response metadata.
Note any disagreements between sources.""",
        success_indicators=["claims_verified", "citations_present", "multiple_sources"],
        failure_indicators=["single_source_only", "no_citations"],
    ),
    "watch_and_learn": Strategy(
        name="watch_and_learn",
        description="Observe patterns without intervening",
        applicable_contexts=["observe", "train"],
        approach_prompt="""Focus on understanding patterns in what you observe.
Note recurring events, user behaviors, and environmental changes.
Store observations for later analysis.""",
        success_indicators=["patterns_identified", "no_interference"],
        failure_indicators=["interrupted_observation", "missed_pattern"],
    ),
    "minimal_processing": Strategy(
        name="minimal_processing",
        description="Reduce activity to conserve resources",
        applicable_contexts=["sleep"],
        approach_prompt="""Minimize all non-essential processing. Keep only audio monitoring
active for wake commands. Ignore most stimuli.""",
        success_indicators=["low_resource_usage", "wake_command_detected"],
        failure_indicators=["unnecessary_processing", "missed_wake_command"],
    ),
    "listen_for_wake": Strategy(
        name="listen_for_wake",
        description="Monitor for wake commands while inactive",
        applicable_contexts=["sleep"],
        approach_prompt="""Listen specifically for wake commands like 'Maxim, wake up',
'Hey Maxim', or 'I need you'. Ignore all other audio.""",
        success_indicators=["wake_command_detected", "quick_wake_response"],
        failure_indicators=["false_wake", "missed_wake_command"],
    ),
    "observe_demonstrations": Strategy(
        name="observe_demonstrations",
        description="Learn from user demonstrations",
        applicable_contexts=["train"],
        approach_prompt="""Watch carefully as the user demonstrates a behavior or task.
Note the sequence of actions, the conditions, and the outcomes.
Ask for repetition if needed.""",
        success_indicators=["demonstration_recorded", "pattern_extracted"],
        failure_indicators=["missed_demonstration", "wrong_pattern"],
    ),
    "request_feedback": Strategy(
        name="request_feedback",
        description="Ask the user for feedback on actions",
        applicable_contexts=["train", "supervised"],
        approach_prompt="""After taking an action, ask if it was correct. Accept corrections
gracefully. Use feedback to improve future behavior.""",
        success_indicators=["feedback_received", "improvement_noted"],
        failure_indicators=["feedback_ignored", "defensive_response"],
    ),
    # ─────────────────────────────────────────────────────────────────────────
    # Exploration Strategies
    # ─────────────────────────────────────────────────────────────────────────
    "novelty_exploration": Strategy(
        name="novelty_exploration",
        description="Systematically explore novel objects in the visual field",
        applicable_contexts=["exploration", "any"],
        approach_prompt="""Focus on what's new and unfamiliar:
1. Query novelty_track(action="query", top_k=5) to get novelty rankings
2. Select the most novel object (highest novelty_score)
3. Use novelty_track(action="track") to center vision on it
4. Observe and gather information about the object
5. Store findings in memory
6. Repeat with next most novel object

Prioritize objects you haven't seen before (is_new=True).
Objects become familiar over time (novelty_score decays).
Respect motion limits and cooldowns; avoid blocking the control loop.
Keep the pattern modality-agnostic so it can apply to audio or other sensors.""",
        success_indicators=["novel_object_tracked", "information_gathered"],
        failure_indicators=["stuck_on_familiar_objects", "no_movement"],
    ),
    "curiosity_driven_search": Strategy(
        name="curiosity_driven_search",
        description="Search the internet based on what you observe",
        applicable_contexts=["exploration", "research"],
        approach_prompt="""When you observe something interesting:
1. Identify what you're curious about
2. Formulate a specific search query
3. Use internet_search to find information
4. Use http_fetch to read relevant pages
5. Synthesize findings and store in memory
6. Consider how this relates to your exploration focus

Only search if internet_access is enabled and policy allows.
Enforce allowlists, budgets, and content-type checks; sanitize queries.""",
        success_indicators=["relevant_sources_found", "knowledge_gained"],
        failure_indicators=["irrelevant_searches", "policy_blocked"],
    ),
    "autonomous_analysis": Strategy(
        name="autonomous_analysis",
        description="Write and execute Python scripts to analyze observations",
        applicable_contexts=["exploration", "any"],
        approach_prompt="""When you have data to analyze:
1. Assess if analysis would be valuable
2. Write a focused Python script using allowed imports
3. Use sandbox tools to execute safely
4. Review results and store insights
5. Consider if results warrant model training

Available imports: numpy, pandas, scipy, sklearn, matplotlib, cv2
Scripts run in sandbox with resource limits and no network access.
Only run if policy allows scripts; execute via a worker queue.
Summarize outputs for context instead of attaching large artifacts.""",
        success_indicators=["analysis_completed", "insights_generated"],
        failure_indicators=["script_failed", "no_actionable_insights"],
    ),
    "incremental_training": Strategy(
        name="incremental_training",
        description="Propose and execute model training when useful patterns emerge",
        applicable_contexts=["exploration", "train"],
        approach_prompt="""When exploration reveals useful training opportunities:
1. Identify the model that could benefit (vision, motor, memory)
2. Prepare training data from observations
3. Write training script with proper validation
4. Request appropriate autonomy level for execution
5. Monitor training progress
6. Evaluate improvement

Training is expensive - only propose when:
- Clear pattern of useful data exists
- Expected improvement justifies compute cost
- Current model shows suboptimal behavior on this case
Only run if policy allows training and GPU is available; never block the control loop.""",
        success_indicators=["model_improved", "training_converged"],
        failure_indicators=["training_diverged", "no_improvement"],
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
            applicable = (
                "any" in strategy.applicable_contexts
                or mode_name in strategy.applicable_contexts
            )

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
            (1 - learning_rate) * strategy.historical_success_rate
            + learning_rate * target
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize library state for persistence."""
        return {
            "strategies": {
                name: strategy.to_dict()
                for name, strategy in self.strategies.items()
            }
        }

    def load_stats(self, data: dict[str, Any]) -> None:
        """Load strategy statistics from persisted data."""
        strategies_data = data.get("strategies", {})
        for name, stats in strategies_data.items():
            if name in self.strategies:
                self.strategies[name].historical_success_rate = float(
                    stats.get("historical_success_rate", 0.5)
                )
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
