"""Agent-definable live mode intent.

Allows the agent to evolve its own behavioral intent for live mode
through reflection and experience.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.modes.definitions import ModeDefinition

logger = logging.getLogger(__name__)


# Bounds for agent-adjustable parameters
MIN_INITIATIVE = 0.3
MAX_INITIATIVE = 0.8
DEFAULT_INITIATIVE = 0.6

# Primary goal - immutable
LIVE_MODE_PRIMARY_GOAL = "Understand reality and help people"


@dataclass
class LiveModeIntent:
    """Agent-defined behavioral intent for live mode.

    The agent can modify these values through the DefineLiveModeIntentTool.
    Changes are persisted and loaded on startup.
    """

    # Core intent - agent's self-defined purpose
    context_prompt_extension: str = ""  # Extends base, doesn't replace
    goal_extension: str = ""  # Additional goal context

    # Tunable parameters (within bounds)
    max_initiative: float = DEFAULT_INITIATIVE

    # Strategy preferences
    preferred_strategies: list[str] = field(default_factory=list)
    avoid_strategies: list[str] = field(default_factory=list)

    # Learning and tracking
    historical_outcomes: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1
    created_at: float = field(default_factory=time.time)
    last_modified_at: float = field(default_factory=time.time)
    modification_count: int = 0

    # Reflection tracking
    last_reflection_at: float | None = None
    reflection_insights: list[str] = field(default_factory=list)

    def validate_initiative(self, value: float) -> float:
        """Clamp initiative to safe bounds."""
        return max(MIN_INITIATIVE, min(MAX_INITIATIVE, value))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "context_prompt_extension": self.context_prompt_extension,
            "goal_extension": self.goal_extension,
            "max_initiative": self.max_initiative,
            "preferred_strategies": self.preferred_strategies,
            "avoid_strategies": self.avoid_strategies,
            "historical_outcomes": self.historical_outcomes[-50:],  # Keep last 50
            "version": self.version,
            "created_at": self.created_at,
            "last_modified_at": self.last_modified_at,
            "modification_count": self.modification_count,
            "last_reflection_at": self.last_reflection_at,
            "reflection_insights": self.reflection_insights[-20:],  # Keep last 20
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiveModeIntent:
        """Deserialize from dictionary."""
        intent = cls(
            context_prompt_extension=str(data.get("context_prompt_extension", "")),
            goal_extension=str(data.get("goal_extension", "")),
            max_initiative=float(data.get("max_initiative", DEFAULT_INITIATIVE)),
            preferred_strategies=list(data.get("preferred_strategies", [])),
            avoid_strategies=list(data.get("avoid_strategies", [])),
            historical_outcomes=list(data.get("historical_outcomes", [])),
            version=int(data.get("version", 1)),
            created_at=float(data.get("created_at", time.time())),
            last_modified_at=float(data.get("last_modified_at", time.time())),
            modification_count=int(data.get("modification_count", 0)),
            last_reflection_at=data.get("last_reflection_at"),
            reflection_insights=list(data.get("reflection_insights", [])),
        )
        # Validate bounds
        intent.max_initiative = intent.validate_initiative(intent.max_initiative)
        return intent


class LiveModeIntentStore:
    """Persistent storage for agent-defined live mode intent."""

    def __init__(self, agent_data_dir: str):
        """Initialize store with agent data directory.

        Args:
            agent_data_dir: Path like data/agents/MaximAgent/
        """
        self._data_dir = agent_data_dir
        self._modes_dir = os.path.join(agent_data_dir, "modes")
        self._intent_path = os.path.join(self._modes_dir, "live_intent.json")
        self._intent: LiveModeIntent | None = None

    def _ensure_dir(self) -> None:
        """Ensure modes directory exists."""
        os.makedirs(self._modes_dir, exist_ok=True)

    def load(self) -> LiveModeIntent:
        """Load intent from disk, or create default."""
        if self._intent is not None:
            return self._intent

        if os.path.exists(self._intent_path):
            try:
                with open(self._intent_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._intent = LiveModeIntent.from_dict(data)
                logger.debug(f"Loaded live mode intent v{self._intent.version}")
            except Exception as e:
                logger.warning(f"Failed to load live intent, using default: {e}")
                self._intent = LiveModeIntent()
        else:
            self._intent = LiveModeIntent()

        return self._intent

    def save(self, intent: LiveModeIntent | None = None) -> None:
        """Save intent to disk."""
        self._ensure_dir()

        if intent is not None:
            self._intent = intent

        if self._intent is None:
            return

        self._intent.last_modified_at = time.time()

        try:
            with open(self._intent_path, "w", encoding="utf-8") as f:
                json.dump(self._intent.to_dict(), f, indent=2)
            logger.debug(f"Saved live mode intent v{self._intent.version}")
        except Exception as e:
            logger.error(f"Failed to save live intent: {e}")

    def update(
        self,
        context_prompt_extension: str | None = None,
        goal_extension: str | None = None,
        max_initiative: float | None = None,
        preferred_strategies: list[str] | None = None,
        avoid_strategies: list[str] | None = None,
    ) -> LiveModeIntent:
        """Update intent with new values and save."""
        intent = self.load()

        if context_prompt_extension is not None:
            intent.context_prompt_extension = context_prompt_extension
        if goal_extension is not None:
            intent.goal_extension = goal_extension
        if max_initiative is not None:
            intent.max_initiative = intent.validate_initiative(max_initiative)
        if preferred_strategies is not None:
            intent.preferred_strategies = preferred_strategies
        if avoid_strategies is not None:
            intent.avoid_strategies = avoid_strategies

        intent.modification_count += 1
        intent.version += 1

        self.save(intent)
        return intent

    def record_outcome(
        self,
        action: str,
        success: bool,
        context: str,
        reasoning: str = "",
    ) -> None:
        """Record an outcome for learning."""
        intent = self.load()
        intent.historical_outcomes.append(
            {
                "timestamp": time.time(),
                "action": action,
                "success": success,
                "context": context,
                "reasoning": reasoning,
            }
        )
        # Trim to last 50
        if len(intent.historical_outcomes) > 50:
            intent.historical_outcomes = intent.historical_outcomes[-50:]
        self.save(intent)

    def record_reflection_insight(self, insight: str) -> None:
        """Record an insight from reflection mode."""
        intent = self.load()
        intent.reflection_insights.append(insight)
        intent.last_reflection_at = time.time()
        # Trim to last 20
        if len(intent.reflection_insights) > 20:
            intent.reflection_insights = intent.reflection_insights[-20:]
        self.save(intent)


def get_live_mode_with_intent(intent: LiveModeIntent) -> "ModeDefinition":
    """Create a live mode definition with agent-defined intent.

    Similar to get_exploration_mode_with_policy(), this merges
    agent preferences with the base live mode.

    Args:
        intent: Agent-defined intent configuration

    Returns:
        Modified ModeDefinition for live mode
    """
    from maxim.modes.definitions import MODES, ModeDefinition

    base_mode = MODES["live"]

    # Build extended context prompt
    extended_context = base_mode.context_prompt
    if intent.context_prompt_extension:
        extended_context += (
            f"\n\n--- Agent-Defined Intent ---\n{intent.context_prompt_extension}"
        )

    # Build extended goal - primary goal is immutable
    extended_goal = f"{LIVE_MODE_PRIMARY_GOAL}. {base_mode.goal}"
    if intent.goal_extension:
        extended_goal = f"{extended_goal}. Additionally: {intent.goal_extension}"

    # Merge strategy preferences
    preferred = list(base_mode.preferred_strategies)
    for strategy in intent.preferred_strategies:
        if strategy not in preferred:
            preferred.append(strategy)

    avoided = list(base_mode.avoid_strategies)
    for strategy in intent.avoid_strategies:
        if strategy not in avoided:
            avoided.append(strategy)

    return ModeDefinition(
        name="live",
        goal=extended_goal,
        success_criteria=base_mode.success_criteria,
        forbidden_tools=set(),  # Keep empty for full access
        max_initiative=intent.max_initiative,
        preferred_strategies=preferred,
        avoid_strategies=avoided,
        context_prompt=extended_context,
        outcome_memory_key="live_intent_outcomes",
    )
