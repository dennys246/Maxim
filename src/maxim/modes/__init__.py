"""Modes package for goal-driven mode execution.

Modes define goals and constraints rather than fixed procedures.
The agent selects strategies to achieve mode goals.
"""

from maxim.modes.definitions import (
    ModeDefinition,
    MODES,
    get_mode,
    get_exploration_mode_with_policy,
)
from maxim.modes.strategies import Strategy, StrategyLibrary, get_strategy_library
from maxim.modes.exploration import (
    ExplorationPolicy,
    ExplorationConstraints,
    ExplorationBudget,
    ExplorationSession,
    ExplorationSubGoal,
    CuriosityManager,
    AdversarialFocusValidator,
    FocusDecomposer,
    parse_explore_command,
)

__all__ = [
    # Definitions
    "ModeDefinition",
    "MODES",
    "get_mode",
    "get_exploration_mode_with_policy",
    # Strategies
    "Strategy",
    "StrategyLibrary",
    "get_strategy_library",
    # Exploration
    "ExplorationPolicy",
    "ExplorationConstraints",
    "ExplorationBudget",
    "ExplorationSession",
    "ExplorationSubGoal",
    "CuriosityManager",
    "AdversarialFocusValidator",
    "FocusDecomposer",
    "parse_explore_command",
]
