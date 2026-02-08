"""Unit tests for DecisionEngine - Action selection and plan evaluation.

Tests the core functionality of plan evaluation, constraint checking,
and action selection.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest


class TestDecisionEngineBasic:
    """Test basic decision engine functionality."""

    def test_decide_returns_none_when_no_plans(self, mock_planner, mock_policy):
        """Returns None when planner proposes no plans."""
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = []

        engine = DecisionEngine(mock_planner, mock_policy)
        result = engine.decide("find mug", state={}, memory={})

        assert result is None

    def test_decide_returns_action_from_plan(self, mock_planner, mock_policy):
        """Returns the first action from the best plan."""
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = [
            [{"tool_name": "look_around", "params": {}}]
        ]

        engine = DecisionEngine(mock_planner, mock_policy)
        result = engine.decide("find mug", state={}, memory={})

        assert result is not None
        assert result["action"]["tool_name"] == "look_around"

    def test_decide_includes_full_plan(self, mock_planner, mock_policy):
        """Result includes the full plan."""
        from maxim.planning.decision_engine import DecisionEngine

        plan = [
            {"tool_name": "look_around"},
            {"tool_name": "move_to"},
            {"tool_name": "grasp"},
        ]
        mock_planner.propose_plans.return_value = [plan]

        engine = DecisionEngine(mock_planner, mock_policy)
        result = engine.decide("find mug", state={}, memory={})

        assert result["plan"] == plan
        assert len(result["plan"]) == 3


class TestDecisionEnginePlanSelection:
    """Test plan scoring and selection."""

    def test_selects_highest_scored_plan(self, mock_planner, mock_policy):
        """Selects the plan with highest policy score."""
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = [
            [{"tool_name": "slow_method"}],
            [{"tool_name": "fast_method"}],
            [{"tool_name": "medium_method"}],
        ]

        # Score plans differently
        mock_policy.score.side_effect = [0.3, 0.9, 0.5]

        engine = DecisionEngine(mock_planner, mock_policy)
        result = engine.decide("goal", state={}, memory={})

        assert result["action"]["tool_name"] == "fast_method"
        assert result["score"] == 0.9

    def test_returns_score_in_result(self, mock_planner, mock_policy):
        """Result includes the score."""
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = [[{"tool_name": "test"}]]
        mock_policy.score.return_value = 0.75

        engine = DecisionEngine(mock_planner, mock_policy)
        result = engine.decide("goal", state={}, memory={})

        assert result["score"] == 0.75


class TestDecisionEngineConstraints:
    """Test constraint checking."""

    def test_rejects_plans_violating_constraints(self, mock_planner, mock_policy):
        """Plans violating constraints are rejected."""
        from maxim.planning.constraints import ConstraintViolation
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = [
            [{"tool_name": "dangerous_action"}]
        ]

        constraint = Mock()
        constraint.check.side_effect = ConstraintViolation("Not safe")

        engine = DecisionEngine(mock_planner, mock_policy, constraints=[constraint])
        result = engine.decide("goal", state={}, memory={})

        assert result is None

    def test_passes_plans_that_satisfy_constraints(self, mock_planner, mock_policy):
        """Plans satisfying constraints are accepted."""
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = [[{"tool_name": "safe_action"}]]

        constraint = Mock()
        constraint.check.return_value = None  # No exception = satisfied

        engine = DecisionEngine(mock_planner, mock_policy, constraints=[constraint])
        result = engine.decide("goal", state={}, memory={})

        assert result is not None

    def test_skips_violating_plan_tries_next(self, mock_planner, mock_policy):
        """Skips violating plan and tries next."""
        from maxim.planning.constraints import ConstraintViolation
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = [
            [{"tool_name": "bad_plan"}],
            [{"tool_name": "good_plan"}],
        ]

        constraint = Mock()
        # First plan fails, second passes
        constraint.check.side_effect = [ConstraintViolation("Bad"), None]

        engine = DecisionEngine(mock_planner, mock_policy, constraints=[constraint])
        result = engine.decide("goal", state={}, memory={})

        assert result is not None
        assert result["action"]["tool_name"] == "good_plan"


class TestDecisionEnginePolicy:
    """Test policy enforcement."""

    def test_rejects_plans_blocked_by_policy(self, mock_planner, mock_policy):
        """Plans blocked by policy.allow() are rejected."""
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = [[{"tool_name": "blocked"}]]
        mock_policy.allow.return_value = False

        engine = DecisionEngine(mock_planner, mock_policy)
        result = engine.decide("goal", state={}, memory={})

        assert result is None

    def test_accepts_plans_allowed_by_policy(self, mock_planner, mock_policy):
        """Plans allowed by policy are accepted."""
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = [[{"tool_name": "allowed"}]]
        mock_policy.allow.return_value = True

        engine = DecisionEngine(mock_planner, mock_policy)
        result = engine.decide("goal", state={}, memory={})

        assert result is not None

    def test_checks_all_actions_in_plan(self, mock_planner, mock_policy):
        """Policy checks all actions in a plan."""
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = [
            [
                {"tool_name": "action_1"},
                {"tool_name": "action_2"},
                {"tool_name": "action_3"},
            ]
        ]

        # Second action is blocked
        mock_policy.allow.side_effect = [True, False, True]

        engine = DecisionEngine(mock_planner, mock_policy)
        result = engine.decide("goal", state={}, memory={})

        assert result is None  # Plan rejected due to blocked action


class TestDecisionEngineEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_empty_plan(self, mock_planner, mock_policy):
        """Handles empty plan gracefully."""
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = [[]]  # Empty plan

        engine = DecisionEngine(mock_planner, mock_policy)
        result = engine.decide("goal", state={}, memory={})

        assert result is None

    def test_handles_invalid_plan_format(self, mock_planner, mock_policy):
        """Handles invalid plan format gracefully."""
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = ["not a list"]

        engine = DecisionEngine(mock_planner, mock_policy)
        result = engine.decide("goal", state={}, memory={})

        assert result is None

    def test_handles_none_plans(self, mock_planner, mock_policy):
        """Handles None from planner gracefully."""
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = None

        engine = DecisionEngine(mock_planner, mock_policy)
        result = engine.decide("goal", state={}, memory={})

        assert result is None

    def test_handles_planner_with_no_plans_method(self, mock_policy):
        """Handles planner that returns empty list."""
        from maxim.planning.decision_engine import DecisionEngine

        planner = Mock()
        planner.propose_plans.return_value = []

        engine = DecisionEngine(planner, mock_policy)
        result = engine.decide("goal", state={}, memory={})

        assert result is None


class TestDecisionEngineMultipleConstraints:
    """Test behavior with multiple constraints."""

    def test_all_constraints_must_pass(self, mock_planner, mock_policy):
        """All constraints must be satisfied."""
        from maxim.planning.constraints import ConstraintViolation
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = [[{"tool_name": "test"}]]

        constraint1 = Mock()
        constraint1.check.return_value = None

        constraint2 = Mock()
        constraint2.check.side_effect = ConstraintViolation("Failed")

        engine = DecisionEngine(
            mock_planner, mock_policy, constraints=[constraint1, constraint2]
        )
        result = engine.decide("goal", state={}, memory={})

        assert result is None

    def test_constraints_checked_in_order(self, mock_planner, mock_policy):
        """Constraints are checked in order."""
        from maxim.planning.decision_engine import DecisionEngine

        mock_planner.propose_plans.return_value = [[{"tool_name": "test"}]]

        check_order = []

        def make_check(name):
            def check(plan, state):
                check_order.append(name)

            return check

        constraint1 = Mock()
        constraint1.check = make_check("first")

        constraint2 = Mock()
        constraint2.check = make_check("second")

        engine = DecisionEngine(
            mock_planner, mock_policy, constraints=[constraint1, constraint2]
        )
        engine.decide("goal", state={}, memory={})

        assert check_order == ["first", "second"]