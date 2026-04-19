"""B4 Replanning — Stage 1 tests.

Tests for:
1. structural_diff.py — Jaccard distance on action sequences
2. ReplanContext prior_attempt enrichment
3. _build_replan_context type fix (stub Phase, not raw string)
4. Anti-repetition prompt constraint
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from maxim.planning.structural_diff import (
    action_signature,
    all_pairwise_distances,
    jaccard_distance,
    min_distance_from_prior,
    plan_distance,
)


# ─────────────────────────────────────────────────────────────────────────────
# structural_diff.py
# ─────────────────────────────────────────────────────────────────────────────


class TestActionSignature:
    def test_basic(self):
        assert action_signature({"tool_name": "grab", "params": {"x": 1}}) == "grab(x)"

    def test_sorted_params(self):
        sig = action_signature({"tool_name": "move", "params": {"z": 0, "a": 1, "m": 2}})
        assert sig == "move(a,m,z)"

    def test_no_params(self):
        assert action_signature({"tool_name": "scan"}) == "scan()"

    def test_empty_tool(self):
        assert action_signature({}) == "()"


class TestJaccardDistance:
    def test_identical_sets(self):
        s = {"a", "b", "c"}
        assert jaccard_distance(s, s) == 0.0

    def test_disjoint_sets(self):
        assert jaccard_distance({"a", "b"}, {"c", "d"}) == 1.0

    def test_partial_overlap(self):
        # {a,b,c} ∩ {b,c,d} = {b,c}, union = {a,b,c,d}
        d = jaccard_distance({"a", "b", "c"}, {"b", "c", "d"})
        assert d == pytest.approx(0.5)

    def test_empty_sets(self):
        assert jaccard_distance(set(), set()) == 0.0

    def test_one_empty(self):
        assert jaccard_distance({"a"}, set()) == 1.0


class TestPlanDistance:
    def test_identical_plans(self):
        plan = [{"tool_name": "grab", "params": {"x": 1}}, {"tool_name": "scan"}]
        assert plan_distance(plan, plan) == 0.0

    def test_completely_different_plans(self):
        plan_a = [{"tool_name": "grab"}, {"tool_name": "scan"}]
        plan_b = [{"tool_name": "move"}, {"tool_name": "wait"}]
        assert plan_distance(plan_a, plan_b) == 1.0

    def test_same_tools_different_params(self):
        # Same tool names but different param keys = different signatures
        plan_a = [{"tool_name": "grab", "params": {"x": 1}}]
        plan_b = [{"tool_name": "grab", "params": {"y": 2}}]
        assert plan_distance(plan_a, plan_b) == 1.0

    def test_same_tools_same_param_keys(self):
        # Same tool + same param key names = same signature (values don't matter)
        plan_a = [{"tool_name": "grab", "params": {"x": 1}}]
        plan_b = [{"tool_name": "grab", "params": {"x": 99}}]
        assert plan_distance(plan_a, plan_b) == 0.0


class TestMinDistanceFromPrior:
    def test_no_prior(self):
        assert min_distance_from_prior([{"tool_name": "grab"}], []) == 1.0

    def test_identical_prior(self):
        plan = [{"tool_name": "grab"}]
        assert min_distance_from_prior(plan, [plan]) == 0.0

    def test_multiple_priors_picks_minimum(self):
        candidate = [{"tool_name": "grab"}, {"tool_name": "scan"}]
        prior_1 = [{"tool_name": "grab"}, {"tool_name": "scan"}]  # identical
        prior_2 = [{"tool_name": "move"}, {"tool_name": "wait"}]  # disjoint
        assert min_distance_from_prior(candidate, [prior_1, prior_2]) == 0.0


class TestAllPairwiseDistances:
    def test_single_plan(self):
        assert all_pairwise_distances([[{"tool_name": "a"}]]) == []

    def test_two_identical(self):
        plan = [{"tool_name": "a"}]
        assert all_pairwise_distances([plan, plan]) == [0.0]

    def test_three_plans(self):
        # 3 plans = 3 pairwise distances
        plans = [
            [{"tool_name": "a"}],
            [{"tool_name": "b"}],
            [{"tool_name": "c"}],
        ]
        distances = all_pairwise_distances(plans)
        assert len(distances) == 3
        assert all(d == 1.0 for d in distances)  # all disjoint


# ─────────────────────────────────────────────────────────────────────────────
# ReplanContext prior_attempt enrichment
# ─────────────────────────────────────────────────────────────────────────────


class TestReplanContextPriorAttempts:
    def test_default_empty(self):
        from maxim.planning.plan_document import Phase, PhaseStatus, ReplanContext

        ctx = ReplanContext(
            failed_phase=Phase(id="p1", description="test", status=PhaseStatus.FAILED, plan_id=""),
            failure_reason="timeout",
            failure_type="timeout",
        )
        assert ctx.prior_attempt_actions == []
        assert ctx.prior_attempt_summaries == []

    def test_serialization_round_trip(self):
        from maxim.planning.plan_document import Phase, PhaseStatus, ReplanContext

        ctx = ReplanContext(
            failed_phase=Phase(id="p1", description="test", status=PhaseStatus.FAILED, plan_id=""),
            failure_reason="timeout",
            failure_type="timeout",
            prior_attempt_actions=[[{"tool_name": "grab"}], [{"tool_name": "scan"}]],
            prior_attempt_summaries=["attempt 1", "attempt 2"],
        )
        d = ctx.to_dict()
        restored = ReplanContext.from_dict(d)
        assert len(restored.prior_attempt_actions) == 2
        assert restored.prior_attempt_actions[0] == [{"tool_name": "grab"}]
        assert restored.prior_attempt_summaries == ["attempt 1", "attempt 2"]

    def test_prompt_section_includes_prior_attempts(self):
        from maxim.planning.plan_document import Phase, PhaseStatus, ReplanContext

        ctx = ReplanContext(
            failed_phase=Phase(id="p1", description="navigate maze", status=PhaseStatus.FAILED, plan_id=""),
            failure_reason="dead end",
            failure_type="tool_error",
            prior_attempt_actions=[
                [{"tool_name": "move_north"}, {"tool_name": "move_east"}],
            ],
            prior_attempt_summaries=["went north then east, hit wall"],
        )
        prompt = ctx.to_llm_prompt_section()
        assert "Prior Plan Attempts" in prompt
        assert "move_north" in prompt
        assert "structurally different" in prompt.lower()

    def test_prompt_section_without_prior_attempts_unchanged(self):
        from maxim.planning.plan_document import Phase, PhaseStatus, ReplanContext

        ctx = ReplanContext(
            failed_phase=Phase(id="p1", description="test", status=PhaseStatus.FAILED, plan_id=""),
            failure_reason="error",
            failure_type="tool_error",
        )
        prompt = ctx.to_llm_prompt_section()
        assert "Prior Plan Attempts" not in prompt


# ─────────────────────────────────────────────────────────────────────────────
# _build_replan_context type fix
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildReplanContextTypeFix:
    """Verify _build_replan_context returns a valid ReplanContext
    with a Phase object (not a raw string) as failed_phase.
    """

    def test_failed_phase_is_phase_object(self):
        from maxim.planning.plan_document import Phase
        from maxim.runtime.loop_state import _build_replan_context

        @dataclass
        class FakeResult:
            success: bool = False
            error: str = "timeout"
            error_kind: str = "timeout"
            output: str = ""

        intent = {"goal": "navigate to kitchen"}
        action = {"tool_name": "move"}
        result = FakeResult()
        state = MagicMock()
        state.data = {}

        ctx = _build_replan_context(intent, action, result, state)
        assert isinstance(ctx.failed_phase, Phase)
        assert ctx.failed_phase.description == "navigate to kitchen"

    def test_to_llm_prompt_section_does_not_crash(self):
        """Previously crashed with AttributeError: 'str' has no attribute 'description'."""
        from maxim.runtime.loop_state import _build_replan_context

        @dataclass
        class FakeResult:
            success: bool = False
            error: str = "tool broke"
            error_kind: str = "tool_error"
            output: str = ""

        ctx = _build_replan_context(
            {"goal": "test goal"},
            {"tool_name": "grab"},
            FakeResult(),
            MagicMock(data={}),
        )
        # This line used to crash before the fix
        prompt = ctx.to_llm_prompt_section()
        assert "test goal" in prompt
        assert "REPLAN REQUIRED" in prompt

    def test_hippocampus_retrieval_populates_prior_attempts(self):
        from maxim.runtime.loop_state import _build_replan_context

        @dataclass
        class FakeResult:
            success: bool = False
            error: str = "failed"
            error_kind: str = "tool_error"
            output: str = ""

        @dataclass
        class FakeEpisode:
            content: dict | str
            metadata: dict | None = None

        hippo = MagicMock()
        hippo.recall.return_value = [
            FakeEpisode(
                content={
                    "decision": {"action": {"tool_name": "grab", "params": {"obj": "cup"}}},
                },
            ),
            FakeEpisode(content="text episode"),  # should be skipped
        ]

        ctx = _build_replan_context(
            {"goal": "grab cup"},
            {"tool_name": "grab"},
            FakeResult(),
            MagicMock(data={}),
            hippocampus=hippo,
        )
        hippo.recall.assert_called_once_with(goal="grab cup", limit=10)
        assert len(ctx.prior_attempt_actions) == 1
        assert ctx.prior_attempt_actions[0] == [{"tool_name": "grab", "params": {"obj": "cup"}}]

    def test_hippocampus_failure_graceful(self):
        from maxim.runtime.loop_state import _build_replan_context

        @dataclass
        class FakeResult:
            success: bool = False
            error: str = "x"
            error_kind: str = "x"
            output: str = ""

        hippo = MagicMock()
        hippo.recall.side_effect = RuntimeError("DB down")

        ctx = _build_replan_context(
            {"goal": "test"},
            {"tool_name": "t"},
            FakeResult(),
            MagicMock(data={}),
            hippocampus=hippo,
        )
        assert ctx.prior_attempt_actions == []


# ─────────────────────────────────────────────────────────────────────────────
# PlanManager._build_replan_context prior attempt enrichment
# ─────────────────────────────────────────────────────────────────────────────


class TestPlanManagerPriorAttempts:
    def test_replan_history_populates_prior_attempts(self):
        from maxim.agents.bus import AgentBus, FailureStrategy, SubGoal, SubGoalStatus
        from maxim.planning.plan_document import (
            Phase,
            PhaseStatus,
            PlanDocument,
            PlanStatus,
            ReplanContext,
            ReplanRecord,
        )
        from maxim.planning.plan_manager import PlanManager

        bus = MagicMock(spec=AgentBus)
        bus.publish = MagicMock()
        pm = PlanManager(plans_dir="/tmp/test_plans", bus=bus)

        phase = Phase(
            id="phase_1",
            description="get water",
            status=PhaseStatus.ACTIVE,
            plan_id="plan_1",
            sub_goals=[
                SubGoal(
                    id="sg1",
                    description="grab cup",
                    tool_name="grab",
                    tool_params={"obj": "cup"},
                    status=SubGoalStatus.FAILED,
                    on_failure=FailureStrategy.REPLAN,
                ),
            ],
        )

        plan = PlanDocument(
            id="plan_1",
            objective="get water",
            created_at=0.0,
            updated_at=0.0,
            status=PlanStatus.ACTIVE,
            phases=[phase],
        )

        # Add a prior replan record
        prior_ctx = ReplanContext(
            failed_phase=phase,
            failure_reason="cup too heavy",
            failure_type="tool_error",
        )
        plan.replan_history.append(
            ReplanRecord(
                timestamp=0.0,
                failed_phase_id="phase_1",
                failure_reason="cup too heavy",
                replan_context=prior_ctx,
                original_sub_goals=[
                    {"tool_name": "grab", "tool_params": {"obj": "cup"}, "description": "grab cup"},
                ],
            )
        )

        pm._active_plan = plan
        ctx = pm._build_replan_context(plan, phase, "cup broke", attempt_count=1)

        assert len(ctx.prior_attempt_actions) == 1
        assert ctx.prior_attempt_actions[0][0]["tool_name"] == "grab"
        assert "cup too heavy" in ctx.prior_attempt_summaries[0]


# ─────────────────────────────────────────────────────────────────────────────
# B4 pass gate: structural difference > 0.3
# ─────────────────────────────────────────────────────────────────────────────


class TestB4PassGate:
    """The B4 pass gate requires Jaccard distance > 0.3 between successive plans."""

    def test_structurally_different_plans_pass(self):
        plan_1 = [
            {"tool_name": "grab", "params": {"obj": "cup"}},
            {"tool_name": "move", "params": {"dir": "north"}},
            {"tool_name": "place", "params": {"loc": "table"}},
        ]
        plan_2 = [
            {"tool_name": "find_container"},
            {"tool_name": "fill_container", "params": {"source": "tap"}},
            {"tool_name": "move", "params": {"dir": "north"}},
        ]
        distance = plan_distance(plan_1, plan_2)
        assert distance > 0.3, f"Plans not different enough: Jaccard={distance}"

    def test_rearranged_plan_fails_gate(self):
        """Just rearranging steps should NOT pass the novelty gate."""
        plan_1 = [
            {"tool_name": "grab"},
            {"tool_name": "move"},
            {"tool_name": "place"},
        ]
        plan_2 = [
            {"tool_name": "move"},
            {"tool_name": "grab"},
            {"tool_name": "place"},
        ]
        distance = plan_distance(plan_1, plan_2)
        assert distance == 0.0, "Rearranged steps should have zero distance (same signature set)"

    def test_three_plans_all_differ(self):
        """Plan 3 must differ from both plan 1 and plan 2."""
        plan_1 = [{"tool_name": "grab"}, {"tool_name": "move"}]
        plan_2 = [{"tool_name": "scan"}, {"tool_name": "reach"}]
        plan_3 = [{"tool_name": "ask_help"}, {"tool_name": "wait"}]

        d_12 = plan_distance(plan_1, plan_2)
        d_13 = plan_distance(plan_1, plan_3)
        d_23 = plan_distance(plan_2, plan_3)

        assert d_12 > 0.3
        assert d_13 > 0.3
        assert d_23 > 0.3
        assert min_distance_from_prior(plan_3, [plan_1, plan_2]) > 0.3
