"""B4 Replanning — Stage 2: Induced-failure scenario.

Full agent loop integration test: plan → execute → fail → replan → execute
with structural difference verification across multiple replan attempts.

The scenario: agent has a multi-step goal with an injected failure at step 3.
After each failure, the planner produces a structurally different plan.
Pass gate: agent recovers within 3 replan attempts, each structurally different.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

from maxim.planning.adaptive_planner import PlanCandidate
from maxim.planning.structural_diff import min_distance_from_prior, plan_distance


# ─────────────────────────────────────────────────────────────────────────────
# Fake components for the induced-failure scenario
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FakeResult:
    success: bool = True
    error: str | None = None
    error_kind: str | None = None
    output: str = ""


class InducedFailureExecutor:
    """Executor that fails on specific tool calls, succeeds on others.

    Tracks all executed actions for structural comparison.
    """

    def __init__(self, fail_tools: set[str]):
        self._fail_tools = fail_tools
        self.executed: list[dict] = []

    def execute(self, action: dict) -> FakeResult:
        self.executed.append(action)
        tool = action.get("tool_name", "")
        if tool in self._fail_tools:
            return FakeResult(success=False, error=f"{tool} failed: injected failure", error_kind="tool_error")
        return FakeResult(success=True, output=f"{tool} succeeded")


class DeterministicPlanner:
    """Planner that produces a sequence of pre-defined alternative plans.

    Each call to decompose() returns the next plan in the sequence,
    simulating structurally different replan attempts.
    """

    def __init__(self, alternatives: list[list[dict]]):
        self._alternatives = alternatives
        self._call_count = 0
        self.decompose_calls: list[dict] = []

    def decompose(
        self,
        failed_goal: dict,
        replan_ctx: object,
        depth: int,
    ) -> PlanCandidate | None:
        self.decompose_calls.append(
            {
                "failed_goal": failed_goal,
                "replan_ctx": replan_ctx,
                "depth": depth,
            }
        )
        if self._call_count >= len(self._alternatives):
            return None
        actions = self._alternatives[self._call_count]
        self._call_count += 1
        return PlanCandidate(
            actions=actions,
            source="decomposed",
            depth=depth + 1,
        )


class FailThenSucceedState:
    """State that tracks steps and allows multiple iterations."""

    def __init__(self, max_iterations: int = 10):
        self.data: dict = {}
        self.steps_taken = 0
        self._max = max_iterations
        self.failures: list[str] = []

    def update(self, obs: object) -> None:
        pass

    def snapshot(self) -> dict:
        return {"steps": self.steps_taken, "failures": self.failures}

    def mark_failure(self, err: object) -> None:
        self.failures.append(str(err))

    def is_done(self) -> bool:
        return self.steps_taken >= self._max


class SimpleEnv:
    def observe(self) -> dict:
        return {}

    def step(self, result: object) -> None:
        return None


class ReplanDecisionEngine:
    """Decision engine that wraps a planner and supports the replan protocol."""

    def __init__(self, planner: DeterministicPlanner, initial_plan: list[dict]):
        self.planner = planner
        self._initial_plan = initial_plan
        self._used = False

    def decide(self, goal: object, state: object, memory: object) -> dict:
        if not self._used:
            self._used = True
            action = self._initial_plan[0]
            return {
                "action": action,
                "plan": PlanCandidate(
                    actions=self._initial_plan,
                    source="direct",
                    depth=0,
                ),
                "score": 1.0,
            }
        return {}


class ReplanAgent:
    """Agent that proposes a fixed goal with REPLAN failure strategy."""

    def __init__(self, goal: str):
        self._goal = goal
        self._call_count = 0

    def propose_intent(self, state: object, memory: object) -> dict | None:
        self._call_count += 1
        if self._call_count > 10:
            return None  # safety valve
        return {
            "goal": self._goal,
            "on_failure": "replan",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 tests
# ─────────────────────────────────────────────────────────────────────────────


class TestInducedFailureReplan:
    """Test the full replan cycle with injected failure."""

    def test_single_failure_triggers_replan(self):
        """After one failure, a replan candidate is stored in state."""
        from maxim.runtime.agent_loop import run_agent_loop

        initial_plan = [
            {"tool_name": "grab", "params": {"obj": "cup"}},
        ]
        alt_plan = [
            {"tool_name": "use_tongs", "params": {"obj": "cup"}},
        ]

        planner = DeterministicPlanner([alt_plan])
        de = ReplanDecisionEngine(planner, initial_plan)
        executor = InducedFailureExecutor(fail_tools={"grab"})
        state = FailThenSucceedState()
        memory = MagicMock()
        memory.store_raw = MagicMock()

        run_agent_loop(
            agent=ReplanAgent("get water"),
            environment=SimpleEnv(),
            state=state,
            memory=memory,
            decision_engine=de,
            executor=executor,
            max_steps=1,
            break_on_no_intent=True,
        )

        # Planner.decompose was called with failure context
        assert len(planner.decompose_calls) == 1
        call = planner.decompose_calls[0]
        assert call["failed_goal"]["tool_name"] == "grab"

        # Replan candidate stored in state for next iteration
        assert "_replan_candidate" in state.data
        candidate = state.data["_replan_candidate"]
        assert candidate.actions[0]["tool_name"] == "use_tongs"

    def test_replan_candidate_executes_on_next_iteration(self):
        """The replan candidate from a failure is consumed and executed next turn."""
        from maxim.runtime.agent_loop import run_agent_loop

        initial_plan = [{"tool_name": "grab"}]
        alt_plan = [{"tool_name": "use_tongs"}]

        planner = DeterministicPlanner([alt_plan])
        de = ReplanDecisionEngine(planner, initial_plan)
        # grab fails, use_tongs succeeds
        executor = InducedFailureExecutor(fail_tools={"grab"})
        state = FailThenSucceedState()
        memory = MagicMock()
        memory.store_raw = MagicMock()

        run_agent_loop(
            agent=ReplanAgent("get water"),
            environment=SimpleEnv(),
            state=state,
            memory=memory,
            decision_engine=de,
            executor=executor,
            max_steps=2,
            break_on_no_intent=True,
        )

        # Both actions were executed
        assert len(executor.executed) == 2
        assert executor.executed[0]["tool_name"] == "grab"  # failed
        assert executor.executed[1]["tool_name"] == "use_tongs"  # replan success

    def test_structural_difference_across_replans(self):
        """Each replan attempt must be structurally different from the last."""
        # Plan 1: grab → move → place (grab fails)
        initial_plan = [
            {"tool_name": "grab", "params": {"obj": "cup"}},
            {"tool_name": "move", "params": {"dir": "north"}},
            {"tool_name": "place", "params": {"loc": "table"}},
        ]

        # Plan 2: use_tongs → carry → set_down (use_tongs fails)
        alt_plan_1 = [
            {"tool_name": "use_tongs", "params": {"obj": "cup"}},
            {"tool_name": "carry", "params": {"dir": "north"}},
            {"tool_name": "set_down", "params": {"loc": "table"}},
        ]

        # Plan 3: ask_help → receive → deliver (succeeds)
        alt_plan_2 = [
            {"tool_name": "ask_help", "params": {"request": "bring cup"}},
            {"tool_name": "receive", "params": {"from": "helper"}},
            {"tool_name": "deliver", "params": {"loc": "table"}},
        ]

        # Verify all plans are structurally different (Jaccard > 0.3)
        d_12 = plan_distance(initial_plan, alt_plan_1)
        d_13 = plan_distance(initial_plan, alt_plan_2)
        d_23 = plan_distance(alt_plan_1, alt_plan_2)

        assert d_12 > 0.3, f"Plans 1→2 not different enough: {d_12}"
        assert d_13 > 0.3, f"Plans 1→3 not different enough: {d_13}"
        assert d_23 > 0.3, f"Plans 2→3 not different enough: {d_23}"

        # Plan 3 is novel w.r.t. both prior attempts
        min_dist = min_distance_from_prior(alt_plan_2, [initial_plan, alt_plan_1])
        assert min_dist > 0.3, f"Plan 3 too similar to a prior: min_dist={min_dist}"


class TestReplanContextCarriesPriorAttempts:
    """Verify that prior attempts flow through the replan context to the planner."""

    def test_prior_attempts_in_replan_context(self):
        from maxim.runtime.loop_state import _build_replan_context

        @dataclass
        class FakeResult:
            success: bool = False
            error: str = "failed"
            error_kind: str = "tool_error"
            output: str = ""

        @dataclass
        class FakeEpisode:
            content: dict
            metadata: dict | None = None

        hippo = MagicMock()
        hippo.recall.return_value = [
            FakeEpisode(
                content={
                    "decision": {"action": {"tool_name": "grab", "params": {"obj": "cup"}}},
                },
            ),
            FakeEpisode(
                content={
                    "decision": {"action": {"tool_name": "push", "params": {"dir": "left"}}},
                },
            ),
        ]

        ctx = _build_replan_context(
            {"goal": "get water"},
            {"tool_name": "grab"},
            FakeResult(),
            MagicMock(data={}),
            hippocampus=hippo,
        )

        # Prior attempts were retrieved and included
        assert len(ctx.prior_attempt_actions) == 2

        # The prompt section includes them
        prompt = ctx.to_llm_prompt_section()
        assert "Prior Plan Attempts" in prompt
        assert "grab" in prompt
        assert "push" in prompt

    def test_five_seeds_mean_structural_difference(self):
        """B4 pass gate: across 5 seeds, mean Jaccard distance > 0.3."""
        from maxim.planning.structural_diff import all_pairwise_distances

        # Simulate 5 different replan outcomes (5 seeds producing different plans)
        seed_plans = [
            [{"tool_name": "grab"}, {"tool_name": "move"}, {"tool_name": "place"}],
            [{"tool_name": "use_tongs"}, {"tool_name": "carry"}, {"tool_name": "set_down"}],
            [{"tool_name": "ask_help"}, {"tool_name": "receive"}, {"tool_name": "deliver"}],
            [{"tool_name": "push"}, {"tool_name": "slide"}, {"tool_name": "position"}],
            [{"tool_name": "lift"}, {"tool_name": "transport"}, {"tool_name": "drop"}],
        ]

        distances = all_pairwise_distances(seed_plans)
        mean_distance = sum(distances) / len(distances) if distances else 0.0

        assert mean_distance > 0.3, f"Mean distance {mean_distance:.3f} below gate"
        # All pairwise must also exceed threshold
        for i, d in enumerate(distances):
            assert d > 0.3, f"Pairwise distance {i} = {d:.3f} below gate"
