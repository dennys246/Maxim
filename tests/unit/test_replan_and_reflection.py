"""Tests for Phase 3 (replan loop) and Phase 4 (reflection generation/storage).

Phase 3: ADaPT replan in run_agent_loop — failure with REPLAN strategy
         triggers planner.decompose(), candidate consumed on next iteration.
Phase 4: Reflexion in ToolPainBridge — surprising failures (RPE > 0.3)
         generate verbal reflections stored as episodic memories.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Replan helpers
# ─────────────────────────────────────────────────────────────────────────────

from maxim.runtime.agent_loop import (
    _get_failure_strategy,
    _get_plan_depth,
    _build_replan_context,
)


class TestGetFailureStrategy:
    def test_extracts_from_intent(self):
        intent = {"goal": "test", "on_failure": "REPLAN"}
        assert _get_failure_strategy(intent, {}) == "replan"

    def test_extracts_from_sub_goals(self):
        intent = {
            "goal": "test",
            "sub_goals": [
                {"tool_name": "grab", "on_failure": "REPLAN"},
                {"tool_name": "scan", "on_failure": "RETRY"},
            ],
        }
        assert _get_failure_strategy(intent, {"tool_name": "grab"}) == "replan"
        assert _get_failure_strategy(intent, {"tool_name": "scan"}) == "retry"

    def test_returns_empty_when_missing(self):
        assert _get_failure_strategy({"goal": "x"}, {}) == ""

    def test_handles_non_dict_intent(self):
        assert _get_failure_strategy("not a dict", {}) == ""


class TestGetPlanDepth:
    def test_extracts_from_plan_candidate(self):
        @dataclass
        class FakePlan:
            depth: int = 2
        decision = {"plan": FakePlan()}
        assert _get_plan_depth(decision) == 2

    def test_defaults_to_zero(self):
        assert _get_plan_depth({"plan": [{"tool_name": "x"}]}) == 0
        assert _get_plan_depth({}) == 0


class TestBuildReplanContext:
    def test_builds_from_failure_info(self):
        @dataclass
        class FakeResult:
            error: str = "timeout"
            error_kind: str = "transient"

        ctx = _build_replan_context(
            intent={"goal": "pick up cup"},
            action={"tool_name": "grab"},
            result=FakeResult(),
            state=None,
        )
        assert ctx.failed_phase == "pick up cup"
        assert ctx.failure_reason == "timeout"
        assert "grab" in ctx.attempted_tools


class TestReplanLoopIntegration:
    """Integration test: run_agent_loop with replan candidate consumption."""

    def test_replan_candidate_consumed_on_next_iteration(self):
        """When state has _replan_candidate, it should be executed directly,
        bypassing propose_intent/decide."""
        from maxim.planning.adaptive_planner import PlanCandidate

        replan = PlanCandidate(
            actions=[{"tool_name": "fallback_tool", "params": {"retry": True}}],
            source="decomposed",
            depth=1,
        )

        # State with replan candidate pre-loaded
        class FakeState:
            def __init__(self):
                self.data = {"_replan_candidate": replan, "_replan_goal": "original_goal"}
                self.steps_taken = 0
            def update(self, obs): pass
            def snapshot(self): return {}
            def mark_failure(self, err): pass
            def is_done(self): return True  # stop after one iteration

        class FakeEnv:
            def observe(self): return {}
            def step(self, result): return None

        class FakeResult:
            success = True

        class FakeExecutor:
            def execute(self, action):
                self.last_action = action
                return FakeResult()

        executor = FakeExecutor()
        state = FakeState()
        agent = MagicMock()
        memory = MagicMock()
        de = MagicMock()

        from maxim.runtime.agent_loop import run_agent_loop

        run_agent_loop(
            agent=agent,
            environment=FakeEnv(),
            state=state,
            memory=memory,
            decision_engine=de,
            executor=executor,
            max_steps=1,
            break_on_no_intent=True,
        )

        # The replan candidate should have been executed
        assert executor.last_action["tool_name"] == "fallback_tool"
        # propose_intent should NOT have been called (bypassed by replan)
        agent.propose_intent.assert_not_called()
        # decide should NOT have been called
        de.decide.assert_not_called()
        # Candidate should be consumed from state
        assert "_replan_candidate" not in state.data

    def test_normal_path_without_replan(self):
        """Without _replan_candidate, normal propose_intent → decide path."""

        class FakeState:
            def __init__(self):
                self.data = {}
                self.steps_taken = 0
            def update(self, obs): pass
            def snapshot(self): return {}
            def mark_failure(self, err): pass
            def is_done(self): return True

        class FakeEnv:
            def observe(self): return {}
            def step(self, result): return None

        class FakeResult:
            success = True

        class FakeExecutor:
            def execute(self, action): return FakeResult()

        agent = MagicMock()
        agent.propose_intent.return_value = {"goal": "test"}

        de = MagicMock()
        de.decide.return_value = {"action": {"tool_name": "t", "params": {}}, "plan": None, "score": 1.0}

        memory = MagicMock()

        from maxim.runtime.agent_loop import run_agent_loop

        run_agent_loop(
            agent=agent,
            environment=FakeEnv(),
            state=FakeState(),
            memory=memory,
            decision_engine=de,
            executor=FakeExecutor(),
            max_steps=1,
            break_on_no_intent=True,
        )

        agent.propose_intent.assert_called_once()
        de.decide.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Reflection generation
# ─────────────────────────────────────────────────────────────────────────────


def _make_link(
    link_id: str = "test-link",
    predicted_value: float = 0.7,
    last_rpe: float = 0.5,
    memory_ids: list[str] | None = None,
):
    from maxim.decisions.causal_link import CausalLink, TemporalDelta, Valence

    link = CausalLink(
        id=link_id,
        event_type="tool",
        event_signature="tool:test",
        event_context={},
        outcome_type="result",
        outcome_signature="result:test",
        outcome_valence=Valence.NEGATIVE,
        temporal_delta=TemporalDelta(observed_deltas=(1.0,)),
        predicted_value=predicted_value,
        observation_count=3,
        confidence=0.8,
        memory_ids=memory_ids or [],
    )
    link.last_rpe = last_rpe
    return link


def _make_bridge(hippocampus=None, llm=None):
    from maxim.decisions.nac import NAc
    from maxim.proprioception.pain import PainDetector

    nac = NAc()
    detector = PainDetector()
    return ToolPainBridge(
        nac=nac,
        pain_detector=detector,
        hippocampus=hippocampus,
        llm=llm,
    )


# Lazy import to avoid circular deps at collect time
from maxim.bridges.tool_pain_bridge import ToolPainBridge


class TestReflectionGeneration:
    def test_template_reflection_without_llm(self):
        bridge = _make_bridge()
        link = _make_link(last_rpe=0.6, predicted_value=0.8)
        reflection = bridge._generate_reflection(
            links=[link],
            action={"tool_name": "grab", "params": {}},
            error="object not found",
            context={"goal": "pick up cup"},
        )
        assert reflection is not None
        assert "grab" in reflection
        assert "object not found" in reflection
        assert "0.80" in reflection  # predicted_value

    def test_no_reflection_below_rpe_threshold(self):
        bridge = _make_bridge()
        link = _make_link(last_rpe=0.1)
        reflection = bridge._generate_reflection(
            links=[link],
            action={"tool_name": "t"},
            error="err",
            context={},
        )
        assert reflection is None

    def test_rate_limiting(self):
        bridge = _make_bridge()
        link = _make_link(last_rpe=0.6)
        action = {"tool_name": "grab"}

        # First call: should generate
        r1 = bridge._generate_reflection([link], action, "err", {})
        assert r1 is not None

        # Second call within cooldown: should be rate-limited
        r2 = bridge._generate_reflection([link], action, "err", {})
        assert r2 is None

    def test_different_tools_not_rate_limited(self):
        bridge = _make_bridge()
        link = _make_link(last_rpe=0.6)

        r1 = bridge._generate_reflection([link], {"tool_name": "grab"}, "err", {})
        r2 = bridge._generate_reflection([link], {"tool_name": "scan"}, "err", {})
        assert r1 is not None
        assert r2 is not None

    def test_llm_reflection_when_available(self):
        llm = MagicMock()
        llm.generate.return_value = "The grab tool failed because the object was out of reach."
        bridge = _make_bridge(llm=llm)
        link = _make_link(last_rpe=0.6)

        reflection = bridge._generate_reflection(
            [link], {"tool_name": "grab"}, "out of reach", {},
        )
        assert reflection == "The grab tool failed because the object was out of reach."
        llm.generate.assert_called_once()

    def test_llm_failure_returns_none(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM unavailable")
        bridge = _make_bridge(llm=llm)
        link = _make_link(last_rpe=0.6)

        reflection = bridge._generate_reflection(
            [link], {"tool_name": "grab"}, "err", {},
        )
        assert reflection is None


class TestReflectionStorage:
    def test_stores_in_hippocampus(self):
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        hippocampus = Hippocampus(HippocampusConfig())
        bridge = _make_bridge(hippocampus=hippocampus)

        initial_count = hippocampus._stats.get("memories_captured", 0)
        bridge._store_reflection("tool failed due to X", {"tool_name": "grab", "params": {}})

        # Verify a memory was captured
        assert hippocampus._stats["memories_captured"] == initial_count + 1

        # Find the reflection memory and verify its structure
        for mem in hippocampus._memories.values():
            if hasattr(mem, "outcome") and hasattr(mem.outcome, "result"):
                if isinstance(mem.outcome.result, dict) and "reflection" in mem.outcome.result:
                    assert mem.outcome.result["reflection"] == "tool failed due to X"
                    assert mem.outcome.success is False
                    assert mem.action.tool_name == "grab"
                    assert mem.perception.salience == 0.9
                    break
        else:
            pytest.fail("Reflection memory not found in hippocampus")

    def test_no_store_without_hippocampus(self):
        bridge = _make_bridge(hippocampus=None)
        # Should not raise
        bridge._store_reflection("reflection text", {"tool_name": "t"})

    def test_no_store_with_empty_reflection(self):
        hippocampus = MagicMock()
        bridge = _make_bridge(hippocampus=hippocampus)
        bridge._store_reflection("", {"tool_name": "t"})
        hippocampus.capture.assert_not_called()


class TestReflectionEndToEnd:
    """Test that _on_pain triggers reflection for surprising failures."""

    def test_on_pain_generates_reflection_for_high_rpe(self):
        from maxim.decisions.nac import NAc
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
        from maxim.proprioception.pain import PainDetector, PainSignal, PainType

        nac = NAc()
        detector = PainDetector()
        hippocampus = Hippocampus(HippocampusConfig())

        bridge = ToolPainBridge(nac=nac, pain_detector=detector, hippocampus=hippocampus)

        # Record an event first so NAc has something to update
        bridge.record_tool_start("grab", "inv-1", context={"goal": "pick up"})

        initial_count = hippocampus._stats.get("memories_captured", 0)

        # Simulate pain signal
        signal = PainSignal(
            pain_type=PainType.TOOL_FAILURE,
            intensity=0.8,
            timestamp=time.time(),
            context={"tool_name": "grab", "invocation_id": "inv-1", "error": "collision"},
        )

        # _on_pain handles the signal
        bridge._on_pain(signal)

        # For a first-time event, NAc RPE = |0 - 0.5| = 0.5 > 0.3
        # so a reflection should be generated and stored
        assert bridge._last_rpe > 0.3, f"Expected RPE > 0.3 but got {bridge._last_rpe}"
        assert hippocampus._stats["memories_captured"] == initial_count + 1
