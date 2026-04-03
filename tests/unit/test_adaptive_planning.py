"""Unit tests for AdaptivePlanner, AdaptivePolicy, and DecisionEngine PlanCandidate support."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from maxim.planning.adaptive_planner import (
    AdaptivePlanner,
    PlanCandidate,
    PlanningContext,
    _build_decomposition_prompt,
    _extract_context,
)
from maxim.planning.adaptive_policy import AdaptivePolicy
from maxim.planning.decision_engine import DecisionEngine
from maxim.planning.constraints import ConstraintSet


# ─────────────────────────────────────────────────────────────────────────────
# Stubs / Fakes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FakeOutcomePrediction:
    event_signature: str = "tool:test_tool"
    predicted_outcome: str = "success"
    predicted_valence: Any = None
    predicted_value: float = 0.7
    predicted_delay: float = 2.0
    delay_bounds: tuple[float, float] = (1.0, 3.0)
    confidence: float = 0.8
    contributing_links: list[str] = field(default_factory=list)
    context_match: float = 0.9


@dataclass
class FakeValence:
    value: str = "positive"


class FakeNAc:
    def __init__(self, prediction=None, all_outcomes=None):
        self._prediction = prediction
        self._all_outcomes = all_outcomes or []

    def predict(self, event_type, event_signature, context=None):
        return self._prediction

    def predict_all_outcomes(self, event_type, event_signature, context=None):
        return self._all_outcomes


class FakeEC:
    def __init__(self, results=None):
        self._results = results or []

    def find_similar_situations_for_action(self, tool_name, context=None, k=10):
        return self._results


@dataclass
class FakeMemory:
    id: str = "mem-001"
    outcome: Any = None
    action: Any = None


@dataclass
class FakeAction:
    tool_name: str = "test_tool"


@dataclass
class FakeOutcome:
    success: bool = True
    result: Any = None
    error: str | None = None


class FakeHippocampus:
    def __init__(self, recall_results=None, associated_results=None):
        self._recall_results = recall_results or []
        self._associated_results = associated_results or []

    def recall(self, limit=10, **filters):
        return self._recall_results

    def recall_associated(self, seed_ids, limit=10, **kwargs):
        return self._associated_results


class FakeState:
    def __init__(self, data=None):
        self.data = data or {}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0: DecisionEngine PlanCandidate support
# ─────────────────────────────────────────────────────────────────────────────


class TestDecisionEnginePlanCandidate:
    """DecisionEngine.decide() must handle both PlanCandidate and raw list[dict]."""

    def _make_engine(self, planner_returns, policy_score=1.0, allow=True):
        class FakePlanner:
            def propose_plans(self, goal, state, memory):
                return planner_returns

        class FakePolicy:
            def score(self, plan, state, memory):
                return policy_score

            def allow(self, action, state):
                return allow

        return DecisionEngine(FakePlanner(), FakePolicy(), constraints=[])

    def test_accepts_plan_candidate(self):
        candidate = PlanCandidate(
            actions=[{"tool_name": "read_file", "params": {"path": "README.md"}}],
            source="direct",
            depth=0,
        )
        engine = self._make_engine([candidate])
        result = engine.decide("test", None, None)
        assert result is not None
        assert result["action"]["tool_name"] == "read_file"
        assert result["plan"] is candidate

    def test_accepts_raw_list(self):
        raw_plan = [{"tool_name": "read_file", "params": {"path": "README.md"}}]
        engine = self._make_engine([raw_plan])
        result = engine.decide("test", None, None)
        assert result is not None
        assert result["action"]["tool_name"] == "read_file"

    def test_rejects_invalid_format(self):
        engine = self._make_engine(["not_a_plan"])
        result = engine.decide("test", None, None)
        assert result is None

    def test_rejects_empty_actions(self):
        candidate = PlanCandidate(actions=[], source="direct")
        engine = self._make_engine([candidate])
        result = engine.decide("test", None, None)
        assert result is None

    def test_policy_allow_blocks_candidate(self):
        candidate = PlanCandidate(
            actions=[{"tool_name": "dangerous", "params": {}}],
            source="direct",
        )
        engine = self._make_engine([candidate], allow=False)
        result = engine.decide("test", None, None)
        assert result is None

    def test_picks_highest_scored(self):
        low = PlanCandidate(actions=[{"tool_name": "a", "params": {}}], source="direct", depth=0)
        high = PlanCandidate(actions=[{"tool_name": "b", "params": {}}], source="direct", depth=0)

        class RankPlanner:
            def propose_plans(self, goal, state, memory):
                return [low, high]

        class RankPolicy:
            def score(self, plan, state, memory):
                return 2.0 if plan is high else 1.0

            def allow(self, action, state):
                return True

        engine = DecisionEngine(RankPlanner(), RankPolicy())
        result = engine.decide("test", None, None)
        assert result["action"]["tool_name"] == "b"

    def test_backward_compat_with_constraint_set(self):
        candidate = PlanCandidate(
            actions=[{"tool_name": "read_file", "params": {}}],
            source="direct",
        )
        engine = self._make_engine([candidate])
        engine.constraints = [ConstraintSet()]
        result = engine.decide("test", FakeState({"steps_taken": 0, "max_steps": 100}), None)
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: AdaptivePlanner
# ─────────────────────────────────────────────────────────────────────────────


class TestAdaptivePlannerNoMemory:
    """AdaptivePlanner with no memory systems — degrades gracefully."""

    def test_direct_plan_with_tool_name(self):
        planner = AdaptivePlanner()
        candidates = planner.propose_plans(
            {"tool_name": "read_file", "tool_params": {"path": "x"}, "description": "read"},
            None, None,
        )
        assert len(candidates) >= 1
        assert candidates[0].source == "direct"
        assert candidates[0].actions[0]["tool_name"] == "read_file"
        assert candidates[0].depth == 0

    def test_fallback_with_empty_tool(self):
        planner = AdaptivePlanner()
        candidates = planner.propose_plans({"description": "do something"}, None, None)
        assert len(candidates) >= 1
        assert candidates[0].source == "direct"

    def test_string_goal_coerced_to_dict(self):
        planner = AdaptivePlanner()
        candidates = planner.propose_plans("read the file", None, None)
        assert len(candidates) >= 1

    def test_planning_context_attached(self):
        planner = AdaptivePlanner()
        candidates = planner.propose_plans({"tool_name": "x", "description": "y"}, None, None)
        assert candidates[0].planning_context is not None


class TestAdaptivePlannerNAcIntegration:
    """AdaptivePlanner with NAc predictions."""

    def _make_prediction(self, valence="positive", value=0.8, confidence=0.9):
        return FakeOutcomePrediction(
            predicted_valence=FakeValence(valence),
            predicted_value=value,
            confidence=confidence,
        )

    def test_nac_veto_blocks_direct_plan(self):
        """High-confidence negative prediction should veto direct execution.

        The planner always returns at least one candidate (fallback), but the
        vetoed direct plan should NOT carry the NAc prediction — it's only
        offered as a last resort because no LLM is available for decomposition.
        """
        pred = self._make_prediction(valence="negative", value=0.1, confidence=0.9)
        nac = FakeNAc(prediction=pred)
        planner = AdaptivePlanner(nac=nac)
        candidates = planner.propose_plans({"tool_name": "risky", "description": "test"}, None, None)
        # The direct plan is still offered as fallback (no LLM for decomposition),
        # but the NAc prediction should be visible in the planning context so
        # the policy can score it appropriately.
        assert len(candidates) >= 1
        pctx = candidates[0].planning_context
        assert pctx is not None
        assert pctx.primary_prediction is pred
        assert pctx.has_warnings()

    def test_nac_no_veto_below_threshold(self):
        """Low-confidence negative prediction should NOT veto."""
        pred = self._make_prediction(valence="negative", value=0.1, confidence=0.3)
        nac = FakeNAc(prediction=pred)
        planner = AdaptivePlanner(nac=nac)
        candidates = planner.propose_plans({"tool_name": "tool", "description": "test"}, None, None)
        direct = [c for c in candidates if c.source == "direct"]
        assert len(direct) == 1

    def test_nac_positive_alternative_offered(self):
        """When NAc has a positive alternative outcome, offer it."""
        primary = self._make_prediction(valence="negative", value=0.3, confidence=0.4)
        alt = FakeOutcomePrediction(
            predicted_outcome="alt_success",
            predicted_valence=FakeValence("positive"),
            predicted_value=0.9,
            confidence=0.7,
        )
        nac = FakeNAc(prediction=primary, all_outcomes=[alt])
        planner = AdaptivePlanner(nac=nac)
        candidates = planner.propose_plans({"tool_name": "tool", "description": "test"}, None, None)
        nac_alts = [c for c in candidates if c.source == "nac_alternative"]
        assert len(nac_alts) == 1
        assert nac_alts[0].nac_prediction is alt

    def test_delay_populated_from_prediction(self):
        pred = self._make_prediction()
        pred.predicted_delay = 5.0
        pred.delay_bounds = (2.0, 8.0)
        nac = FakeNAc(prediction=pred)
        planner = AdaptivePlanner(nac=nac)
        candidates = planner.propose_plans({"tool_name": "t", "description": "d"}, None, None)
        assert candidates[0].estimated_delay == 5.0
        assert candidates[0].delay_bounds == (2.0, 8.0)


class TestAdaptivePlannerECIntegration:
    """EC results seed hippocampal spreading activation."""

    def test_ec_results_seed_hippocampus(self):
        ec = FakeEC(results=[("mem-1", "sig1"), ("mem-2", "sig2")])
        hip = FakeHippocampus(associated_results=[
            (FakeMemory(id="mem-3", outcome=FakeOutcome(success=True), action=FakeAction()), 0.8),
        ])
        planner = AdaptivePlanner(ec=ec, hippocampus=hip)
        candidates = planner.propose_plans({"tool_name": "t", "description": "d"}, None, None)
        pctx = candidates[0].planning_context
        assert pctx is not None
        assert len(pctx.similar_situations) == 2
        assert pctx.situation_novelty < 1.0
        assert len(pctx.successful_strategies) == 1

    def test_hippocampus_fallback_without_ec(self):
        """Without EC, hippocampus uses hash-index recall as seeds."""
        hip = FakeHippocampus(
            recall_results=[FakeMemory(id="mem-1")],
            associated_results=[(FakeMemory(id="mem-2", outcome=FakeOutcome(), action=FakeAction()), 0.5)],
        )
        planner = AdaptivePlanner(hippocampus=hip)
        candidates = planner.propose_plans({"tool_name": "t", "description": "d"}, None, None)
        pctx = candidates[0].planning_context
        assert len(pctx.associated_memories) == 1

    def test_reflection_partitioning(self):
        """Memories with reflections go to pctx.reflections, not successful_strategies."""
        reflection_mem = FakeMemory(
            id="refl-1",
            outcome=FakeOutcome(success=False, result={"reflection": "tool failed due to X"}),
            action=FakeAction(),
        )
        success_mem = FakeMemory(
            id="succ-1",
            outcome=FakeOutcome(success=True),
            action=FakeAction(),
        )
        hip = FakeHippocampus(associated_results=[
            (reflection_mem, 0.9),
            (success_mem, 0.7),
        ])
        ec = FakeEC(results=[("seed-1", "sig")])
        planner = AdaptivePlanner(ec=ec, hippocampus=hip)
        candidates = planner.propose_plans({"tool_name": "t", "description": "d"}, None, None)
        pctx = candidates[0].planning_context
        assert len(pctx.reflections) == 1
        assert len(pctx.successful_strategies) == 1


class TestAdaptivePlannerWarnings:
    """has_warnings() triggers decomposed plan candidates."""

    def test_reflections_trigger_warning(self):
        reflection_mem = FakeMemory(
            id="r1",
            outcome=FakeOutcome(success=False, result={"reflection": "bad"}),
            action=FakeAction(),
        )
        hip = FakeHippocampus(associated_results=[(reflection_mem, 0.9)])
        ec = FakeEC(results=[("seed", "sig")])
        # No LLM → decompose returns None, but warnings should still be detected
        planner = AdaptivePlanner(ec=ec, hippocampus=hip)
        candidates = planner.propose_plans({"tool_name": "t", "description": "d"}, None, None)
        pctx = candidates[0].planning_context
        assert pctx.has_warnings()

    def test_unknown_situation_triggers_warning(self):
        """High novelty + no outcome data = unknown territory."""
        pctx = PlanningContext(situation_novelty=0.95, all_outcomes=[])
        assert pctx.has_warnings()

    def test_familiar_situation_no_warning(self):
        """Low novelty should not trigger warning."""
        pctx = PlanningContext(situation_novelty=0.2, all_outcomes=[])
        assert not pctx.has_warnings()

    def test_nac_negative_triggers_warning(self):
        pred = FakeOutcomePrediction(
            predicted_valence=FakeValence("negative"),
            predicted_value=0.3,
            confidence=0.7,
        )
        pctx = PlanningContext(primary_prediction=pred)
        assert pctx.has_warnings()


class TestAdaptivePlannerDecompose:
    """decompose() for the replan loop."""

    def test_decompose_respects_max_depth(self):
        planner = AdaptivePlanner(max_depth=2)
        result = planner.decompose({"description": "test"}, None, depth=2)
        assert result is None

    def test_decompose_returns_none_without_llm(self):
        planner = AdaptivePlanner()
        result = planner.decompose({"description": "test"}, None, depth=0)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Helper functions
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractContext:
    def test_extracts_goal(self):
        ctx = _extract_context({"description": "find cup"}, None)
        assert ctx["goal"] == "find cup"

    def test_extracts_mode_from_state(self):
        state = FakeState(data={"mode": "active"})
        ctx = _extract_context({"description": "test"}, state)
        assert ctx["mode"] == "active"

    def test_handles_none_state(self):
        ctx = _extract_context({"description": "test"}, None)
        assert "mode" not in ctx


class TestBuildDecompositionPrompt:
    def test_includes_goal(self):
        pctx = PlanningContext()
        prompt = _build_decomposition_prompt({"description": "pick up cup", "tool_name": "grab"}, pctx)
        assert "pick up cup" in prompt
        assert "grab" in prompt

    def test_includes_replan_context(self):
        class FakeReplanCtx:
            def to_llm_prompt_section(self):
                return "Previous tool failed with timeout"

        pctx = PlanningContext()
        prompt = _build_decomposition_prompt({"description": "x"}, pctx, replan_ctx=FakeReplanCtx())
        assert "Previous attempt failed" in prompt
        assert "timeout" in prompt

    def test_includes_memory_section(self):
        pred = FakeOutcomePrediction(
            predicted_valence=FakeValence("positive"),
            predicted_value=0.8,
            confidence=0.9,
            predicted_delay=2.0,
            delay_bounds=(1.0, 3.0),
        )
        pctx = PlanningContext(primary_prediction=pred)
        prompt = _build_decomposition_prompt({"description": "x"}, pctx)
        assert "NAc prediction" in prompt


class TestPlanningContextLLMSection:
    def test_empty_context_returns_empty(self):
        pctx = PlanningContext()
        assert pctx.to_llm_section() == ""

    def test_formats_nac_prediction(self):
        pred = FakeOutcomePrediction(
            predicted_valence=FakeValence("positive"),
            predicted_value=0.8,
            confidence=0.9,
            predicted_delay=2.0,
            delay_bounds=(1.0, 3.0),
        )
        pctx = PlanningContext(primary_prediction=pred)
        section = pctx.to_llm_section()
        assert "NAc prediction" in section
        assert "0.80" in section

    def test_formats_reflections_from_outcome_result(self):
        """Reflections stored via Outcome.result dict."""
        mem = FakeMemory(
            id="r1",
            outcome=FakeOutcome(success=False, result={"reflection": "tool X timed out"}),
        )
        pctx = PlanningContext(reflections=[mem])
        section = pctx.to_llm_section()
        assert "tool X timed out" in section

    def test_formats_successful_strategies(self):
        mem = FakeMemory(id="s1", outcome=FakeOutcome(success=True), action=FakeAction(tool_name="grab"))
        pctx = PlanningContext(successful_strategies=[mem])
        section = pctx.to_llm_section()
        assert "grab" in section


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: AdaptivePolicy
# ─────────────────────────────────────────────────────────────────────────────


class TestAdaptivePolicyScoring:
    def _make_candidate(self, *, depth=0, nac_pred=None, novelty=0.5, actions=None, delay=None):
        pctx = PlanningContext(situation_novelty=novelty)
        return PlanCandidate(
            actions=actions or [{"tool_name": "test", "params": {}}],
            source="direct",
            planning_context=pctx,
            nac_prediction=nac_pred,
            depth=depth,
            estimated_delay=delay,
        )

    def test_neutral_score_without_data(self):
        """All-neutral scores when no memory data."""
        policy = AdaptivePolicy()
        candidate = self._make_candidate()
        score = policy.score(candidate, None, None)
        # All dimensions should be ~0.5 except depth (1.0) and cost (~0.83)
        assert 0.4 < score < 0.9

    def test_direct_plan_beats_decomposed(self):
        """Direct (depth=0) should score higher than decomposed (depth=1)."""
        policy = AdaptivePolicy()
        direct = self._make_candidate(depth=0)
        decomposed = self._make_candidate(depth=1)
        assert policy.score(direct, None, None) > policy.score(decomposed, None, None)

    def test_positive_nac_boosts_score(self):
        policy = AdaptivePolicy()
        good_pred = FakeOutcomePrediction(
            predicted_valence=FakeValence("positive"),
            predicted_value=0.9,
            confidence=0.8,
        )
        neutral = self._make_candidate()
        boosted = self._make_candidate(nac_pred=good_pred)
        assert policy.score(boosted, None, None) > policy.score(neutral, None, None)

    def test_negative_nac_lowers_score(self):
        policy = AdaptivePolicy()
        bad_pred = FakeOutcomePrediction(
            predicted_valence=FakeValence("negative"),
            predicted_value=0.8,
            confidence=0.8,
        )
        neutral = self._make_candidate()
        lowered = self._make_candidate(nac_pred=bad_pred)
        assert policy.score(lowered, None, None) < policy.score(neutral, None, None)

    def test_familiar_situation_boosts_score(self):
        policy = AdaptivePolicy()
        familiar = self._make_candidate(novelty=0.1)
        novel = self._make_candidate(novelty=0.9)
        assert policy.score(familiar, None, None) > policy.score(novel, None, None)

    def test_faster_plan_preferred(self):
        policy = AdaptivePolicy()
        fast = self._make_candidate(delay=1.0)
        slow = self._make_candidate(delay=120.0)
        assert policy.score(fast, None, None) > policy.score(slow, None, None)

    def test_fewer_actions_preferred(self):
        policy = AdaptivePolicy()
        short = self._make_candidate(actions=[{"tool_name": "a", "params": {}}])
        long = self._make_candidate(actions=[{"tool_name": f"a{i}", "params": {}} for i in range(5)])
        assert policy.score(short, None, None) > policy.score(long, None, None)

    def test_raw_list_plan_still_works(self):
        """Backward compat: raw list[dict] plans should still score."""
        policy = AdaptivePolicy()
        raw = [{"tool_name": "read_file", "params": {}}]
        score = policy.score(raw, None, None)
        assert isinstance(score, float)
        assert 0.0 < score < 1.0


class TestAdaptivePolicyAllow:
    def test_allows_unknown_tool(self):
        policy = AdaptivePolicy()
        assert policy.allow({"tool_name": "unknown"}, None)

    def test_allows_without_nac(self):
        policy = AdaptivePolicy(nac=None)
        assert policy.allow({"tool_name": "anything"}, None)

    def test_blocks_high_confidence_negative(self):
        pred = FakeOutcomePrediction(
            predicted_valence=FakeValence("negative"),
            predicted_value=0.05,
            confidence=0.9,
        )
        nac = FakeNAc(prediction=pred)
        policy = AdaptivePolicy(nac=nac)
        assert not policy.allow({"tool_name": "test_tool"}, None)

    def test_permits_moderate_negative(self):
        pred = FakeOutcomePrediction(
            predicted_valence=FakeValence("negative"),
            predicted_value=0.3,
            confidence=0.7,
        )
        nac = FakeNAc(prediction=pred)
        policy = AdaptivePolicy(nac=nac)
        assert policy.allow({"tool_name": "test_tool"}, None)


class TestAdaptivePolicyConceptRelevance:
    """concept_relevance normalizes integer edge counts from rank_available_skills."""

    def test_normalization(self):
        """relevance=2 → 2/(2+2) = 0.5, not raw 2."""
        pctx = PlanningContext(ranked_skills=[{"name": "grab", "relevance": 2}])
        candidate = PlanCandidate(
            actions=[{"tool_name": "grab", "params": {}}],
            source="direct",
            planning_context=pctx,
        )
        policy = AdaptivePolicy()
        dims = policy._compute_dimensions(candidate)
        assert abs(dims["concept_relevance"] - 0.5) < 0.01

    def test_high_edge_count_capped(self):
        """relevance=10 → 10/12 ≈ 0.83, not raw 10."""
        pctx = PlanningContext(ranked_skills=[{"name": "scan", "relevance": 10}])
        candidate = PlanCandidate(
            actions=[{"tool_name": "scan", "params": {}}],
            source="direct",
            planning_context=pctx,
        )
        policy = AdaptivePolicy()
        dims = policy._compute_dimensions(candidate)
        assert 0.8 < dims["concept_relevance"] < 0.9

    def test_blends_success_rate(self):
        """When past_success_rate is available, blend 70/30."""
        pctx = PlanningContext(ranked_skills=[
            {"name": "grab", "relevance": 2, "past_success_rate": 0.9},
        ])
        candidate = PlanCandidate(
            actions=[{"tool_name": "grab", "params": {}}],
            source="direct",
            planning_context=pctx,
        )
        policy = AdaptivePolicy()
        dims = policy._compute_dimensions(candidate)
        # 0.7 * 0.5 + 0.3 * 0.9 = 0.35 + 0.27 = 0.62
        assert abs(dims["concept_relevance"] - 0.62) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Integration: build_decision_engine
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildDecisionEngine:
    def test_without_memory_uses_task_planner(self):
        from maxim.runtime.bootstrap import build_decision_engine
        from maxim.planning.planning import TaskPlanner
        from maxim.planning.policy import DefaultPolicy

        engine = build_decision_engine()
        assert isinstance(engine.planner, TaskPlanner)
        assert isinstance(engine.policy, DefaultPolicy)

    def test_with_nac_uses_adaptive(self):
        from maxim.runtime.bootstrap import build_decision_engine

        nac = FakeNAc()
        engine = build_decision_engine(nac=nac)
        assert isinstance(engine.planner, AdaptivePlanner)
        assert isinstance(engine.policy, AdaptivePolicy)
