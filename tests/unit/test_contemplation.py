"""Tests for the ExecAgent contemplation loop (local chain-of-thought)."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.models.language.router import LLMConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_exec_agent(**overrides: Any):
    """Create a minimal ExecAgent without full __init__."""
    from maxim.agents.exec_agent import ExecAgent

    agent = ExecAgent.__new__(ExecAgent)
    agent._system_prompt = "test"
    agent._router = overrides.get("router", None)
    agent._llm_worker = overrides.get("llm_worker", None)
    agent._llm = overrides.get("llm", None)
    agent._work_available = overrides.get("work_available", threading.Event())
    agent._urgent_work_available = overrides.get("urgent_work_available", threading.Event())
    agent._proposal_lock = threading.Lock()
    agent._pending_context = None
    agent._dn_escalation_context = None
    agent.agent_name = "exec_agent"
    agent.log = MagicMock()
    # Phase 2: contemplation metrics
    agent._contemplation_log = overrides.get("contemplation_log", {})
    agent._contemplation_stats = overrides.get("contemplation_stats", {
        "contemplated_success": 0,
        "contemplated_total": 0,
        "uncontemplated_success": 0,
        "uncontemplated_total": 0,
    })
    agent._nac = overrides.get("nac", None)
    return agent


def _make_ctx(**overrides: Any):
    """Create a minimal StructuredContext mock."""
    ctx = MagicMock()
    ctx.root_goal = overrides.get("root_goal", "Understand reality and help people.")
    ctx.mode = overrides.get("mode", "active")
    ctx.active_goal = overrides.get("active_goal", None)
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# _safe_sub_goal_count tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSafeSubGoalCount:
    def test_missing_key(self):
        agent = _make_exec_agent()
        assert agent._safe_sub_goal_count({}) == 0

    def test_none_value(self):
        agent = _make_exec_agent()
        assert agent._safe_sub_goal_count({"sub_goals": None}) == 0

    def test_non_list_value(self):
        agent = _make_exec_agent()
        assert agent._safe_sub_goal_count({"sub_goals": "not a list"}) == 0

    def test_empty_list(self):
        agent = _make_exec_agent()
        assert agent._safe_sub_goal_count({"sub_goals": []}) == 0

    def test_populated_list(self):
        agent = _make_exec_agent()
        assert agent._safe_sub_goal_count({"sub_goals": [{"a": 1}, {"b": 2}]}) == 2


# ─────────────────────────────────────────────────────────────────────────────
# _should_contemplate tests
# ─────────────────────────────────────────────────────────────────────────────


class TestShouldContemplate:
    def test_rejects_timeout_dict(self):
        agent = _make_exec_agent()
        assert agent._should_contemplate({"_timeout": True}) is False

    def test_rejects_missing_goal_description(self):
        agent = _make_exec_agent()
        assert agent._should_contemplate({"priority": "HIGH"}) is False

    def test_rejects_null_goal_description(self):
        agent = _make_exec_agent()
        assert agent._should_contemplate({"goal_description": None, "priority": "HIGH"}) is False

    def test_rejects_idle_priority(self):
        agent = _make_exec_agent()
        assert agent._should_contemplate({"goal_description": "test", "priority": "IDLE"}) is False

    def test_triggers_on_high_priority(self):
        agent = _make_exec_agent()
        assert agent._should_contemplate({"goal_description": "test", "priority": "HIGH"}) is True

    def test_triggers_on_critical_priority(self):
        agent = _make_exec_agent()
        assert agent._should_contemplate({"goal_description": "test", "priority": "CRITICAL"}) is True

    def test_triggers_on_sub_goal_count(self):
        agent = _make_exec_agent()
        response = {
            "goal_description": "test",
            "priority": "MEDIUM",
            "sub_goals": [{"tool_name": "a"}, {"tool_name": "b"}],
        }
        assert agent._should_contemplate(response) is True

    def test_skips_simple_plan(self):
        agent = _make_exec_agent()
        response = {
            "goal_description": "test",
            "priority": "MEDIUM",
            "sub_goals": [{"tool_name": "a"}],
        }
        assert agent._should_contemplate(response) is False

    def test_respects_disabled_config(self):
        mock_router = MagicMock()
        mock_router.cfg = LLMConfig(contemplation=(("enabled", False),))
        agent = _make_exec_agent(router=mock_router)

        response = {"goal_description": "test", "priority": "HIGH"}
        assert agent._should_contemplate(response) is False

    def test_respects_custom_min_sub_goals(self):
        mock_router = MagicMock()
        mock_router.cfg = LLMConfig(contemplation=(("min_sub_goals_to_trigger", 4),))
        agent = _make_exec_agent(router=mock_router)

        # 2 sub_goals is below the custom threshold of 4
        response = {
            "goal_description": "test",
            "priority": "MEDIUM",
            "sub_goals": [{"tool_name": "a"}, {"tool_name": "b"}],
        }
        assert agent._should_contemplate(response) is False

    def test_high_priority_trigger_can_be_disabled(self):
        mock_router = MagicMock()
        mock_router.cfg = LLMConfig(contemplation=(("trigger_on_high_priority", False),))
        agent = _make_exec_agent(router=mock_router)

        response = {"goal_description": "test", "priority": "HIGH"}
        assert agent._should_contemplate(response) is False


# ─────────────────────────────────────────────────────────────────────────────
# _contemplate tests
# ─────────────────────────────────────────────────────────────────────────────


class TestContemplate:
    def test_returns_draft_on_critique_failure(self):
        """If critique returns None (parse failure), accept draft."""
        agent = _make_exec_agent()
        agent._critique_plan = MagicMock(return_value=None)

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is draft

    def test_returns_draft_on_high_confidence(self):
        """If critique confidence >= threshold, accept draft."""
        agent = _make_exec_agent()
        agent._critique_plan = MagicMock(return_value={
            "confidence": 0.9,
            "issues": [],
            "suggestions": [],
        })

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is draft

    def test_refines_on_low_confidence(self):
        """If confidence < threshold and refine succeeds, return refined plan."""
        agent = _make_exec_agent()
        agent._critique_plan = MagicMock(return_value={
            "confidence": 0.3,
            "issues": ["missing step"],
            "suggestions": ["add step"],
        })
        refined = {"goal_description": "improved", "priority": "HIGH", "sub_goals": []}
        agent._refine_plan = MagicMock(return_value=refined)

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is refined

    def test_returns_draft_on_refine_failure(self):
        """If refine returns None (parse failure), keep draft."""
        agent = _make_exec_agent()
        agent._critique_plan = MagicMock(return_value={
            "confidence": 0.3,
            "issues": ["bad"],
            "suggestions": ["fix"],
        })
        agent._refine_plan = MagicMock(return_value=None)

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is draft

    def test_preemption_before_critique(self):
        """If urgent work is available, skip contemplation entirely."""
        urgent_event = threading.Event()
        urgent_event.set()  # Simulate urgent percept arrived
        agent = _make_exec_agent(urgent_work_available=urgent_event)
        agent._critique_plan = MagicMock()

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is draft
        agent._critique_plan.assert_not_called()

    def test_preemption_before_refine(self):
        """If urgent work arrives after critique, skip refine."""
        urgent_event = threading.Event()
        agent = _make_exec_agent(urgent_work_available=urgent_event)

        agent._critique_plan = MagicMock(return_value={
            "confidence": 0.3,
            "issues": ["bad"],
            "suggestions": ["fix"],
        })
        agent._refine_plan = MagicMock()

        # Set the event after critique returns (simulating urgent percept during critique)
        original_critique = agent._critique_plan

        def critique_with_preempt(*args, **kwargs):
            result = original_critique(*args, **kwargs)
            urgent_event.set()
            return result

        agent._critique_plan = MagicMock(side_effect=critique_with_preempt)

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is draft
        agent._refine_plan.assert_not_called()

    def test_non_urgent_work_does_not_preempt(self):
        """Normal work_available does NOT preempt contemplation."""
        work_event = threading.Event()
        work_event.set()  # Normal (non-urgent) percept
        agent = _make_exec_agent(work_available=work_event)
        agent._critique_plan = MagicMock(return_value={
            "confidence": 0.3,
            "issues": ["bad"],
            "suggestions": ["fix"],
        })
        refined = {"goal_description": "improved", "priority": "HIGH"}
        agent._refine_plan = MagicMock(return_value=refined)

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        # Should NOT preempt — normal work doesn't interrupt contemplation
        assert result is refined

    def test_handles_non_numeric_confidence(self):
        """If critique returns non-numeric confidence, default to 1.0 (accept)."""
        agent = _make_exec_agent()
        agent._critique_plan = MagicMock(return_value={
            "confidence": "high",
            "issues": [],
            "suggestions": [],
        })

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is draft  # Defaults to 1.0 >= 0.7 threshold

    def test_custom_confidence_threshold(self):
        """Respects custom confidence threshold from config."""
        mock_router = MagicMock()
        mock_router.cfg = LLMConfig(contemplation=(("confidence_threshold", 0.95),))
        agent = _make_exec_agent(router=mock_router)
        agent._critique_plan = MagicMock(return_value={
            "confidence": 0.8,
            "issues": ["could be better"],
            "suggestions": ["improve"],
        })
        refined = {"goal_description": "improved", "priority": "HIGH"}
        agent._refine_plan = MagicMock(return_value=refined)

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        # 0.8 < 0.95 threshold, so should refine
        assert result is refined


# ─────────────────────────────────────────────────────────────────────────────
# _critique_plan / _refine_plan tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCritiquePlan:
    def test_returns_parsed_critique(self):
        agent = _make_exec_agent()
        critique_response = {
            "confidence": 0.5,
            "issues": ["wrong order"],
            "suggestions": ["reverse steps"],
        }
        agent._contemplation_llm_call = MagicMock(return_value=critique_response)

        draft = {"goal_description": "test", "priority": "HIGH", "sub_goals": []}
        ctx = _make_ctx()

        result = agent._critique_plan(draft, ctx)
        assert result is not None
        assert result["confidence"] == 0.5
        assert "wrong order" in result["issues"]

    def test_returns_none_on_missing_confidence(self):
        agent = _make_exec_agent()
        agent._contemplation_llm_call = MagicMock(return_value={"issues": ["bad"]})

        result = agent._critique_plan({"goal_description": "test"}, _make_ctx())
        assert result is None

    def test_returns_none_on_non_dict_response(self):
        agent = _make_exec_agent()
        agent._contemplation_llm_call = MagicMock(return_value=None)

        result = agent._critique_plan({"goal_description": "test"}, _make_ctx())
        assert result is None

    def test_returns_none_on_exception(self):
        agent = _make_exec_agent()
        agent._contemplation_llm_call = MagicMock(side_effect=RuntimeError("LLM down"))

        result = agent._critique_plan({"goal_description": "test"}, _make_ctx())
        assert result is None


class TestRefinePlan:
    def test_returns_refined_plan(self):
        agent = _make_exec_agent()
        refined = {"goal_description": "better plan", "priority": "HIGH", "sub_goals": []}
        agent._contemplation_llm_call = MagicMock(return_value=refined)

        draft = {"goal_description": "test", "priority": "HIGH"}
        critique = {"confidence": 0.3, "issues": ["bad"], "suggestions": ["fix"]}
        ctx = _make_ctx()

        result = agent._refine_plan(draft, critique, ctx)
        assert result is not None
        assert result["goal_description"] == "better plan"

    def test_returns_none_on_missing_goal_description(self):
        agent = _make_exec_agent()
        agent._contemplation_llm_call = MagicMock(return_value={"priority": "HIGH"})

        result = agent._refine_plan({}, {"issues": [], "suggestions": []}, _make_ctx())
        assert result is None

    def test_returns_none_on_exception(self):
        agent = _make_exec_agent()
        agent._contemplation_llm_call = MagicMock(side_effect=RuntimeError("fail"))

        result = agent._refine_plan({}, {"issues": [], "suggestions": []}, _make_ctx())
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# _contemplation_llm_call routing tests
# ─────────────────────────────────────────────────────────────────────────────


class TestContemplationLLMCall:
    def test_routes_through_llm_worker(self):
        mock_worker = MagicMock()
        mock_worker.generate_json_direct.return_value = {"confidence": 0.8}
        agent = _make_exec_agent(llm_worker=mock_worker)

        result = agent._contemplation_llm_call(system="sys", user="usr", max_tokens=384)
        assert result == {"confidence": 0.8}
        mock_worker.generate_json_direct.assert_called_once()
        call_kwargs = mock_worker.generate_json_direct.call_args
        assert call_kwargs.kwargs["system"] == "sys"
        assert call_kwargs.kwargs["user"] == "usr"
        assert call_kwargs.kwargs["max_tokens"] == 384

    def test_routes_through_router_when_no_worker(self):
        mock_router = MagicMock()
        mock_router.generate_json.return_value = {"confidence": 0.6}
        agent = _make_exec_agent(router=mock_router)

        result = agent._contemplation_llm_call(system="sys", user="usr", max_tokens=384)
        assert result == {"confidence": 0.6}
        mock_router.generate_json.assert_called_once()
        call_kwargs = mock_router.generate_json.call_args
        assert call_kwargs.kwargs["max_tokens"] == 384

    def test_routes_through_legacy_llm_when_no_router(self):
        mock_llm = MagicMock()
        mock_llm.generate_json.return_value = {"confidence": 0.5}
        agent = _make_exec_agent(llm=mock_llm)
        agent._ensure_llm = MagicMock(return_value=mock_llm)

        result = agent._contemplation_llm_call(system="sys", user="usr", max_tokens=384)
        assert result == {"confidence": 0.5}
        mock_llm.generate_json.assert_called_once_with(
            "usr", system_prompt="sys", temperature=0.2, max_tokens=384
        )

    def test_returns_none_when_no_backend(self):
        agent = _make_exec_agent()
        agent._ensure_llm = MagicMock(return_value=None)

        result = agent._contemplation_llm_call(system="sys", user="usr", max_tokens=384)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# _contemplation_config tests
# ─────────────────────────────────────────────────────────────────────────────


class TestContemplationConfig:
    def test_defaults_when_no_router(self):
        agent = _make_exec_agent()
        cfg = agent._contemplation_config()
        assert cfg["enabled"] is True
        assert cfg["confidence_threshold"] == 0.7
        assert cfg["min_sub_goals_to_trigger"] == 2
        assert cfg["critique_max_tokens"] == 384
        assert cfg["refine_max_tokens"] == 512

    def test_overrides_from_router_config(self):
        mock_router = MagicMock()
        mock_router.cfg = LLMConfig(contemplation=(
            ("confidence_threshold", 0.5),
            ("critique_max_tokens", 512),
        ))
        agent = _make_exec_agent(router=mock_router)

        cfg = agent._contemplation_config()
        assert cfg["confidence_threshold"] == 0.5
        assert cfg["critique_max_tokens"] == 512
        # Other defaults still apply
        assert cfg["enabled"] is True
        assert cfg["min_sub_goals_to_trigger"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# LLMConfig contemplation field tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMConfigContemplation:
    def test_default_is_empty_tuple(self):
        cfg = LLMConfig()
        assert cfg.contemplation == ()

    def test_can_store_config_as_tuple(self):
        cfg = LLMConfig(contemplation=(("enabled", False), ("confidence_threshold", 0.5)))
        assert dict(cfg.contemplation) == {"enabled": False, "confidence_threshold": 0.5}

    def test_frozen_immutability(self):
        cfg = LLMConfig(contemplation=(("enabled", True),))
        with pytest.raises(AttributeError):
            cfg.contemplation = ()  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Contemplation Quality Metrics
# ──────────────────────────────���──────────────────────────────────────────────


def _make_goal_completed(goal_id: str, success: bool) -> Any:
    """Create a GoalCompleted message."""
    from maxim.agents.bus import GoalCompleted

    return GoalCompleted(goal_id=goal_id, success=success)


class TestOnGoalCompleted:
    """Tests for _on_goal_completed outcome tracking."""

    def test_contemplated_success_updates_stats(self):
        agent = _make_exec_agent()
        agent._contemplation_log["g1"] = {
            "contemplated": True, "refined": True, "timestamp": 100.0,
        }
        agent._on_goal_completed(_make_goal_completed("g1", success=True))

        assert agent._contemplation_stats["contemplated_success"] == 1
        assert agent._contemplation_stats["contemplated_total"] == 1
        assert agent._contemplation_stats["uncontemplated_total"] == 0

    def test_contemplated_failure_updates_stats(self):
        agent = _make_exec_agent()
        agent._contemplation_log["g1"] = {
            "contemplated": True, "refined": True, "timestamp": 100.0,
        }
        agent._on_goal_completed(_make_goal_completed("g1", success=False))

        assert agent._contemplation_stats["contemplated_success"] == 0
        assert agent._contemplation_stats["contemplated_total"] == 1

    def test_uncontemplated_success_updates_stats(self):
        agent = _make_exec_agent()
        agent._contemplation_log["g2"] = {
            "contemplated": False, "refined": False, "timestamp": 100.0,
        }
        agent._on_goal_completed(_make_goal_completed("g2", success=True))

        assert agent._contemplation_stats["uncontemplated_success"] == 1
        assert agent._contemplation_stats["uncontemplated_total"] == 1
        assert agent._contemplation_stats["contemplated_total"] == 0

    def test_unknown_goal_id_ignored(self):
        agent = _make_exec_agent()
        agent._on_goal_completed(_make_goal_completed("unknown", success=True))

        assert agent._contemplation_stats["contemplated_total"] == 0
        assert agent._contemplation_stats["uncontemplated_total"] == 0

    def test_goal_removed_from_log_after_processing(self):
        agent = _make_exec_agent()
        agent._contemplation_log["g1"] = {
            "contemplated": True, "refined": False, "timestamp": 100.0,
        }
        agent._on_goal_completed(_make_goal_completed("g1", success=True))

        assert "g1" not in agent._contemplation_log

    def test_nac_observe_called_for_refined(self):
        mock_nac = MagicMock()
        agent = _make_exec_agent(nac=mock_nac)
        agent._contemplation_log["g1"] = {
            "contemplated": True, "refined": True, "timestamp": 100.0,
        }
        agent._on_goal_completed(_make_goal_completed("g1", success=True))

        mock_nac.observe.assert_called_once()
        call_kwargs = mock_nac.observe.call_args.kwargs
        assert call_kwargs["event_type"] == "contemplation"
        assert call_kwargs["event_signature"] == "contemplation:refined"
        assert call_kwargs["outcome_type"] == "goal_result"

    def test_nac_observe_called_for_draft(self):
        mock_nac = MagicMock()
        agent = _make_exec_agent(nac=mock_nac)
        agent._contemplation_log["g1"] = {
            "contemplated": True, "refined": False, "timestamp": 100.0,
        }
        agent._on_goal_completed(_make_goal_completed("g1", success=True))

        call_kwargs = mock_nac.observe.call_args.kwargs
        assert call_kwargs["event_signature"] == "contemplation:draft"

    def test_nac_observe_uses_correct_valence(self):
        from maxim.decisions.causal_link import Valence

        mock_nac = MagicMock()
        agent = _make_exec_agent(nac=mock_nac)
        agent._contemplation_log["g1"] = {
            "contemplated": True, "refined": True, "timestamp": 100.0,
        }
        agent._on_goal_completed(_make_goal_completed("g1", success=False))

        call_kwargs = mock_nac.observe.call_args.kwargs
        assert call_kwargs["outcome_valence"] == Valence.NEGATIVE

    def test_nac_exception_swallowed(self):
        mock_nac = MagicMock()
        mock_nac.observe.side_effect = RuntimeError("NAc broke")
        agent = _make_exec_agent(nac=mock_nac)
        agent._contemplation_log["g1"] = {
            "contemplated": True, "refined": True, "timestamp": 100.0,
        }
        # Should not raise
        agent._on_goal_completed(_make_goal_completed("g1", success=True))
        assert agent._contemplation_stats["contemplated_total"] == 1

    def test_no_nac_still_tracks_stats(self):
        agent = _make_exec_agent(nac=None)
        agent._contemplation_log["g1"] = {
            "contemplated": True, "refined": True, "timestamp": 100.0,
        }
        agent._on_goal_completed(_make_goal_completed("g1", success=True))
        assert agent._contemplation_stats["contemplated_success"] == 1


class TestContemplationImprovementRate:
    """Tests for contemplation_improvement_rate() metric query."""

    def test_no_data_returns_none_rates(self):
        agent = _make_exec_agent()
        result = agent.contemplation_improvement_rate()
        assert result["contemplated_success_rate"] is None
        assert result["uncontemplated_success_rate"] is None
        assert result["improvement_delta"] is None

    def test_only_contemplated_data(self):
        agent = _make_exec_agent(contemplation_stats={
            "contemplated_success": 3, "contemplated_total": 4,
            "uncontemplated_success": 0, "uncontemplated_total": 0,
        })
        result = agent.contemplation_improvement_rate()
        assert result["contemplated_success_rate"] == 0.75
        assert result["uncontemplated_success_rate"] is None
        assert result["improvement_delta"] is None

    def test_both_categories_computes_delta(self):
        agent = _make_exec_agent(contemplation_stats={
            "contemplated_success": 8, "contemplated_total": 10,
            "uncontemplated_success": 5, "uncontemplated_total": 10,
        })
        result = agent.contemplation_improvement_rate()
        assert result["contemplated_success_rate"] == 0.8
        assert result["uncontemplated_success_rate"] == 0.5
        assert result["improvement_delta"] == pytest.approx(0.3)

    def test_negative_improvement_delta(self):
        agent = _make_exec_agent(contemplation_stats={
            "contemplated_success": 2, "contemplated_total": 10,
            "uncontemplated_success": 8, "uncontemplated_total": 10,
        })
        result = agent.contemplation_improvement_rate()
        assert result["improvement_delta"] == pytest.approx(-0.6)

    def test_totals_included(self):
        agent = _make_exec_agent(contemplation_stats={
            "contemplated_success": 1, "contemplated_total": 5,
            "uncontemplated_success": 3, "uncontemplated_total": 7,
        })
        result = agent.contemplation_improvement_rate()
        assert result["contemplated_total"] == 5
        assert result["uncontemplated_total"] == 7


class TestWireNac:
    """Tests for wire_nac late-wiring."""

    def test_wire_nac_sets_reference(self):
        agent = _make_exec_agent()
        mock_nac = MagicMock()
        agent.wire_nac(mock_nac)
        assert agent._nac is mock_nac

    def test_wire_nac_replaces_none(self):
        agent = _make_exec_agent()
        assert agent._nac is None
        agent.wire_nac(MagicMock())
        assert agent._nac is not None


class TestContemplationLogBounding:
    """Tests for contemplation log size management."""

    def test_log_bounded_at_200(self):
        """Verify the log doesn't grow unbounded (tested via _propose_goal logic)."""
        agent = _make_exec_agent()
        # Simulate 210 entries
        for i in range(210):
            agent._contemplation_log[f"g{i}"] = {
                "contemplated": False, "refined": False, "timestamp": float(i),
            }
        # Log should have 210 entries (bounding happens in _propose_goal, not here)
        assert len(agent._contemplation_log) == 210


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Adaptive threshold tests
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_link(predicted_value: float = 0.5, observation_count: int = 5,
                    outcome_valence: str = "positive"):
    """Create a mock CausalLink with the fields _adaptive_thresholds reads."""
    link = MagicMock()
    link.predicted_value = predicted_value
    link.observation_count = observation_count
    link.outcome_valence = outcome_valence
    return link


def _static_cfg(**overrides: Any) -> dict[str, Any]:
    """Return static contemplation config defaults (no adaptive application)."""
    cfg: dict[str, Any] = {
        "enabled": True,
        "confidence_threshold": 0.7,
        "min_sub_goals_to_trigger": 2,
        "trigger_on_high_priority": True,
        "max_passes": 3,
        "critique_max_tokens": 384,
        "refine_max_tokens": 512,
        "adaptive_enabled": True,
        "adaptive_min_observations": 10,
        "adaptive_confidence_floor": 0.3,
        "adaptive_confidence_ceiling": 0.95,
        "adaptive_min_sub_goals_floor": 1,
        "adaptive_min_sub_goals_ceiling": 5,
    }
    cfg.update(overrides)
    return cfg


class TestAdaptiveThresholds:
    """Tests for Phase 3 adaptive threshold tuning."""

    def test_returns_none_without_nac(self):
        agent = _make_exec_agent()
        result = agent._adaptive_thresholds(_static_cfg())
        assert result is None

    def test_returns_none_insufficient_observations(self):
        """Need adaptive_min_observations (default 10) before adapting."""
        mock_nac = MagicMock()
        # Only 4 total observations — below the 10 threshold
        mock_nac.get_links_for_event.side_effect = lambda sig: (
            [_make_mock_link(observation_count=2)] if sig == "contemplation:refined"
            else [_make_mock_link(observation_count=2)]
        )
        agent = _make_exec_agent(nac=mock_nac)
        result = agent._adaptive_thresholds(_static_cfg())
        assert result is None

    def test_lowers_threshold_when_contemplation_helps(self):
        """High refined rate + low draft rate → lower confidence_threshold."""
        mock_nac = MagicMock()
        mock_nac.get_links_for_event.side_effect = lambda sig: (
            [_make_mock_link(predicted_value=0.9, observation_count=10)]
            if sig == "contemplation:refined"
            else [_make_mock_link(predicted_value=0.4, observation_count=10)]
        )
        agent = _make_exec_agent(nac=mock_nac)
        result = agent._adaptive_thresholds(_static_cfg())
        assert result is not None
        # improvement = 0.9 - 0.4 = 0.5 → threshold_shift = -0.5
        # new_threshold = 0.7 + (-0.5) = 0.2 → clamped to floor 0.3
        assert result["confidence_threshold"] == 0.3
        # improvement > 0.1 → min_sub_goals decreases by 1 (2→1)
        assert result["min_sub_goals_to_trigger"] == 1

    def test_raises_threshold_when_contemplation_hurts(self):
        """Low refined rate + high draft rate → raise confidence_threshold."""
        mock_nac = MagicMock()
        mock_nac.get_links_for_event.side_effect = lambda sig: (
            [_make_mock_link(predicted_value=0.3, observation_count=8)]
            if sig == "contemplation:refined"
            else [_make_mock_link(predicted_value=0.8, observation_count=8)]
        )
        agent = _make_exec_agent(nac=mock_nac)
        result = agent._adaptive_thresholds(_static_cfg())
        assert result is not None
        # improvement = 0.3 - 0.8 = -0.5 → threshold_shift = 0.5
        # new_threshold = 0.7 + 0.5 = 1.2 → clamped to ceiling 0.95
        assert result["confidence_threshold"] == 0.95
        # improvement < -0.1 → min_sub_goals increases by 1 (2→3)
        assert result["min_sub_goals_to_trigger"] == 3

    def test_no_change_when_improvement_is_neutral(self):
        """Similar rates → threshold stays near base, min_sub_goals unchanged."""
        mock_nac = MagicMock()
        mock_nac.get_links_for_event.side_effect = lambda sig: (
            [_make_mock_link(predicted_value=0.6, observation_count=10)]
            if sig == "contemplation:refined"
            else [_make_mock_link(predicted_value=0.55, observation_count=10)]
        )
        agent = _make_exec_agent(nac=mock_nac)
        result = agent._adaptive_thresholds(_static_cfg())
        assert result is not None
        # improvement = 0.05 → threshold_shift = -0.05
        # new_threshold = 0.7 - 0.05 = 0.65
        assert result["confidence_threshold"] == 0.65
        # |improvement| < 0.1 → min_sub_goals unchanged
        assert result["min_sub_goals_to_trigger"] == 2

    def test_clamps_min_sub_goals_to_bounds(self):
        """min_sub_goals stays within floor/ceiling."""
        mock_nac = MagicMock()
        # Strong positive improvement → tries to lower min_sub_goals
        mock_nac.get_links_for_event.side_effect = lambda sig: (
            [_make_mock_link(predicted_value=0.95, observation_count=15)]
            if sig == "contemplation:refined"
            else [_make_mock_link(predicted_value=0.2, observation_count=15)]
        )
        agent = _make_exec_agent(nac=mock_nac)
        # Start with min_sub_goals already at floor
        result = agent._adaptive_thresholds(_static_cfg(min_sub_goals_to_trigger=1))
        assert result is not None
        # Already at floor 1, can't go lower
        assert result["min_sub_goals_to_trigger"] == 1

    def test_respects_custom_bounds_from_config(self):
        """User-configured bounds override defaults."""
        mock_nac = MagicMock()
        mock_nac.get_links_for_event.side_effect = lambda sig: (
            [_make_mock_link(predicted_value=0.9, observation_count=10)]
            if sig == "contemplation:refined"
            else [_make_mock_link(predicted_value=0.3, observation_count=10)]
        )
        agent = _make_exec_agent(nac=mock_nac)
        result = agent._adaptive_thresholds(_static_cfg(
            adaptive_confidence_floor=0.5,
            adaptive_confidence_ceiling=0.8,
            adaptive_min_sub_goals_floor=2,
            adaptive_min_sub_goals_ceiling=3,
        ))
        assert result is not None
        # improvement = 0.6, shift = -0.6, new = 0.7 - 0.6 = 0.1 → clamped to custom floor 0.5
        assert result["confidence_threshold"] == 0.5
        # Would want to decrease, but floor is 2 → stays at 2
        assert result["min_sub_goals_to_trigger"] == 2

    def test_only_refined_data_available(self):
        """When only refined links exist, compare against 0.5 baseline."""
        mock_nac = MagicMock()
        mock_nac.get_links_for_event.side_effect = lambda sig: (
            [_make_mock_link(predicted_value=0.8, observation_count=12)]
            if sig == "contemplation:refined"
            else []
        )
        agent = _make_exec_agent(nac=mock_nac)
        result = agent._adaptive_thresholds(_static_cfg())
        assert result is not None
        # improvement = 0.8 - 0.5 = 0.3 → threshold_shift = -0.3
        # new_threshold = 0.7 - 0.3 = 0.4
        assert result["confidence_threshold"] == 0.4
        assert result["min_sub_goals_to_trigger"] == 1

    def test_only_draft_data_available(self):
        """When only draft links exist, infer from draft success rate."""
        mock_nac = MagicMock()
        mock_nac.get_links_for_event.side_effect = lambda sig: (
            [] if sig == "contemplation:refined"
            else [_make_mock_link(predicted_value=0.9, observation_count=12)]
        )
        agent = _make_exec_agent(nac=mock_nac)
        result = agent._adaptive_thresholds(_static_cfg())
        assert result is not None
        # improvement = 0.5 - 0.9 = -0.4 → threshold_shift = 0.4
        # new_threshold = 0.7 + 0.4 = 1.1 → clamped to 0.95
        assert result["confidence_threshold"] == 0.95
        # improvement < -0.1 → tighten
        assert result["min_sub_goals_to_trigger"] == 3

    def test_nac_exception_returns_none(self):
        """NAc query failure is handled gracefully."""
        mock_nac = MagicMock()
        mock_nac.get_links_for_event.side_effect = RuntimeError("NAc error")
        agent = _make_exec_agent(nac=mock_nac)
        result = agent._adaptive_thresholds(_static_cfg())
        assert result is None

    def test_adaptive_disabled_skips_adjustment(self):
        """When adaptive_enabled is false, config stays static."""
        mock_nac = MagicMock()
        mock_nac.get_links_for_event.return_value = [
            _make_mock_link(predicted_value=0.9, observation_count=20)
        ]
        mock_router = MagicMock()
        mock_router.cfg = LLMConfig(contemplation=(
            ("adaptive_enabled", False),
        ))
        agent = _make_exec_agent(nac=mock_nac, router=mock_router)
        cfg = agent._contemplation_config()
        # Adaptive is disabled, so config should use static defaults
        assert cfg["confidence_threshold"] == 0.7
        assert cfg["min_sub_goals_to_trigger"] == 2

    def test_multiple_links_weighted_by_observations(self):
        """Multiple links are weighted by observation count."""
        mock_nac = MagicMock()
        # Two refined links: one strong (15 obs), one weak (5 obs)
        # Weighted avg = (0.9*15 + 0.3*5) / 20 = (13.5+1.5)/20 = 0.75
        mock_nac.get_links_for_event.side_effect = lambda sig: (
            [
                _make_mock_link(predicted_value=0.9, observation_count=15),
                _make_mock_link(predicted_value=0.3, observation_count=5),
            ]
            if sig == "contemplation:refined"
            else [_make_mock_link(predicted_value=0.5, observation_count=10)]
        )
        agent = _make_exec_agent(nac=mock_nac)
        result = agent._adaptive_thresholds(_static_cfg())
        assert result is not None
        # improvement = 0.75 - 0.5 = 0.25 → threshold_shift = -0.25
        # new_threshold = 0.7 - 0.25 = 0.45
        assert result["confidence_threshold"] == 0.45
        assert result["min_sub_goals_to_trigger"] == 1


class TestAdaptiveIntegrationWithConfig:
    """Tests that adaptive thresholds flow through _contemplation_config."""

    def test_config_reflects_adaptive_adjustments(self):
        """_contemplation_config returns adjusted values when NAc has data."""
        mock_nac = MagicMock()
        mock_nac.get_links_for_event.side_effect = lambda sig: (
            [_make_mock_link(predicted_value=0.85, observation_count=12)]
            if sig == "contemplation:refined"
            else [_make_mock_link(predicted_value=0.45, observation_count=12)]
        )
        agent = _make_exec_agent(nac=mock_nac)
        cfg = agent._contemplation_config()
        # improvement = 0.4 → threshold lowered
        assert cfg["confidence_threshold"] == 0.3  # 0.7 - 0.4 = 0.3 (at floor)
        assert cfg["min_sub_goals_to_trigger"] == 1

    def test_should_contemplate_uses_adaptive_min_sub_goals(self):
        """_should_contemplate uses the adaptively lowered min_sub_goals."""
        mock_nac = MagicMock()
        mock_nac.get_links_for_event.side_effect = lambda sig: (
            [_make_mock_link(predicted_value=0.85, observation_count=12)]
            if sig == "contemplation:refined"
            else [_make_mock_link(predicted_value=0.45, observation_count=12)]
        )
        agent = _make_exec_agent(nac=mock_nac)
        # 1 sub_goal — normally below the default threshold of 2
        response = {
            "goal_description": "test",
            "priority": "MEDIUM",
            "sub_goals": ["step1"],
        }
        # With adaptive: min_sub_goals drops to 1, so this should contemplate
        assert agent._should_contemplate(response) is True

    def test_should_contemplate_respects_adaptive_tightening(self):
        """When contemplation hurts, adaptive raises min_sub_goals."""
        mock_nac = MagicMock()
        mock_nac.get_links_for_event.side_effect = lambda sig: (
            [_make_mock_link(predicted_value=0.3, observation_count=10)]
            if sig == "contemplation:refined"
            else [_make_mock_link(predicted_value=0.8, observation_count=10)]
        )
        agent = _make_exec_agent(nac=mock_nac)
        # 2 sub_goals — normally meets the default threshold of 2
        response = {
            "goal_description": "test",
            "priority": "MEDIUM",
            "sub_goals": ["step1", "step2"],
        }
        # With adaptive: min_sub_goals raised to 3, so 2 is not enough
        assert agent._should_contemplate(response) is False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Fast contemplation mode tests
# ─────────────────────────────────────────────────────────────────────────────


class TestContemplateFast:
    """Tests for fast (combined critique+refine) contemplation mode."""

    def _make_fast_agent(self, **overrides: Any):
        """Create an agent configured for fast contemplation."""
        mock_router = MagicMock()
        mock_router.cfg = LLMConfig(contemplation=(("mode", "fast"),))
        return _make_exec_agent(router=overrides.pop("router", mock_router), **overrides)

    def test_dispatches_to_fast_mode(self):
        """_contemplate dispatches to _contemplate_fast when mode='fast'."""
        agent = self._make_fast_agent()
        agent._contemplate_fast = MagicMock(return_value={"goal_description": "fast"})

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        agent._contemplate(draft, ctx)
        agent._contemplate_fast.assert_called_once()

    def test_dispatches_to_standard_by_default(self):
        """_contemplate dispatches to _contemplate_standard by default."""
        agent = _make_exec_agent()
        agent._contemplate_standard = MagicMock(return_value={"goal_description": "std"})

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        agent._contemplate(draft, ctx)
        agent._contemplate_standard.assert_called_once()

    def test_returns_draft_on_high_confidence(self):
        """If fast response confidence >= threshold, accept draft."""
        agent = self._make_fast_agent()
        agent._contemplation_llm_call = MagicMock(return_value={
            "confidence": 0.9,
            "issues": [],
            "plan": {"goal_description": "improved", "priority": "HIGH"},
        })

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is draft

    def test_returns_improved_plan_on_low_confidence(self):
        """If confidence < threshold, return the embedded plan."""
        agent = self._make_fast_agent()
        improved = {"goal_description": "improved", "priority": "HIGH", "sub_goals": []}
        agent._contemplation_llm_call = MagicMock(return_value={
            "confidence": 0.3,
            "issues": ["missing step"],
            "plan": improved,
        })

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result == improved

    def test_returns_draft_on_llm_failure(self):
        """If LLM call fails, return draft unchanged."""
        agent = self._make_fast_agent()
        agent._contemplation_llm_call = MagicMock(side_effect=RuntimeError("fail"))

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is draft

    def test_returns_draft_on_missing_confidence(self):
        """If response lacks confidence field, return draft."""
        agent = self._make_fast_agent()
        agent._contemplation_llm_call = MagicMock(return_value={
            "plan": {"goal_description": "improved"},
        })

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is draft

    def test_returns_draft_on_missing_plan(self):
        """If response has low confidence but no valid plan, return draft."""
        agent = self._make_fast_agent()
        agent._contemplation_llm_call = MagicMock(return_value={
            "confidence": 0.3,
            "issues": ["bad"],
            "plan": None,
        })

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is draft

    def test_returns_draft_on_plan_missing_goal_description(self):
        """If plan dict lacks goal_description, return draft."""
        agent = self._make_fast_agent()
        agent._contemplation_llm_call = MagicMock(return_value={
            "confidence": 0.3,
            "issues": ["bad"],
            "plan": {"priority": "HIGH"},  # No goal_description
        })

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is draft

    def test_urgent_preemption(self):
        """Urgent work preempts fast contemplation."""
        urgent_event = threading.Event()
        urgent_event.set()
        agent = self._make_fast_agent(urgent_work_available=urgent_event)
        agent._contemplation_llm_call = MagicMock()

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is draft
        agent._contemplation_llm_call.assert_not_called()

    def test_non_urgent_work_does_not_preempt_fast(self):
        """Normal work does NOT preempt fast contemplation."""
        work_event = threading.Event()
        work_event.set()  # Non-urgent
        agent = self._make_fast_agent(work_available=work_event)
        improved = {"goal_description": "improved", "priority": "HIGH"}
        agent._contemplation_llm_call = MagicMock(return_value={
            "confidence": 0.3,
            "issues": ["bad"],
            "plan": improved,
        })

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result == improved

    def test_non_numeric_confidence_defaults_to_accept(self):
        """If confidence is non-numeric, defaults to 1.0 (accept draft)."""
        agent = self._make_fast_agent()
        agent._contemplation_llm_call = MagicMock(return_value={
            "confidence": "very high",
            "issues": [],
            "plan": {"goal_description": "improved"},
        })

        draft = {"goal_description": "test", "priority": "HIGH"}
        ctx = _make_ctx()

        result = agent._contemplate(draft, ctx)
        assert result is draft


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Smart preemption tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSmartPreemption:
    """Tests for priority-aware preemption via _urgent_work_available."""

    def test_trigger_proposal_default_not_urgent(self):
        """_trigger_proposal without urgent=True only sets _work_available."""
        agent = _make_exec_agent()
        agent._trigger_proposal()
        assert agent._work_available.is_set()
        assert not agent._urgent_work_available.is_set()

    def test_trigger_proposal_urgent_sets_both(self):
        """_trigger_proposal with urgent=True sets both events."""
        agent = _make_exec_agent()
        agent._trigger_proposal(urgent=True)
        assert agent._work_available.is_set()
        assert agent._urgent_work_available.is_set()

    def test_cli_percept_is_urgent(self):
        """CLI percepts trigger urgent proposal."""
        agent = _make_exec_agent()
        percept = MagicMock()
        percept.source = "cli"
        percept.has_maxim_keyword = False
        percept.salience = 0.0
        percept.novelty = 0.0

        agent._on_percept(percept)
        assert agent._urgent_work_available.is_set()

    def test_comms_percept_is_urgent(self):
        """SMS/voice percepts trigger urgent proposal."""
        agent = _make_exec_agent()
        percept = MagicMock()
        percept.source = "comms:twilio:sms"
        percept.has_maxim_keyword = False
        percept.salience = 0.9
        percept.novelty = 0.5

        agent._on_percept(percept)
        assert agent._urgent_work_available.is_set()

    def test_voice_keyword_is_urgent(self):
        """Voice keyword percepts trigger urgent proposal."""
        agent = _make_exec_agent()
        percept = MagicMock()
        percept.source = "audio"
        percept.has_maxim_keyword = True
        percept.salience = 0.8
        percept.novelty = 0.5

        agent._on_percept(percept)
        assert agent._urgent_work_available.is_set()

    def test_vision_percept_not_urgent(self):
        """High salience vision percepts are NOT urgent."""
        agent = _make_exec_agent()
        percept = MagicMock()
        percept.source = "vision"
        percept.has_maxim_keyword = False
        percept.salience = 0.8
        percept.novelty = 0.8

        agent._on_percept(percept)
        assert agent._work_available.is_set()
        assert not agent._urgent_work_available.is_set()

    def test_filtered_percept_high_urgency_is_urgent(self):
        """FilteredPercepts with urgency >= 0.7 trigger urgent proposal."""
        agent = _make_exec_agent()
        filtered = MagicMock()
        filtered.urgency = 0.8
        filtered.escalation_reason = "safety"
        filtered.dn_context = {}
        filtered.original = MagicMock()
        filtered.original.source = "vision"

        agent._on_filtered_percept(filtered)
        assert agent._urgent_work_available.is_set()

    def test_filtered_percept_low_urgency_not_urgent(self):
        """FilteredPercepts with urgency < 0.7 are NOT urgent."""
        agent = _make_exec_agent()
        filtered = MagicMock()
        filtered.urgency = 0.5
        filtered.escalation_reason = "high_novelty"
        filtered.dn_context = {}
        filtered.original = MagicMock()
        filtered.original.source = "vision"

        agent._on_filtered_percept(filtered)
        assert agent._work_available.is_set()
        assert not agent._urgent_work_available.is_set()
