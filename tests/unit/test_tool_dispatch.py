"""Tests for runtime/tool_dispatch.py — extracted from agent_loop.py."""

from __future__ import annotations

from unittest.mock import MagicMock
from maxim.runtime.tool_dispatch import safe_agent_name, record_outcome


class TestSafeAgentName:
    """Test safe_agent_name extraction and sanitization."""

    def test_uses_state_name(self):
        agent = MagicMock()
        agent.state_name = "my_agent"
        assert safe_agent_name(agent) == "my_agent"

    def test_uses_agent_name_fallback(self):
        agent = MagicMock(spec=[])
        agent.agent_name = "bot-1"
        assert safe_agent_name(agent) == "bot-1"

    def test_uses_name_fallback(self):
        agent = MagicMock(spec=[])
        agent.name = "helper"
        assert safe_agent_name(agent) == "helper"

    def test_uses_class_name_if_no_attrs(self):
        class CustomAgent:
            pass

        agent = CustomAgent()
        assert safe_agent_name(agent) == "CustomAgent"

    def test_sanitizes_special_characters(self):
        agent = MagicMock()
        agent.state_name = "my agent (v2) [test]"
        result = safe_agent_name(agent)
        assert "agent" in result
        # Special chars replaced with underscore(s)
        assert "(" not in result and "[" not in result

    def test_strips_leading_trailing_separators(self):
        agent = MagicMock()
        agent.state_name = "..._agent_..."
        assert safe_agent_name(agent) == "agent"

    def test_empty_name_returns_agent(self):
        agent = MagicMock()
        agent.state_name = ""
        agent.agent_name = ""
        agent.name = ""
        result = safe_agent_name(agent)
        # Falls through to type(agent).__name__ which is MagicMock
        assert result  # Not empty

    def test_none_attrs_uses_class(self):
        agent = MagicMock()
        agent.state_name = None
        agent.agent_name = None
        agent.name = None
        result = safe_agent_name(agent)
        assert result  # Not empty


class TestRecordOutcome:
    """Test record_outcome dispatches to all sinks correctly."""

    def _make_context_pool(self):
        pool = MagicMock()
        pool.add_outcome = MagicMock()
        return pool

    def test_appends_to_recent_outcomes(self):
        outcomes: list = []
        pool = self._make_context_pool()
        record_outcome(
            agent_id="test",
            tool_name="look",
            success=True,
            result_summary="saw a cup",
            error=None,
            reasoning="exploring",
            recent_outcomes=outcomes,
            max_recent=10,
            llm_worker=None,
            context_pool=pool,
        )
        assert len(outcomes) == 1
        assert outcomes[0]["tool"] == "look"
        assert outcomes[0]["success"] is True

    def test_caps_recent_outcomes(self):
        outcomes: list = [{"tool": f"t{i}"} for i in range(5)]
        pool = self._make_context_pool()
        record_outcome(
            agent_id="test",
            tool_name="new_tool",
            success=True,
            result_summary=None,
            error=None,
            reasoning="",
            recent_outcomes=outcomes,
            max_recent=5,
            llm_worker=None,
            context_pool=pool,
        )
        assert len(outcomes) == 5
        assert outcomes[-1]["tool"] == "new_tool"

    def test_records_to_llm_worker(self):
        worker = MagicMock()
        pool = self._make_context_pool()
        record_outcome(
            agent_id="test",
            tool_name="say",
            success=True,
            result_summary="hello",
            error=None,
            reasoning="greeting",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=worker,
            context_pool=pool,
        )
        worker.record_outcome.assert_called_once()
        call_kwargs = worker.record_outcome.call_args[1]
        assert call_kwargs["tool_name"] == "say"
        assert call_kwargs["success"] is True

    def test_records_to_context_pool(self):
        pool = self._make_context_pool()
        record_outcome(
            agent_id="test",
            tool_name="examine",
            success=False,
            result_summary=None,
            error="not found",
            reasoning="looking",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=pool,
        )
        pool.add_outcome.assert_called_once_with(
            tool_name="examine",
            success=False,
            result_summary=None,
            error="not found",
        )

    def test_nac_observation_on_success(self):
        nac = MagicMock()
        pool = self._make_context_pool()
        record_outcome(
            agent_id="test",
            tool_name="move",
            success=True,
            result_summary="moved forward",
            error=None,
            reasoning="navigate",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=pool,
            nac=nac,
            elapsed_s=0.5,
        )
        # Should call nac.observe at least once (tool result)
        assert nac.observe.called

    def test_nac_energy_observation_for_slow_tools(self):
        nac = MagicMock()
        pool = self._make_context_pool()
        record_outcome(
            agent_id="test",
            tool_name="heavy_compute",
            success=True,
            result_summary="done",
            error=None,
            reasoning="computing",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=pool,
            nac=nac,
            elapsed_s=5.0,  # >2s = expensive
        )
        # Should call nac.observe twice: tool result + energy cost
        assert nac.observe.call_count >= 2

    def test_nac_none_skips_observation(self):
        pool = self._make_context_pool()
        # Should not raise even with nac=None
        record_outcome(
            agent_id="test",
            tool_name="look",
            success=True,
            result_summary="ok",
            error=None,
            reasoning="",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=pool,
            nac=None,
        )

    def test_nac_exception_does_not_propagate(self):
        nac = MagicMock()
        nac.observe.side_effect = RuntimeError("NAc broken")
        pool = self._make_context_pool()
        # Should not raise — NAc errors are logged, not propagated
        record_outcome(
            agent_id="test",
            tool_name="look",
            success=True,
            result_summary="ok",
            error=None,
            reasoning="test",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=pool,
            nac=nac,
        )


class TestExecuteParallelActions:
    """Test parallel action batch execution."""

    def _make_context_pool(self):
        pool = MagicMock()
        pool.add_outcome = MagicMock()
        return pool

    def _make_autonomy(self, allow=True):
        ac = MagicMock()
        ac.can_execute_action.return_value = (allow, "" if allow else "rejected")
        return ac

    def _make_executor(self, success=True):
        ex = MagicMock()
        result = MagicMock()
        result.success = success
        result.output = "ok" if success else None
        result.error = None if success else "failed"
        ex.execute.return_value = result
        return ex

    def test_executes_all_actions(self):
        from maxim.runtime.tool_dispatch import execute_parallel_actions

        actions = [
            {"tool_name": "look", "params": {"target": "room"}},
            {"tool_name": "examine", "params": {"target": "door"}},
        ]
        results, combined = execute_parallel_actions(
            agent_id="test",
            actions=actions,
            executor=self._make_executor(),
            autonomy_controller=self._make_autonomy(),
            confidence=0.9,
            reasoning="exploring",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=self._make_context_pool(),
        )
        assert len(results) == 2
        assert all(r["success"] for r in results)
        assert "BATCHED EXPLORATION RESULTS" in combined

    def test_rejected_action_recorded(self):
        from maxim.runtime.tool_dispatch import execute_parallel_actions

        actions = [{"tool_name": "delete", "params": {}}]
        results, _ = execute_parallel_actions(
            agent_id="test",
            actions=actions,
            executor=self._make_executor(),
            autonomy_controller=self._make_autonomy(allow=False),
            confidence=0.5,
            reasoning="cleanup",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=self._make_context_pool(),
        )
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "Rejected" in results[0]["error"]

    def test_execution_failure_captured(self):
        from maxim.runtime.tool_dispatch import execute_parallel_actions

        ex = MagicMock()
        ex.execute.side_effect = RuntimeError("crash")
        actions = [{"tool_name": "boom", "params": {}}]
        results, combined = execute_parallel_actions(
            agent_id="test",
            actions=actions,
            executor=ex,
            autonomy_controller=self._make_autonomy(),
            confidence=0.9,
            reasoning="test",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=self._make_context_pool(),
        )
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "FAILED" in combined

    def test_records_outcomes_for_each_action(self):
        from maxim.runtime.tool_dispatch import execute_parallel_actions

        outcomes: list = []
        actions = [
            {"tool_name": "a", "params": {}},
            {"tool_name": "b", "params": {}},
        ]
        execute_parallel_actions(
            agent_id="test",
            actions=actions,
            executor=self._make_executor(),
            autonomy_controller=self._make_autonomy(),
            confidence=0.9,
            reasoning="test",
            recent_outcomes=outcomes,
            max_recent=10,
            llm_worker=None,
            context_pool=self._make_context_pool(),
        )
        assert len(outcomes) == 2


class TestAgentIdAttribution:
    """P4: agent_id is required + tagged on every NAc observation."""

    def _make_context_pool(self):
        pool = MagicMock()
        pool.add_outcome = MagicMock()
        return pool

    def test_record_outcome_requires_agent_id(self):
        pool = self._make_context_pool()
        import pytest

        with pytest.raises(TypeError, match="agent_id"):
            record_outcome(  # type: ignore[call-arg]
                tool_name="look",
                success=True,
                result_summary="ok",
                error=None,
                reasoning="",
                recent_outcomes=[],
                max_recent=10,
                llm_worker=None,
                context_pool=pool,
            )

    def test_nac_observation_tagged_with_agent_id(self):
        nac = MagicMock()
        pool = self._make_context_pool()
        record_outcome(
            agent_id="alpha",
            tool_name="move",
            success=True,
            result_summary="moved",
            error=None,
            reasoning="navigate",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=pool,
            nac=nac,
            elapsed_s=0.5,
        )
        # First call is the tool result NAc.observe
        first_kwargs = nac.observe.call_args_list[0].kwargs
        assert first_kwargs["context"]["agent_id"] == "alpha"

    def test_two_agents_isolated_in_nac_context(self):
        nac = MagicMock()
        pool = self._make_context_pool()
        record_outcome(
            agent_id="alpha",
            tool_name="t",
            success=True,
            result_summary="ok",
            error=None,
            reasoning="",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=pool,
            nac=nac,
        )
        record_outcome(
            agent_id="bravo",
            tool_name="t",
            success=True,
            result_summary="ok",
            error=None,
            reasoning="",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=pool,
            nac=nac,
        )
        agent_ids = [c.kwargs["context"]["agent_id"] for c in nac.observe.call_args_list]
        assert "alpha" in agent_ids
        assert "bravo" in agent_ids


class TestExecuteParallelActiveGoal:
    """Pre-existing-bug regression: active_goal must reach record_outcome."""

    def _make_context_pool(self):
        pool = MagicMock()
        pool.add_outcome = MagicMock()
        return pool

    def _make_autonomy(self, allow=True):
        ac = MagicMock()
        ac.can_execute_action.return_value = (allow, "" if allow else "rejected")
        return ac

    def _make_executor(self, success=True):
        ex = MagicMock()
        result = MagicMock()
        result.success = success
        result.output = "ok"
        result.error = None
        ex.execute.return_value = result
        return ex

    def test_active_goal_propagates_to_per_action_credit(self):
        # Pre-fix: passing active_goal= to execute_parallel_actions raised
        # TypeError because the parameter was missing from the signature.
        from maxim.runtime.tool_dispatch import execute_parallel_actions

        nac = MagicMock()
        execute_parallel_actions(
            agent_id="test",
            actions=[{"tool_name": "look", "params": {}}],
            executor=self._make_executor(success=True),
            autonomy_controller=self._make_autonomy(),
            confidence=0.9,
            reasoning="r",
            recent_outcomes=[],
            max_recent=10,
            llm_worker=None,
            context_pool=self._make_context_pool(),
            nac=nac,
            active_goal="explore_room",
        )
        # credit_goal is what active_goal triggers inside record_outcome.
        nac.credit_goal.assert_called_with("explore_room", 1.0)


class TestImportPaths:
    """Verify internal import paths work."""

    def test_import_record_outcome_from_agent_loop(self):
        from maxim.runtime.agent_loop import _record_outcome

        assert callable(_record_outcome)

    def test_import_safe_agent_name_from_agent_loop(self):
        from maxim.runtime.agent_loop import _safe_agent_name

        assert callable(_safe_agent_name)

    def test_loop_controller_import_still_works(self):
        from maxim.runtime.loop_controller import LoopController

        assert LoopController is not None
