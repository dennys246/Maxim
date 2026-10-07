"""Red gates for #1133: a confirmed tool action that SUCCEEDS was reported as failed.

``LoopController.handle_confirmation`` called ``display_action(pc.tool_name, pc.params or {})``
after a successful execute; ``PendingConfirmation`` has no ``params`` (the params live in
``pc.action["params"]``, the shape ``agent_loop`` stores as ``confirmation_data``). The
AttributeError landed in the broad ``except Exception``, which logged a false
"Confirmed action failed", showed "Action failed", skipped the outcome record, and left
``confirmed_result_str`` None so no follow-up LLM cycle was queued. Since d6b64e9f
(2026-04-09). Reach: interactive confirmations AND sims (``sim.resolve_confirmation``
auto-answers through the same handler).

These drive the real ``handle_confirmation``; only its collaborators are faked.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from maxim.agents.llm_types import LLMProposal
from maxim.runtime.loop_controller import LoopController
from maxim.runtime.loop_types import PendingConfirmation
from maxim.simulation.response_policy import PolicyType, ResponsePolicy

# A tool with a followup type ("process"), so a successful confirm must queue a follow-up.
_TOOL = "read_file"
_PARAMS = {"path": ".maxim_workspace/notes.txt"}


def _confirmation_data() -> dict:
    """The confirmation as ``PendingConfirmation.policy_view()`` shows it to a response policy."""
    return {
        "action": {"tool_name": _TOOL, "params": dict(_PARAMS)},
        "reasoning": "need the notes",
        "confidence": 0.9,
        "tool_name": _TOOL,
    }


def _pending() -> PendingConfirmation:
    """The typed record the loop parks (``PendingConfirmation.from_proposal``)."""
    data = _confirmation_data()
    return PendingConfirmation.from_proposal(
        LLMProposal(
            request_id="r-1133",
            action=data["action"],
            reasoning=data["reasoning"],
            strategy_used=None,
            confidence=data["confidence"],
            mode_goal_achieved=False,
        )
    )


def _controller() -> LoopController:
    state = MagicMock()
    state.data = {"mode": "live"}
    executor = MagicMock()
    executor.execute.return_value = MagicMock(success=True, error=None, output="file contents here")
    ctrl = LoopController(
        agent=MagicMock(),
        environment=MagicMock(),
        state=state,
        memory=MagicMock(),
        decision_engine=MagicMock(),
        executor=executor,
        autonomy_controller=MagicMock(),
    )
    ctrl.context_pool = MagicMock()
    ctrl.recent_outcomes = []
    ctrl.pending_confirmation = _pending()
    return ctrl


def _confirm(ctrl: LoopController, answer: str = "yes"):
    """Run handle_confirmation with display + outcome sinks spied."""
    with (
        patch("maxim.simulation.sim_logger.display_action") as d_action,
        patch("maxim.simulation.sim_logger.display_status") as d_status,
        patch("maxim.runtime.loop_controller._record_outcome") as rec,
    ):
        consumed = ctrl.handle_confirmation(answer)
    return consumed, d_action, d_status, rec


def _status_texts(d_status: MagicMock) -> list[str]:
    return [str(c.args[0]) for c in d_status.call_args_list]


_RED_1133 = pytest.mark.xfail(strict=True, reason="#1133: PendingConfirmation has no .params")


class TestConfirmedSuccessIsReportedAsSuccess:
    @_RED_1133
    def test_result_is_displayed_not_action_failed(self):
        ctrl = _controller()
        consumed, d_action, d_status, _ = _confirm(ctrl)
        assert consumed is True
        d_action.assert_called_once_with(_TOOL, _PARAMS)
        texts = _status_texts(d_status)
        assert not any(t.startswith("Action failed") for t in texts), texts
        assert "Result: file contents here" in texts

    @_RED_1133
    def test_outcome_recorded_with_success(self):
        ctrl = _controller()
        _, _, _, rec = _confirm(ctrl)
        rec.assert_called_once()
        kwargs = rec.call_args.kwargs
        assert kwargs["tool_name"] == _TOOL
        assert kwargs["success"] is True
        assert kwargs["result_summary"] == "file contents here"

    @_RED_1133
    def test_followup_is_queued(self):
        ctrl = _controller()
        _confirm(ctrl)
        fu = ctrl.pending_action_followup
        assert fu is not None
        assert fu.tool == _TOOL
        assert fu.result == "file contents here"
        assert fu.followup_type == "process"

    @_RED_1133
    def test_no_false_failure_error_log(self, caplog):
        ctrl = _controller()
        with caplog.at_level(logging.ERROR, logger="maxim.runtime.loop_controller"):
            _confirm(ctrl)
        assert not any("Confirmed action failed" in r.getMessage() for r in caplog.records)


class TestSimAutoApprovedConfirmation:
    """The sim path: ResponsePolicy answers, agent_loop injects the answer as cli input."""

    @_RED_1133
    def test_sim_auto_approve_reaches_success_and_followup(self, caplog):
        ctrl = _controller()
        sim_response = ResponsePolicy(policy=PolicyType.AUTO_APPROVE).resolve_confirmation(_confirmation_data())
        assert sim_response == "yes"
        with caplog.at_level(logging.ERROR, logger="maxim.runtime.loop_controller"):
            _, _, d_status, rec = _confirm(ctrl, sim_response)
        assert not any("Confirmed action failed" in r.getMessage() for r in caplog.records)
        assert not any(t.startswith("Action failed") for t in _status_texts(d_status))
        assert rec.call_args.kwargs["success"] is True
        assert ctrl.pending_action_followup is not None


class TestGenuineFailureStillReportsFailure:
    """Control: a failed execute is still a failure (no follow-up, recorded success=False)."""

    def test_failed_execute(self):
        ctrl = _controller()
        ctrl.executor.execute.return_value = MagicMock(success=False, error="boom", output=None)
        _, d_action, d_status, rec = _confirm(ctrl)
        d_action.assert_not_called()
        assert "Action failed: boom" in _status_texts(d_status)
        assert rec.call_args.kwargs["success"] is False
        assert ctrl.pending_action_followup is None


class TestPlanRejectToolName:
    """#1133 (mypy arg-type, loop_controller plan-reject branch): the rejected tool's name comes from a
    model-chosen action dict. A non-empty action with no (or a null) ``tool_name`` passed ``None`` into
    ``record_outcome(tool_name: str)``, recording the rejection under ``tool:None``. The branch's own
    fallback for a missing action is ``"unknown"``; a missing name gets the same."""

    def _reject(self, action):
        from maxim.agents.llm_worker import LLMProposal

        ctrl = _controller()
        ctrl.pending_confirmation = None
        ctrl.pending_plan_proposal = LLMProposal(
            request_id="r1",
            action=action,
            reasoning="plan",
            strategy_used=None,
            confidence=0.5,
            mode_goal_achieved=False,
            citations=[],
            latency_ms=0.0,
            plan_text="do the thing",
            requires_approval=True,
        )
        with patch("maxim.runtime.loop_controller._record_outcome") as rec:
            ctrl.handle_plan_approval("no")
        rec.assert_called_once()
        return rec.call_args.kwargs["tool_name"]

    @pytest.mark.xfail(strict=True, reason="#1133: plan-reject passes tool_name=None")
    def test_action_without_tool_name_records_unknown(self):
        assert self._reject({"params": {"x": 1}}) == "unknown"

    @pytest.mark.xfail(strict=True, reason="#1133: plan-reject passes tool_name=None")
    def test_action_with_null_tool_name_records_unknown(self):
        assert self._reject({"tool_name": None, "params": {}}) == "unknown"

    def test_named_action_records_its_name(self):
        assert self._reject({"tool_name": "write_file", "params": {}}) == "write_file"

    def test_missing_action_records_unknown(self):
        assert self._reject(None) == "unknown"
