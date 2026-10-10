"""#1133: a confirmed tool action that SUCCEEDS was reported as failed (unit level).

``LoopController.handle_confirmation`` called ``display_action(pc.tool_name, pc.params or {})``
after a successful execute; ``PendingConfirmation`` has no ``params`` (the params live in
``pc.action["params"]``, the shape ``agent_loop`` stores as ``confirmation_data``). The
AttributeError landed in the broad ``except Exception``, which logged a false
"Confirmed action failed", showed "Action failed", skipped the outcome record, and left
``confirmed_result_str`` None so no follow-up LLM cycle was queued. Since d6b64e9f
(2026-04-09). Reach: interactive confirmations AND sims (``sim.resolve_confirmation``
auto-answers through the same handler).

These drive the real ``handle_confirmation`` and the real ``execute_and_learn`` it now calls (bound by
``loop_setup._bind_execute_and_learn`` to this controller's handles); only the collaborators are faked
and the run's recorder is a spy. The real-loop gates are ``test_confirmed_path_learns_1133.py``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from maxim.agents.llm_types import LLMProposal
from maxim.runtime.loop_controller import LoopController
from maxim.runtime.loop_types import PendingConfirmation
from maxim.simulation.response_policy import PolicyType, ResponsePolicy

# A tool with a followup type ("process"), so a successful confirm must queue a follow-up.
_TOOL = "read_file"
_PARAMS = {"path": ".maxim_workspace/notes.txt"}
_CLUSTERS = {"interoception": "c_intero_2", "world": "c_world_5"}


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
            triggering_input="show me the notes",
            cluster_id=_CLUSTERS["interoception"],
            clusters=dict(_CLUSTERS),
        )
    )


def _controller() -> LoopController:
    state = MagicMock()
    state.data = {"mode": "live"}
    from maxim.runtime.executor import Executor

    # spec=Executor, and its operational mode said (#963: loop_state.operational_mode refuses a non-str answer).
    executor = MagicMock(spec=Executor)
    executor.effective_operational_mode.return_value = "live"
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


_HUB = "hub_unit_1133"


def _run(ctrl: LoopController, rec: MagicMock) -> Any:
    """The run's two callables (``LoopRun.execute_and_learn`` / ``LoopRun.book_refusal``), bound by the same
    helpers ``build_loop_run`` uses, with the run's recorder replaced by ``rec``."""
    from maxim.runtime.loop_setup import _bind_book_refusal, _bind_execute_and_learn
    from maxim.runtime.sim_adapter import NullSimulationAdapter

    return SimpleNamespace(
        execute_and_learn=_bind_execute_and_learn(
            ctrl,
            sim=NullSimulationAdapter(),
            agent_name=ctrl.agent_name,
            agent_id=_HUB,
            result_cache=MagicMock(),
            rec_outcome=rec,
            nac=None,
            memory_hub_enabled=False,
        ),
        book_refusal=_bind_book_refusal(ctrl, agent_id=_HUB, rec_outcome=rec, nac=None),
    )


def _confirm(ctrl: LoopController, answer: str = "yes"):
    """Run handle_confirmation with display + outcome sinks spied."""
    rec = MagicMock()
    with (
        patch("maxim.simulation.sim_logger.display_action") as d_action,
        patch("maxim.simulation.sim_logger.display_status") as d_status,
    ):
        run = _run(ctrl, rec)
        consumed = ctrl.handle_confirmation(
            answer, execute_and_learn=run.execute_and_learn, book_refusal=run.book_refusal, observation={}
        )
    return consumed, d_action, d_status, rec


def _status_texts(d_status: MagicMock) -> list[str]:
    return [str(c.args[0]) for c in d_status.call_args_list]


class TestConfirmedSuccessIsReportedAsSuccess:
    def test_result_is_displayed_not_action_failed(self):
        ctrl = _controller()
        consumed, d_action, d_status, _ = _confirm(ctrl)
        assert consumed is True
        d_action.assert_called_once_with(_TOOL, _PARAMS)
        texts = _status_texts(d_status)
        assert not any(t.startswith("Action failed") for t in texts), texts
        assert "Result: file contents here" in texts

    def test_outcome_recorded_with_success(self):
        ctrl = _controller()
        _, _, _, rec = _confirm(ctrl)
        rec.assert_called_once()
        kwargs = rec.call_args.kwargs
        assert kwargs["tool_name"] == _TOOL
        assert kwargs["agent_id"] == _HUB  # the hub's id, not the controller's agent_name
        assert kwargs["success"] is True
        assert kwargs["result_summary"] == "file contents here"

    def test_followup_is_queued(self):
        ctrl = _controller()
        _confirm(ctrl)
        fu = ctrl.pending_action_followup
        assert fu is not None
        assert fu.tool == _TOOL
        assert fu.result == "file contents here"
        assert fu.followup_type == "process"

    def test_no_false_failure_error_log(self, caplog):
        ctrl = _controller()
        with caplog.at_level(logging.ERROR, logger="maxim.runtime.loop_controller"):
            _confirm(ctrl)
        assert not any("Confirmed action failed" in r.getMessage() for r in caplog.records)


class TestSimAutoApprovedConfirmation:
    """The sim path: ResponsePolicy answers, agent_loop injects the answer as cli input."""

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
    """Control: a failed execute is still a failure, recorded success=False. ``read_file`` is a "process"
    tool, so under the main path's follow-up rule (owner decision D3) its error is followed up."""

    def test_failed_execute(self):
        ctrl = _controller()
        ctrl.executor.execute.return_value = MagicMock(success=False, error="boom", output=None)
        _, d_action, d_status, rec = _confirm(ctrl)
        d_action.assert_not_called()
        assert "Action failed: boom" in _status_texts(d_status)
        assert rec.call_args.kwargs["success"] is False
        fu = ctrl.pending_action_followup
        assert (fu.tool, fu.result, fu.followup_type) == (_TOOL, "[ERROR: boom]", "process")


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
        rec = MagicMock()
        ctrl.handle_plan_approval("no", book_refusal=_run(ctrl, rec).book_refusal)
        rec.assert_called_once()
        return rec.call_args.kwargs["tool_name"]

    def test_action_without_tool_name_records_unknown(self):
        assert self._reject({"params": {"x": 1}}) == "unknown"

    def test_action_with_null_tool_name_records_unknown(self):
        assert self._reject({"tool_name": None, "params": {}}) == "unknown"

    def test_named_action_records_its_name(self):
        assert self._reject({"tool_name": "write_file", "params": {}}) == "write_file"

    def test_missing_action_records_unknown(self):
        assert self._reject(None) == "unknown"


class TestReviewPins:
    """Three-lens review folds (#1133): each pins a mutant that survived the first round."""

    def test_a_refused_confirmation_books_under_the_hub_with_its_proposal_situation(self):
        ctrl = _controller()
        _, _, _, rec = _confirm(ctrl, "no")
        kw = rec.call_args.kwargs
        assert (kw["agent_id"], kw["success"], kw["error"]) == (_HUB, False, "User rejected this action")
        assert (kw["cluster_id"], kw["clusters"]) == (_CLUSTERS["interoception"], _CLUSTERS)

    def test_a_refused_plan_books_under_the_hub_with_its_proposal_situation(self):
        ctrl = _controller()
        ctrl.pending_confirmation = None
        ctrl.pending_plan_proposal = LLMProposal(
            request_id="r-plan",
            action={"tool_name": "write_file", "params": {}},
            reasoning="plan",
            strategy_used=None,
            confidence=0.5,
            mode_goal_achieved=False,
            plan_text="do it",
            requires_approval=True,
            cluster_id=_CLUSTERS["interoception"],
            clusters=dict(_CLUSTERS),
        )
        rec = MagicMock()
        ctrl.handle_plan_approval("no", book_refusal=_run(ctrl, rec).book_refusal)
        kw = rec.call_args.kwargs
        assert (kw["agent_id"], kw["cluster_id"], kw["clusters"]) == (_HUB, _CLUSTERS["interoception"], _CLUSTERS)

    def test_a_dispatch_that_raises_still_clears_the_confirmation_and_propagates(self):
        """Same crash surface as the main path: the raise is not swallowed, and the bookkeeping ran."""
        ctrl = _controller()
        ctrl.pending_proposal = LLMProposal(
            request_id="newer",
            action={"tool_name": "x"},
            reasoning="",
            strategy_used=None,
            confidence=0.5,
            mode_goal_achieved=False,
        )
        ctrl.state.data["pending_cli_input"] = "yes"

        def _boom(**_kw):
            raise RuntimeError("capture_before broke")

        with pytest.raises(RuntimeError, match="capture_before broke"):
            ctrl.handle_confirmation("yes", execute_and_learn=_boom, book_refusal=MagicMock(), observation={})
        assert ctrl.pending_confirmation is None
        assert ctrl.pending_proposal is None
        assert "pending_cli_input" not in ctrl.state.data

    def test_a_confirmed_action_is_audited_as_human_involved(self):
        ctrl = _controller()
        _confirm(ctrl)
        executed = [
            c.kwargs
            for c in ctrl.autonomy_controller.log_action.call_args_list
            if c.kwargs.get("action_type") == "executed"
        ]
        assert [kw["human_involved"] for kw in executed] == [True]

    @pytest.mark.parametrize("action", [{"params": {}}, {"tool_name": None, "params": {}}, {"tool_name": ""}])
    def test_the_record_names_a_missing_tool_unknown(self, action):
        pc = PendingConfirmation.from_proposal(
            LLMProposal(
                request_id="r",
                action=action,
                reasoning="",
                strategy_used=None,
                confidence=0.5,
                mode_goal_achieved=False,
            )
        )
        assert pc.tool_name == "unknown"
        assert pc.policy_view()["tool_name"] == "unknown"

    @pytest.mark.parametrize("params", ["not a dict", None, ["a"]])
    def test_non_dict_params_read_as_empty(self, params):
        pc = PendingConfirmation.from_proposal(
            LLMProposal(
                request_id="r",
                action={"tool_name": "t", "params": params},
                reasoning="",
                strategy_used=None,
                confidence=0.5,
                mode_goal_achieved=False,
            )
        )
        assert pc.params == {}

    def test_a_raise_after_a_success_is_not_reported_as_success(self):
        """``ExecutionOutcome.success`` is False when the dispatch raised, even after the tool succeeded."""
        from maxim.memory.encoding import EncodingContractError

        ctrl = _controller()

        class _Hippo:
            def capture_from_loop_async(self, **_kw):
                raise EncodingContractError("capture contract broken")

            def observe_episode_event(self, _e):
                return None

        ctrl.hippocampus = _Hippo()
        rec = MagicMock()
        outcome = _run(ctrl, rec).execute_and_learn(
            action=ctrl.pending_confirmation.action,
            confidence=0.9,
            proposal=ctrl.pending_confirmation.source,
            observation={},
            human_involved=False,
        )
        assert (outcome.success, outcome.raised, outcome.error) == (False, True, "capture contract broken")
        assert [c.kwargs["success"] for c in rec.call_args_list] == [True, False]  # #1145, as pinned in C1
