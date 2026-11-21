"""#1185: a change request at a confirmation or plan prompt was stranded.

The answer was stored as ``state.data["pending_modification"]`` (the dropped action plus the person's words, the
shape ``prompt_builder.build_modification_section`` renders), and nothing woke the loop or submitted it: in LLM mode
it waited for the next unrelated input; in substrate-primary mode nothing ever reads it. The plan prompt also sent
the words to the model twice (as the turn and as the modification). Now:

- LLM mode: ``LoopController.settle_pending_modification`` wakes the loop, ``loop_perception.context_addresses_the_llm``
  makes it a meaningful input, and the §6 submit carries it exactly once, as ``pending_modification`` only;
- substrate-primary: the person is told this mode cannot apply a change (``modification_unsupported``), and it is
  cleared, so it never idle-wakes the loop.

Both driven through the REAL ``run_agentic_loop``.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from tests.policy_helpers import TEST_POLICY
from tests.unit._execute_learn_driver import run_once

CHANGE = "write the file to the other folder"
pytestmark = pytest.mark.timeout(60)


def _llm_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, *, ticks: int = 30) -> tuple[list[dict[str, Any]], list]:
    """The real loop in LLM mode with a fake worker: the first submit returns a plan that requires approval; the
    person answers it with a change request; every later submit is recorded."""
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.agents.llm_types import LLMProposal
    from maxim.agents.perception_agent import PerceptionAgent
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime.bootstrap import build_decision_engine, build_executor, build_memory
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.state import RuntimeState
    from maxim.simulation import sim_logger
    from maxim.tools.registry import ToolRegistry
    from maxim.utils import structured_logging

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_logger, "_interactive_mode", sim_logger.InteractiveMode.ON)  # a person answers
    real_init = LoopController.__init__

    ctrls: list[Any] = []

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        self.llm_submit_interval = 0.0
        ctrls.append(self)

    monkeypatch.setattr(LoopController, "__init__", _init)
    events: list[tuple[str, dict[str, Any]]] = []
    buffer = structured_logging.get_abstraction_buffer()
    real_append = buffer.append

    def _append(record: Any) -> Any:
        if getattr(record, "source", None) == "agent_loop":
            events.append((record.event, dict(record.data or {})))
            if record.event == "plan_awaiting_approval" and not (record.data or {}).get("reask"):
                state.data["pending_cli_input"] = CHANGE  # the person answers the shown plan with a change
                for ctrl in ctrls:  # ... minutes later, as people do: the "submitted recently" wake has lapsed
                    ctrl.last_llm_submit_time = 0.0
        return real_append(record)

    monkeypatch.setattr(buffer, "append", _append)
    (tmp_path / "ws").mkdir(exist_ok=True)
    state = RuntimeState()
    state.data["mode"] = "active"
    clock = {"tick": -1}
    submits: list[dict[str, Any]] = []
    plans = ["bash"]

    class _Env(FileSystemEnv):
        def observe(self) -> dict[str, Any]:
            clock["tick"] += 1
            if clock["tick"] == 0:
                return {"cli_input": "clean the build"}
            return {}

    class _Worker:
        def latest_attempt_state(self) -> None:
            return None

        def get_latest_proposal(self) -> Any:
            if not submits or not plans:
                return None
            tool = plans.pop(0)
            for key in ("pending_user_input", "pending_user_input_time", "pending_user_input_source"):
                state.data.pop(key, None)  # the LLM answered the turn
            return LLMProposal(
                request_id="p-1185-mod",
                action={"tool_name": tool, "params": {"command": "rm -rf build"}},
                reasoning="clean it",
                strategy_used="llm",
                confidence=0.9,
                mode_goal_achieved=False,
                triggering_input="clean the build",
                plan_text="delete the build dir",
                requires_approval=True,
            )

        def submit_context(self, **kw: Any) -> bool:
            submits.append(kw)
            return True

        def requeue_request(self, *_a: Any, **_k: Any) -> bool:
            return True

        def requeue_last_request(self, **_k: Any) -> bool:
            return True

        def record_outcome(self, *_a: Any, **_k: Any) -> None:
            return None

    class _Bus:
        def publish(self, item: Any) -> None:
            return None

        def subscribe(self, *_a: Any, **_k: Any) -> None:
            return None

        def unsubscribe(self, *_a: Any, **_k: Any) -> None:
            return None

    AL.run_agentic_loop(
        PerceptionAgent(bus=_Bus()),
        _Env(str(tmp_path / "ws")),
        state,
        build_memory(),
        build_decision_engine(),
        build_executor(ToolRegistry(), pain_bus=None, policy=TEST_POLICY),
        autonomy_controller=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS),
        llm_worker=_Worker(),
        max_steps=ticks,
        target_hz=200.0,
        idle_sleep_s=0.0,
        stop_event=threading.Event(),
    )
    return submits, events


def _inputs_of(submit: dict[str, Any]) -> list[str]:
    return [str(i) for i in (getattr(submit["context"], "cli_inputs", None) or [])]


def test_in_llm_mode_a_change_request_is_submitted_once_with_the_original_action(monkeypatch, tmp_path):
    submits, events = _llm_loop(monkeypatch, tmp_path)
    assert [e for e, _ in events].count("plan_modify_requested") == 1
    carrying = [s for s in submits if s.get("pending_modification")]
    assert len(carrying) == 1, [s.get("triggering_input") for s in submits]
    [mod_submit] = carrying
    mod = mod_submit["pending_modification"]
    assert mod["user_modification"] == CHANGE
    assert mod["original_tool_name"] == "bash"
    assert mod["original_action"]["params"] == {"command": "rm -rf build"}
    assert mod_submit["triggering_input"] == CHANGE
    # Once: never also delivered as the person's turn (fold 4), in that submit or any other.
    assert all(CHANGE not in _inputs_of(s) for s in submits), [_inputs_of(s) for s in submits]


def test_in_substrate_mode_a_change_request_is_told_and_cleared(monkeypatch, tmp_path):
    obs = run_once(
        monkeypatch, tmp_path, plan_text="delete the build dir", answers=[CHANGE], max_steps=10, interactive=True
    )
    told = [d for e, d in obs.events if e == "modification_unsupported"]
    assert len(told) == 1 and told[0]["modification"] == CHANGE
    assert obs.executed == []
    # Cleared, so nothing is left to wake the loop for (substrate-primary's own tick cadence still runs).
    assert obs.ctrl.state.data.get("pending_modification") is None
