"""A rejected tool call in Console Talk (#1413) and a consumed input that is never cleared (#1414).

Both were found by maxim-pulse live on the 1.3.3 candidate: Talk chose `bash`, autonomy refused it, and then:
- #1413: nothing scheduled a second LLM call, so the turn idled to its 180 s timeout and the chat said only
  "(no reply — the turn timed out before finishing)";
- #1414: the user's `cli_input` stayed in `state.data`, so `propose_intent` re-published it every 0.5 s tick and the
  Hippocampus re-captured it as new memory (about 87 times in 44 s).

Owner decisions 2026-10-22: a rejection becomes a capped follow-up (at most 2 per user turn, then a blocked record
that ends the turn with a reply naming the tool), in every loop; an input is consumed once, with no content dedupe
(a user saying the same thing twice is two observations).
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

# ── #1413: a hard rejection is fed back to the model ───────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1413: the hard-rejection branch never schedules a follow-up")
def test_a_hard_rejection_leaves_a_follow_up_that_names_the_tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Any):
    """The REAL loop, one proposal of a hard-forbidden tool: the model must get a next call, told what was refused
    and that it did not run, so it can re-plan or reply."""
    from tests.unit._execute_learn_driver import run_once

    tool = "probe_forbidden_1413"
    obs = run_once(monkeypatch, tmp_path, tool=tool, level="supervised", policy={"forbidden_tools": {tool}})
    assert obs.executed == []  # refused: the tool never ran
    obs.note_followup()
    assert obs.followups, "a hard rejection left no follow-up: no second LLM call, the turn idles to its timeout"
    followup = obs.followups[-1]
    assert followup.tool == tool and followup.followup_type == "process"
    assert "Rejected by autonomy" in (followup.result or "") and "did not run" in (followup.result or "")


# ── #1414: an input is consumed once ──────────────────────────────────────────────────────────────────────────


def _run_with_one_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, *, ticks: int = 8) -> list[Any]:
    """The REAL loop, no LLM worker: the environment yields ONE cli_input, then empty observations (a drained queue,
    as `runtime/sim_adapter.py` returns). Returns what `state.data["cli_input"]` held at each `propose_intent`."""
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime import substrate_proposal
    from maxim.runtime.bootstrap import build_decision_engine, build_executor, build_memory
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.state import RuntimeState
    from maxim.simulation import sim_logger
    from maxim.tools.registry import ToolRegistry

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim_logger, "_interactive_mode", sim_logger.InteractiveMode.OFF)
    monkeypatch.setattr(substrate_proposal, "propose_via_substrate", lambda **_kw: None)
    real_init = LoopController.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        self.llm_submit_interval = 0.0  # wake every pass, so §3 (propose_intent) runs every pass

    monkeypatch.setattr(LoopController, "__init__", _init)
    (tmp_path / "ws").mkdir(exist_ok=True)

    class _OneInputEnv(FileSystemEnv):
        def __init__(self, root: str) -> None:
            super().__init__(root)
            self._sent = False

        def observe(self) -> dict[str, Any]:
            if self._sent:
                return {}
            self._sent = True
            return {"cli_input": "what is the news today"}

    seen: list[Any] = []

    class _Agent:
        name = "consume-once-probe-1414"

        def propose_intent(self, state: Any, memory: Any) -> None:
            seen.append(state.data.get("cli_input"))
            return None

    state = RuntimeState()
    state.data["mode"] = "active"
    AL.run_agentic_loop(
        _Agent(),
        _OneInputEnv(str(tmp_path / "ws")),
        state,
        build_memory(),
        build_decision_engine(),
        build_executor(ToolRegistry(), pain_bus=None, permissions=None),
        autonomy_controller=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS),
        aut_mode="substrate-primary",
        max_steps=ticks,
        target_hz=200.0,
        idle_sleep_s=0.0,
        stop_event=threading.Event(),
    )
    return seen


@pytest.mark.xfail(strict=True, reason="#1414: a consumed cli_input stays in state.data and is re-read every tick")
def test_a_consumed_input_is_not_seen_again_on_later_ticks(monkeypatch: pytest.MonkeyPatch, tmp_path: Any):
    seen = _run_with_one_input(monkeypatch, tmp_path)
    assert len(seen) >= 3, seen  # the probe ran on several ticks
    assert seen[0] == "what is the news today"
    assert all(v is None for v in seen[1:]), f"the input was re-read on later ticks: {seen}"


# ── The chat names a refused tool ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1413: a blocked tool reads as 'failed' in the chat")
def test_a_turn_ended_by_a_refused_tool_says_the_tool_is_not_allowed():
    from types import SimpleNamespace

    from maxim.console.handle import _describe_silent_turn

    blocked = SimpleNamespace(tool_name="bash", blocked=True, result_success=False, utterance=None)
    text = _describe_silent_turn({"timed_out": False, "actions": [blocked]})
    assert "bash" in text and "not allowed" in text, text
