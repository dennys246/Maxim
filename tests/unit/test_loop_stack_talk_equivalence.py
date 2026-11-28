"""The console talk site's loop call, characterized before and after its move onto ``bootstrap.run_loop_stack``.

``docs/plans/loop_stack_builder.md`` ("Tests the first PR carries"): the first PR migrates ONE site, the talk loop in
``console/handle.py::MaximHandle._launch_talk_loop``, and changes nothing the loop receives. This file was written
against the pre-migration code and must pass unchanged after it. It launches talk through the handle's real path with
``agent_loop.run_agentic_loop`` replaced by a recorder at its home, so the recorder sees the call whichever caller
makes it (the migrated ``run_loop_stack`` looks the loop up at call time). It pins:

- every argument the loop receives: the instance's subsystems by identity, the fresh per-run objects by type, and the
  literals talk states (``max_steps``, ``target_hz``, ``consolidation``) by value;
- which keywords talk passes at all: the loop's own defaults fill the rest, and talk must not gain a ``run_id``,
  ``state_home`` or ``evaluators``;
- the timing: ``FileSystemEnv``, ``build_memory()`` and ``build_decision_engine()`` are built on the
  ``console.talk.<id>`` thread, inside the ``try`` whose ``except`` logs through ``logger.exception``.
"""

from __future__ import annotations

import inspect
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.unit.test_agent_router_slots_1228 import _Router
from tests.unit.test_bounded_stop_handle_1536 import _handle

# The loop's keyword defaults, read from its signature: an argument talk does not pass arrives as one of these.
_DEFAULT = object()


def _fake_worker() -> Any:
    """A started worker as talk receives it: a router, and ``LLMWorker``'s own defaults for the two embodiment
    attributes (``acting_coach = None``, ``is_embodied = False``), which talk never sets."""
    return SimpleNamespace(router=_Router("talk"), acting_coach=None, is_embodied=False)


def _instance() -> Any:
    inst = MagicMock(name="instance")
    inst.bio_stack = SimpleNamespace(bio_enrichment_pipeline=object(), thought_gate=object())
    inst.memory_hub = None  # an unwired handle: talk wires no hub (the wired case is pinned below)
    return inst


def _launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inst: Any, loop: Any) -> Any:
    from maxim.runtime import agent_loop

    monkeypatch.setattr(agent_loop, "run_agentic_loop", loop)
    h = _handle()
    h.instance = inst
    worker = _fake_worker()
    with h._talk_lock:
        h._launch_talk_loop(worker, tmp_path / "ws")
    assert h._talk_thread is not None
    h._talk_thread.join(timeout=10)
    assert not h._talk_thread.is_alive()
    return h, worker


def _recorder() -> tuple[list[dict[str, Any]], Any]:
    calls: list[dict[str, Any]] = []

    def loop(*args: Any, **kwargs: Any) -> None:
        calls.append({"args": args, "kwargs": kwargs, "thread": threading.current_thread().name})

    return calls, loop


def test_the_talk_loop_receives_the_instances_subsystems_and_talks_own_literals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from maxim.agents.maxim_agent import MaximAgent
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime.state import RuntimeState
    from maxim.utils.deadline import EndBound

    inst = _instance()
    calls, loop = _recorder()
    h, worker = _launch(monkeypatch, tmp_path, inst, loop)

    [call] = calls
    args, kw = call["args"], call["kwargs"]
    # The six positionals, in the loop's order (passed by keyword after the migration: normalized below).
    pos = dict(zip(("agent", "environment", "state", "memory", "decision_engine", "executor"), args, strict=False))
    merged = {**pos, **kw}
    assert isinstance(merged["agent"], MaximAgent)
    assert merged["agent"].memory._memory_hub is None  # an unwired handle wires no hub
    assert isinstance(merged["environment"], FileSystemEnv)
    assert isinstance(merged["state"], RuntimeState)
    assert merged["state"].data["mode"] == "active"
    assert merged["state"].data["active_goal"] == "converse with the user"
    assert merged["memory"] is not None and merged["decision_engine"] is not None
    assert merged["executor"] is inst.executor
    assert merged["llm_worker"] is worker
    assert merged["hippocampus"] is inst.hippocampus
    assert merged["memory_hub"] is inst.memory_hub
    assert merged["pain_bus"] is inst.pain_bus
    assert merged["bio_enrichment_pipeline"] is inst.bio_stack.bio_enrichment_pipeline
    assert merged["thought_gate"] is inst.bio_stack.thought_gate
    assert merged["autonomy_controller"] is not None
    bridge = h._talk_bridge
    assert merged["percept_source"] is bridge.percept_source
    assert merged["action_sink"] is bridge.action_sink
    assert merged["response_policy"] is bridge.response_policy
    assert merged["stop_event"] is h._talk_stop
    assert merged["max_steps"] == 0
    assert merged["target_hz"] == 2.0
    assert merged["consolidation"] == "full"
    assert isinstance(merged["end_bound"], EndBound) and merged["end_bound"] is h._talk_bound


def test_every_argument_talk_does_not_state_is_the_loops_own_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Talk states 21 arguments; the loop's defaults fill the other 18. After the migration ``run_loop_stack``
    passes every one of the 39 explicitly, so the comparison is on VALUES: an argument outside talk's set must
    equal the loop's default (``None``, ``"llm-primary"``, the pacing numbers), so talk gains nothing, in
    particular no ``run_id``, ``state_home`` or ``evaluators``."""
    from maxim.runtime import agent_loop

    sig = inspect.signature(agent_loop.run_agentic_loop)
    calls, loop = _recorder()
    _launch(monkeypatch, tmp_path, _instance(), loop)
    [call] = calls
    positional = ("agent", "environment", "state", "memory", "decision_engine", "executor")
    merged = {**dict(zip(positional, call["args"], strict=False)), **call["kwargs"]}
    talk_states = {
        *positional,
        "autonomy_controller",
        "llm_worker",
        "hippocampus",
        "memory_hub",
        "bio_enrichment_pipeline",
        "thought_gate",
        "max_steps",
        "stop_event",
        "target_hz",
        "percept_source",
        "response_policy",
        "action_sink",
        "pain_bus",
        "consolidation",
        "end_bound",
    }
    assert talk_states <= set(merged), sorted(talk_states - set(merged))
    unknown = set(merged) - set(sig.parameters)
    assert not unknown, f"the loop was handed keywords it does not take: {sorted(unknown)}"
    for name, param in sig.parameters.items():
        if name in talk_states:
            continue
        got = merged.get(name, _DEFAULT)
        if got is _DEFAULT:
            continue  # not passed at all: the default applies
        assert got == param.default, f"talk passes {name}={got!r}, not the loop's default {param.default!r}"
    for name in ("run_id", "state_home", "evaluators"):
        assert merged.get(name) is None, f"talk gained {name}"


def test_the_wired_hub_case_hands_the_loop_the_hub_the_agent_holds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inst = _instance()
    inst.memory_hub = MagicMock(name="hub")
    inst.memory_hub.hippocampus = None
    calls, loop = _recorder()
    _launch(monkeypatch, tmp_path, inst, loop)
    [call] = calls
    agent = call["kwargs"].get("agent", call["args"][0] if call["args"] else None)
    assert call["kwargs"]["memory_hub"] is inst.memory_hub
    assert agent.memory._memory_hub is inst.memory_hub  # wired before the thread, on the launching thread


def test_the_per_run_objects_are_built_on_the_talk_thread(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import maxim.environment.filesystem_env as FE
    import maxim.runtime.bootstrap as BS

    built: dict[str, str] = {}
    real_env, real_mem, real_eng = FE.FileSystemEnv, BS.build_memory, BS.build_decision_engine

    def _env(*a: Any, **k: Any) -> Any:
        built["environment"] = threading.current_thread().name
        return real_env(*a, **k)

    def _mem(*a: Any, **k: Any) -> Any:
        built["memory"] = threading.current_thread().name
        return real_mem(*a, **k)

    def _eng(*a: Any, **k: Any) -> Any:
        built["decision_engine"] = threading.current_thread().name
        return real_eng(*a, **k)

    monkeypatch.setattr(FE, "FileSystemEnv", _env)
    monkeypatch.setattr(BS, "build_memory", _mem)
    monkeypatch.setattr(BS, "build_decision_engine", _eng)
    calls, loop = _recorder()
    _launch(monkeypatch, tmp_path, _instance(), loop)
    [call] = calls
    assert call["thread"] == "console.talk.t"
    assert built == {"environment": "console.talk.t", "memory": "console.talk.t", "decision_engine": "console.talk.t"}


def test_a_raise_inside_the_loop_call_reaches_the_talk_loops_logger_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def loop(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("loop blew up")

    caplog.set_level(logging.ERROR, logger="maxim.console.handle")
    _launch(monkeypatch, tmp_path, _instance(), loop)
    [rec] = [r for r in caplog.records if "talk loop failed" in r.getMessage()]
    assert rec.exc_info is not None and "loop blew up" in str(rec.exc_info[1])
